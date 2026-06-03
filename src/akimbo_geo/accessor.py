"""accessor.py — GeoAccessor sub-accessor for akimbo.

Registers ``series.ak.geo`` on all akimbo-enabled dataframe backends.

Usage
-----
    import akimbo.pandas   # register .ak
    import akimbo_geo      # register .ak.geo

    series.ak.geo.area()
    series.ak.geo.length()
    series.ak.geo.bounds()
    series.ak.geo.centroid()
    series.ak.geo.translate(xoff=1.0, yoff=2.0)
    series.ak.geo.affine_transform(matrix=[a, b, d, e, xoff, yoff])
    series.ak.geo.to_wkb()

Each public method is a staticmethod built with ``akimbo.apply_tree.dec``,
which handles the full nested tree walk so that operations work at any depth
of nesting (list-of-geometries, list-of-list-of-geometries, mixed records,
etc.).

Coordinate representations
---------------------------
All three GeoArrow coordinate representations are accepted and automatically
normalised to interleaved flat float64 before the numba kernels run:

- Interleaved flat (spatialpandas): ``list<float>``
- Interleaved FixedSizeList (GeoArrow native): ``list<FixedSizeList[n]<float>>``
- Separated struct (GeoArrow recommended): ``list<Struct<x:float,y:float,...>>``

Heuristic inference also handles:
- ``series(list(2 * float))`` — each row is a list of 2-element coord arrays
- ``series(2 * float)`` — each row is a 2D coordinate pair (Point)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import awkward as ak

from akimbo.apply_tree import dec
from akimbo.mixin import EagerAccessor, LazyAccessor

from akimbo_geo import algorithms as alg
from akimbo_geo.match import (
    CoordKind,
    GeoLayout,
    _geo_layout_of,
    _unwrap,
    extract_offsets_and_values,
    match_any_geom,
    match_line,
    match_multipolygon,
    match_point,
    match_polygon,
)


# ===========================================================================
# Offset scaling helper
# ===========================================================================

def _scale_offsets(offsets, geo: GeoLayout):
    """Scale point-indexed offsets to float-indexed offsets for the kernels.

    The numba kernels from spatialpandas expect offsets that index directly
    into the flat float buffer.  For INTERLEAVED_FLAT the Arrow offsets
    already do this (one offset unit = one float).  For FSL and STRUCT, the
    Arrow list offsets count *coordinate points*, so we multiply by n_dims.
    """
    if geo.coord_kind == CoordKind.INTERLEAVED_FLAT:
        return offsets   # already in float units
    # FSL or STRUCT: offsets are in point units, scale to float units
    return tuple(o * geo.n_dims for o in offsets)


def _rebuild_list1(values_flat, offsets0_float, geo: GeoLayout) -> ak.contents.Content:
    """Rebuild a depth-1 geometry layout from a (possibly new) flat values buffer.

    When a kernel produces a new coordinate buffer with the same offsets
    (e.g. translate, scale, affine_transform, reverse), we re-wrap it in the
    same Arrow list structure so the output has the same type as the input.
    """
    # Offsets are always in float units here (already scaled)
    if geo.coord_kind in (CoordKind.INTERLEAVED_FSL, CoordKind.SEPARATED_STRUCT):
        # Convert back to point-unit offsets for the output Arrow type.
        pt_offsets = offsets0_float // geo.n_dims
        # Build as FixedSizeList if input was FSL
        if geo.coord_kind == CoordKind.INTERLEAVED_FSL:
            # Reshape flat buffer → (N_pts, n_dims) for FixedSizeList
            pts = values_flat.reshape(-1, geo.n_dims)
            return ak.from_arrow(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(values_flat), geo.n_dims
                )
            ).layout
        else:
            # SEPARATED_STRUCT: de-interleave back to struct
            arrays = [pa.array(values_flat[d::geo.n_dims]) for d in range(geo.n_dims)]
            dim_names = ["x", "y", "z", "m"][: geo.n_dims]
            pa_struct = pa.StructArray.from_arrays(arrays, names=dim_names)
            return ak.from_arrow(
                pa.ListArray.from_arrays(
                    pa.array(pt_offsets, type=pa.int32()),
                    pa_struct,
                )
            ).layout
    else:
        # INTERLEAVED_FLAT: wrap directly
        return ak.from_arrow(
            pa.ListArray.from_arrays(
                pa.array(offsets0_float, type=pa.int32()),
                pa.array(values_flat, type=pa.float64()),
            )
        ).layout


# ===========================================================================
# Op functions — existing measurements
# ===========================================================================

def _op_length(layout):
    """Length for depth-1 geometries (Line / Ring / MultiPoint)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.length_map1(values, off0, result, missing)
    return ak.Array(result).layout


def _op_length2(layout):
    """Length for depth-2 geometries (MultiLine / Polygon perimeter)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.length_map2(values, off0, off1, result, missing)
    return ak.Array(result).layout


def _op_length3(layout):
    """Length for depth-3 geometries (MultiPolygon perimeter)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1, off2 = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.length_map3(values, off0, off1, off2, result, missing)
    return ak.Array(result).layout


def _op_area(layout):
    """Area for depth-2 geometries (Polygon / MultiLine)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.area_map2(values, off0, off1, result, missing)
    return ak.Array(result).layout


def _op_area3(layout):
    """Area for depth-3 geometries (MultiPolygon)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1, off2 = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.area_map3(values, off0, off1, off2, result, missing)
    return ak.Array(result).layout


def _op_bounds(layout):
    """Bounding box for any geometry — returns record{xmin,ymin,xmax,ymax}."""
    values, offsets, geo = extract_offsets_and_values(layout)
    scaled = _scale_offsets(offsets, geo)
    n = len(scaled[0]) - 1
    result = np.full((n, 4), np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    if len(scaled) == 1:
        alg.bounds_map1(values, scaled[0], result, missing)
    elif len(scaled) == 2:
        alg.bounds_map2(values, scaled[0], scaled[1], result, missing)
    else:
        alg.bounds_map3(values, scaled[0], scaled[1], scaled[2], result, missing)
    return ak.Array(
        {"xmin": result[:, 0], "ymin": result[:, 1],
         "xmax": result[:, 2], "ymax": result[:, 3]}
    ).layout


def _op_centroid1(layout):
    """Centroid (mean of coords) for depth-1 geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result_x = np.full(n, np.nan, dtype=np.float64)
    result_y = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.centroid_map1(values, off0, result_x, result_y, missing)
    return ak.Array({"x": result_x, "y": result_y}).layout


def _op_centroid2(layout):
    """Area-weighted centroid for depth-2 polygon geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result_x = np.full(n, np.nan, dtype=np.float64)
    result_y = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.centroid_map2(values, off0, off1, result_x, result_y, missing)
    return ak.Array({"x": result_x, "y": result_y}).layout


def _op_intersects_bounds(layout, x0, y0, x1, y1):
    """Boolean: does each depth-1 geometry intersect bounding box?"""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.zeros(n, dtype=np.bool_)
    missing = np.zeros(n, dtype=np.bool_)
    alg.intersects_bounds_map1(
        float(x0), float(y0), float(x1), float(y1),
        values, off0, result, missing
    )
    return ak.Array(result).layout


# ===========================================================================
# Op functions — NEW: counting
# ===========================================================================

def _op_count_coords(layout):
    """Number of coordinate points per depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.zeros(n, dtype=np.int64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.count_coords_map1(off0, geo.n_dims, result, missing)
    return ak.Array(result).layout


def _op_count_geoms(layout):
    """Number of sub-geometries per depth-2 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, _off1 = _scale_offsets(offsets, geo)
    # off0 is the outer-level offset; count sub-geoms from the unscaled off0
    # (point-unit counts are what we want, divide by n_dims only for values)
    # Actually: the count of sub-geometries comes from the outer Arrow offsets
    # regardless of coord kind — it counts items in the outer list.
    outer_offsets = offsets[0]  # unscaled point/item units
    n = len(outer_offsets) - 1
    result = np.zeros(n, dtype=np.int64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.count_geoms_map2(outer_offsets, result, missing)
    return ak.Array(result).layout


def _op_count_interior_rings(layout):
    """Number of interior rings (holes) per depth-2 polygon geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    outer_offsets = offsets[0]  # item-unit outer offsets (ring counts)
    n = len(outer_offsets) - 1
    result = np.zeros(n, dtype=np.int64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.count_interior_rings_map2(outer_offsets, result, missing)
    return ak.Array(result).layout


# ===========================================================================
# Op functions — NEW: predicates
# ===========================================================================

def _op_is_closed(layout):
    """True if first coord == last coord for each depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.zeros(n, dtype=np.bool_)
    missing = np.zeros(n, dtype=np.bool_)
    alg.is_closed_map1(values, off0, result, missing)
    return ak.Array(result).layout


def _op_is_ccw(layout):
    """True if exterior ring is CCW for each depth-2 polygon geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    n = len(off0) - 1
    result = np.zeros(n, dtype=np.bool_)
    missing = np.zeros(n, dtype=np.bool_)
    alg.is_ccw_map2(values, off0, off1, result, missing)
    return ak.Array(result).layout


def _op_has_z(layout):
    """True if the geometry has a Z dimension (n_dims >= 3)."""
    geo = _geo_layout_of(_unwrap(layout))
    n_dims = geo.n_dims if geo is not None else 2
    values, offsets, geo = extract_offsets_and_values(layout)
    n = len(offsets[0]) - 1
    result = np.full(n, n_dims >= 3, dtype=np.bool_)
    return ak.Array(result).layout


def _op_has_m(layout):
    """True if the geometry has an M dimension (n_dims >= 4)."""
    geo = _geo_layout_of(_unwrap(layout))
    n_dims = geo.n_dims if geo is not None else 2
    values, offsets, geo2 = extract_offsets_and_values(layout)
    n = len(offsets[0]) - 1
    result = np.full(n, n_dims >= 4, dtype=np.bool_)
    return ak.Array(result).layout


# ===========================================================================
# Op functions — NEW: coordinate extraction (Point / depth-0)
# ===========================================================================

def _op_get_x(layout):
    """Extract x coordinate for each depth-0 Point geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    # depth-0: no offsets, values IS the flat point buffer
    n = len(values) // geo.n_dims
    result = np.empty(n, dtype=np.float64)
    alg.get_x_map0(values, geo.n_dims, result)
    return ak.Array(result).layout


def _op_get_y(layout):
    """Extract y coordinate for each depth-0 Point geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    n = len(values) // geo.n_dims
    result = np.empty(n, dtype=np.float64)
    alg.get_y_map0(values, geo.n_dims, result)
    return ak.Array(result).layout


def _op_get_z(layout):
    """Extract z coordinate for each depth-0 Point geometry (NaN if 2D)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    n = len(values) // geo.n_dims
    result = np.full(n, np.nan, dtype=np.float64)
    alg.get_z_map0(values, geo.n_dims, result)
    return ak.Array(result).layout


def _op_get_coordinates(layout):
    """Return structured {x, y} record array for each depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    n_floats = int(off0[-1])
    n_pts = n_floats // 2  # always stride-2 in float units
    xs = values[:n_floats:2]
    ys = values[1:n_floats:2]
    # Build a variable-length list of {x, y} records matching the input nesting
    # Produce a ListArray of struct: offsets are in point units
    pt_offsets = off0 // 2
    return ak.from_arrow(
        pa.ListArray.from_arrays(
            pa.array(pt_offsets.astype(np.int32), type=pa.int32()),
            pa.StructArray.from_arrays(
                [pa.array(xs), pa.array(ys)], names=["x", "y"]
            ),
        )
    ).layout


# ===========================================================================
# Op functions — NEW: affine transformations
# ===========================================================================

def _op_translate(layout, xoff, yoff):
    """Translate (add xoff, yoff to every coordinate) for depth-1 geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    result = np.empty_like(values)
    alg.translate_map(values, off0, float(xoff), float(yoff), result)
    return _rebuild_list1(result, off0, geo)


def _op_scale(layout, xfact, yfact, origin):
    """Scale coordinates around an origin point for depth-1 geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    ox, oy = float(origin[0]), float(origin[1])
    result = np.empty_like(values)
    alg.scale_map(values, off0, float(xfact), float(yfact), ox, oy, result)
    return _rebuild_list1(result, off0, geo)


def _op_affine_transform(layout, matrix):
    """Apply a 2D affine transform to depth-1 geometries.

    matrix : sequence of 6 floats [a, b, d, e, xoff, yoff] where:
        x' = a*x + b*y + xoff
        y' = d*x + e*y + yoff
    """
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    a, b, d, e, xoff, yoff = (float(v) for v in matrix)
    result = np.empty_like(values)
    alg.affine_transform_map(values, off0, a, b, d, e, xoff, yoff, result)
    return _rebuild_list1(result, off0, geo)


# ===========================================================================
# Op functions — NEW: coordinate manipulation
# ===========================================================================

def _op_reverse(layout):
    """Reverse vertex order for each depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    result = np.empty_like(values)
    alg.reverse_map1(values, off0, result)
    return _rebuild_list1(result, off0, geo)


def _op_force_2d(layout):
    """Drop Z/M coordinates, keeping only X and Y, for any geometry depth."""
    values, offsets, geo = extract_offsets_and_values(layout)
    if geo.n_dims == 2:
        return layout  # already 2D — return unchanged
    n_pts = len(values) // geo.n_dims
    result_2d = np.empty(n_pts * 2, dtype=np.float64)
    alg.force_2d_map(values, geo.n_dims, result_2d)
    # Rebuild offsets in float-2D units
    new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, 2, geo.list_depth)
    # Offsets must be rescaled: old float offsets / n_dims * 2
    scaled = _scale_offsets(offsets, geo)  # float-unit offsets for old n_dims
    new_offsets = tuple(o * 2 // geo.n_dims for o in scaled)
    # Reconstruct by wrapping new flat 2D buffer in the same list structure
    return _rebuild_depth(result_2d, new_offsets, new_geo)


def _op_force_3d(layout, z_val):
    """Add a constant Z coordinate to 2D geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    if geo.n_dims >= 3:
        return layout  # already has Z — return unchanged
    n_pts = len(values) // geo.n_dims
    result_3d = np.empty(n_pts * 3, dtype=np.float64)
    alg.force_3d_map(values, float(z_val), result_3d)
    new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, 3, geo.list_depth)
    scaled = _scale_offsets(offsets, geo)
    new_offsets = tuple(o * 3 // geo.n_dims for o in scaled)
    return _rebuild_depth(result_3d, new_offsets, new_geo)


def _rebuild_depth(values_flat, offsets_float, geo: GeoLayout) -> ak.contents.Content:
    """Rebuild an arbitrary-depth geometry layout from a new flat buffer.

    When n_dims == 2 the output is INTERLEAVED_FLAT (plain list<float>).
    When n_dims >= 3 the output wraps coordinates in FixedSizeList[n_dims]
    so that the matcher can recover the correct n_dims on the round-trip.
    """
    if geo.n_dims == 2:
        # INTERLEAVED_FLAT — innermost list is plain floats
        if len(offsets_float) == 1:
            return _rebuild_list1(values_flat, offsets_float[0], geo)
        # depth-2+: wrap from inside out
        inner_arr = ak.from_arrow(
            pa.ListArray.from_arrays(
                pa.array(offsets_float[-1].astype(np.int32), type=pa.int32()),
                pa.array(values_flat, type=pa.float64()),
            )
        )
    else:
        # FixedSizeList[n_dims] coordinate leaf so n_dims is recoverable
        n_pts = len(values_flat) // geo.n_dims
        inner_arr = ak.from_arrow(
            pa.FixedSizeListArray.from_arrays(
                pa.array(values_flat, type=pa.float64()), geo.n_dims
            )
        )
        # Wrap in list using the innermost offset (point-unit)
        if len(offsets_float) >= 1:
            pt_offsets = offsets_float[-1] // geo.n_dims
            inner_arr = ak.from_arrow(
                pa.ListArray.from_arrays(
                    pa.array(pt_offsets.astype(np.int32), type=pa.int32()),
                    ak.to_arrow(inner_arr, extensionarray=False),
                )
            )

    # Middle levels (for depth > 1)
    for off in reversed(offsets_float[:-1]):
        inner_arr = ak.from_arrow(
            pa.ListArray.from_arrays(
                pa.array(off.astype(np.int32), type=pa.int32()),
                ak.to_arrow(inner_arr, extensionarray=False),
            )
        )
    return inner_arr.layout


def _op_segmentize(layout, max_segment_length):
    """Insert intermediate points so no edge exceeds max_segment_length."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    max_len = float(max_segment_length)
    n = len(off0) - 1
    # First pass: count output points
    n_out_pts = int(alg.segmentize_count(values, off0, max_len))
    result = np.empty(n_out_pts * 2, dtype=np.float64)
    new_offsets = np.zeros(n + 1, dtype=np.int64)
    alg.segmentize_map1(values, off0, max_len, result, new_offsets)
    return _rebuild_list1(result, new_offsets.astype(np.int32), geo)


def _op_orient_polygons(layout, exterior_cw):
    """Enforce ring orientation for depth-2 polygon geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    # Work on a copy so we don't mutate the source buffer
    values_copy = values.copy()
    alg.orient_polygons_map2(values_copy, off0, off1, bool(exterior_cw))
    # Rebuild with the same offsets
    return _rebuild_depth(values_copy, (off0, off1), geo)


# ===========================================================================
# GeoAccessor class
# ===========================================================================

_METHODS = [
    # Measurements
    "area", "area3",
    "length", "length2", "length3",
    "bounds", "total_bounds",
    "centroid", "centroid_polygon",
    # Counting
    "count_coordinates", "count_geometries", "count_interior_rings",
    # Predicates
    "is_closed", "is_ccw", "has_z", "has_m",
    # Coordinate extraction
    "x", "y", "z", "get_coordinates",
    # Spatial predicates
    "intersects_bounds",
    # Affine transformations
    "translate", "scale", "affine_transform",
    # Coordinate manipulation
    "reverse", "force_2d", "force_3d", "segmentize", "orient_polygons",
    # WKB / WKT
    "from_wkb", "to_wkb", "from_wkt", "to_wkt",
]


class GeoAccessor:
    """Geometry operations on nested / var-length coordinate columns.

    Accepts any of the three GeoArrow coordinate representations plus the
    spatialpandas interleaved-flat convention, and two heuristic forms.

    WKB / WKT are I/O-only formats; in-memory compute never boxes coordinates.
    """

    # --- Measurements -------------------------------------------------------

    length   = staticmethod(dec(_op_length,  match=match_line,         inmode="ak"))
    length2  = staticmethod(dec(_op_length2, match=match_polygon,      inmode="ak"))
    length3  = staticmethod(dec(_op_length3, match=match_multipolygon, inmode="ak"))
    area     = staticmethod(dec(_op_area,    match=match_polygon,      inmode="ak"))
    area3    = staticmethod(dec(_op_area3,   match=match_multipolygon, inmode="ak"))
    bounds   = staticmethod(dec(_op_bounds,  match=match_any_geom,     inmode="ak"))
    centroid         = staticmethod(dec(_op_centroid1, match=match_line,    inmode="ak"))
    centroid_polygon = staticmethod(dec(_op_centroid2, match=match_polygon, inmode="ak"))

    # --- Counting -----------------------------------------------------------

    count_coordinates    = staticmethod(dec(_op_count_coords,          match=match_line,    inmode="ak"))
    count_geometries     = staticmethod(dec(_op_count_geoms,           match=match_polygon, inmode="ak"))
    count_interior_rings = staticmethod(dec(_op_count_interior_rings,  match=match_polygon, inmode="ak"))

    # --- Predicates ---------------------------------------------------------

    is_closed = staticmethod(dec(_op_is_closed, match=match_line,    inmode="ak"))
    is_ccw    = staticmethod(dec(_op_is_ccw,    match=match_polygon, inmode="ak"))
    has_z     = staticmethod(dec(_op_has_z,     match=match_any_geom, inmode="ak"))
    has_m     = staticmethod(dec(_op_has_m,     match=match_any_geom, inmode="ak"))

    # --- Coordinate extraction (Points / depth-0) ---------------------------

    x              = staticmethod(dec(_op_get_x,           match=match_point, inmode="ak"))
    y              = staticmethod(dec(_op_get_y,           match=match_point, inmode="ak"))
    z              = staticmethod(dec(_op_get_z,           match=match_point, inmode="ak"))
    get_coordinates = staticmethod(dec(_op_get_coordinates, match=match_line,  inmode="ak"))

    # --- Spatial predicates -------------------------------------------------

    @staticmethod
    def intersects_bounds(arr, x0, y0, x1, y1):
        """Return bool array: does each depth-1 geometry intersect the bbox?"""
        def _op(layout):
            return _op_intersects_bounds(layout, x0, y0, x1, y1)
        return dec(_op, match=match_line, inmode="ak")(arr)

    # --- Aggregate ----------------------------------------------------------

    @staticmethod
    def total_bounds(arr):
        """Return the aggregate (xmin, ymin, xmax, ymax) over the entire array."""
        flat = ak.ravel(arr)
        values = np.asarray(flat).astype(np.float64)
        return alg.total_bounds(values)

    # --- Affine transformations ---------------------------------------------

    @staticmethod
    def translate(arr, xoff=0.0, yoff=0.0):
        """Translate every coordinate by (xoff, yoff).

        Parameters
        ----------
        arr : ak.Array
        xoff, yoff : float
        """
        def _op(layout):
            return _op_translate(layout, xoff, yoff)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def scale(arr, xfact=1.0, yfact=1.0, origin=(0.0, 0.0)):
        """Scale coordinates by (xfact, yfact) around an origin point.

        Parameters
        ----------
        arr : ak.Array
        xfact, yfact : float
            Scale factors.
        origin : (float, float)
            Centre of scaling.  Default (0, 0).
        """
        def _op(layout):
            return _op_scale(layout, xfact, yfact, origin)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def affine_transform(arr, matrix):
        """Apply a 2D affine transform to every coordinate.

        Parameters
        ----------
        arr : ak.Array
        matrix : sequence of 6 floats [a, b, d, e, xoff, yoff]
            x' = a*x + b*y + xoff
            y' = d*x + e*y + yoff

        This matches the geopandas / shapely 6-element convention.
        """
        def _op(layout):
            return _op_affine_transform(layout, matrix)
        return dec(_op, match=match_line, inmode="ak")(arr)

    # --- Coordinate manipulation --------------------------------------------

    @staticmethod
    def reverse(arr):
        """Reverse the vertex order of each geometry."""
        return dec(_op_reverse, match=match_line, inmode="ak")(arr)

    @staticmethod
    def force_2d(arr):
        """Drop Z/M coordinates, keeping only X and Y."""
        return dec(_op_force_2d, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def force_3d(arr, z=0.0):
        """Promote 2D geometries to 3D by adding a constant Z value.

        Parameters
        ----------
        z : float, default 0.0
        """
        def _op(layout):
            return _op_force_3d(layout, z)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def segmentize(arr, max_segment_length):
        """Insert intermediate points so no edge exceeds max_segment_length.

        Parameters
        ----------
        max_segment_length : float
        """
        def _op(layout):
            return _op_segmentize(layout, max_segment_length)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def orient_polygons(arr, exterior_cw=False):
        """Enforce ring orientation for polygon geometries.

        Parameters
        ----------
        exterior_cw : bool, default False
            If False (geopandas default), exterior rings are CCW and holes CW.
            If True, exterior rings are CW and holes CCW.
        """
        def _op(layout):
            return _op_orient_polygons(layout, exterior_cw)
        return dec(_op, match=match_polygon, inmode="ak")(arr)

    # --- WKB / WKT I/O ------------------------------------------------------

    @staticmethod
    def from_wkb(arr):
        """Decode WKB bytestring column → canonical coordinate layout.

        Requires ``shapely>=2.0``.
        """
        from akimbo_geo.convert import from_wkb as _from_wkb
        return _from_wkb(arr)

    @staticmethod
    def to_wkb(arr):
        """Encode coordinate layout → WKB bytestrings.

        Requires ``shapely>=2.0``.
        """
        from akimbo_geo.convert import to_wkb as _to_wkb
        return _to_wkb(arr)

    @staticmethod
    def from_wkt(arr):
        """Decode WKT string column → canonical coordinate layout.

        Requires ``shapely>=2.0``.
        """
        from akimbo_geo.convert import from_wkt as _from_wkt
        return _from_wkt(arr)

    @staticmethod
    def to_wkt(arr):
        """Encode coordinate layout → WKT strings.

        Requires ``shapely>=2.0``.
        """
        from akimbo_geo.convert import to_wkt as _to_wkt
        return _to_wkt(arr)

    def __dir__(self):
        return _METHODS


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
EagerAccessor.register_accessor("geo", GeoAccessor)
LazyAccessor.register_accessor("geo", GeoAccessor)
