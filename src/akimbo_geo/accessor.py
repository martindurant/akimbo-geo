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
from akimbo_geo._compat import array_module, is_gpu_array, require_shapely
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
    match_wkb,
    match_wkt,
)


# ===========================================================================
# Backend-dispatch helpers
# ===========================================================================

def _alg_gpu():
    """Lazily import algorithms_gpu to avoid importing numba.cuda on CPU-only systems."""
    from akimbo_geo import algorithms_gpu
    return algorithms_gpu


def _xp_full(n, fill, dtype, xp):
    """Create a filled array on the correct device."""
    return xp.full(n, fill, dtype=dtype)


def _xp_zeros(n, dtype, xp):
    return xp.zeros(n, dtype=dtype)


def _xp_empty(n, dtype, xp):
    return xp.empty(n, dtype=dtype)


# ===========================================================================
# Offset scaling helper
# ===========================================================================

def _scale_offsets(offsets, geo: GeoLayout):
    """Scale point-indexed offsets to float-indexed offsets for the kernels.

    The numba kernels expect offsets that index directly into the flat float
    buffer.  For INTERLEAVED_FLAT the Arrow offsets already do this.
    For FSL and STRUCT, offsets count coordinate *points*, so multiply by n_dims.
    Works on both numpy and cupy arrays (arithmetic is device-transparent).
    """
    if geo.coord_kind == CoordKind.INTERLEAVED_FLAT:
        return offsets
    return tuple(o * geo.n_dims for o in offsets)


# ===========================================================================
# Layout reconstruction helpers — direct ak.contents, no PyArrow round-trip
# ===========================================================================

def _wrap_in_list(values_array, offsets_array) -> ak.contents.Content:
    """Wrap a flat array in a ListOffsetArray using raw ak.contents constructors.

    This avoids serialising through PyArrow (``pa.array`` + ``pa.ListArray``)
    and works with both numpy and cupy arrays because ``ak.index.Index32``
    accepts any array-protocol object.
    """
    return ak.contents.ListOffsetArray(
        ak.index.Index32(offsets_array),
        ak.contents.NumpyArray(values_array),
    )


def _wrap_in_regular(values_array, size: int) -> ak.contents.Content:
    """Wrap a flat array in a RegularArray (FixedSizeList) of the given size."""
    return ak.contents.RegularArray(
        ak.contents.NumpyArray(values_array),
        size=size,
    )


def _rebuild_list1(values_flat, offsets0_float, geo: GeoLayout) -> ak.contents.Content:
    """Rebuild a depth-1 geometry layout from a (possibly new) flat values buffer.

    Uses direct ``ak.contents`` construction — no PyArrow round-trip.
    Works on both CPU (numpy) and GPU (cupy) arrays.
    """
    if geo.coord_kind == CoordKind.INTERLEAVED_FSL:
        pt_offsets = offsets0_float // geo.n_dims
        # Outer ListOffsetArray of RegularArray(size=n_dims)
        return ak.contents.ListOffsetArray(
            ak.index.Index32(pt_offsets),
            _wrap_in_regular(values_flat, geo.n_dims),
        )
    elif geo.coord_kind == CoordKind.SEPARATED_STRUCT:
        pt_offsets = offsets0_float // geo.n_dims
        xp = array_module(values_flat)
        dim_names = ["x", "y", "z", "m"][: geo.n_dims]
        contents = [
            ak.contents.NumpyArray(xp.ascontiguousarray(values_flat[d::geo.n_dims]))
            for d in range(geo.n_dims)
        ]
        return ak.contents.ListOffsetArray(
            ak.index.Index32(pt_offsets),
            ak.contents.RecordArray(contents, dim_names),
        )
    else:
        # INTERLEAVED_FLAT — plain flat list
        return _wrap_in_list(values_flat, offsets0_float)


def _rebuild_depth(values_flat, offsets_float, geo: GeoLayout) -> ak.contents.Content:
    """Rebuild an arbitrary-depth geometry layout from a new flat buffer.

    Uses direct ``ak.contents`` construction throughout — no PyArrow.
    Works on both CPU and GPU arrays.

    - depth-0 (Point): plain NumpyArray or RegularArray(size=n_dims)
    - n_dims == 2: INTERLEAVED_FLAT list<float>
    - n_dims >= 3: list<RegularArray(size=n_dims)> so n_dims is recoverable
    """
    # depth-0: no list wrapping
    if len(offsets_float) == 0:
        if geo.n_dims == 2:
            return ak.contents.NumpyArray(values_flat)
        else:
            return _wrap_in_regular(values_flat, geo.n_dims)

    if geo.n_dims == 2:
        if len(offsets_float) == 1:
            return _rebuild_list1(values_flat, offsets_float[0], geo)
        # depth-2+: innermost = flat list, then wrap outer levels
        inner = _wrap_in_list(values_flat, offsets_float[-1])
    else:
        # FixedSizeList[n_dims] leaf — n_dims is preserved on round-trip
        fsl = _wrap_in_regular(values_flat, geo.n_dims)
        if len(offsets_float) >= 1:
            pt_offsets = offsets_float[-1] // geo.n_dims
            inner = ak.contents.ListOffsetArray(
                ak.index.Index32(pt_offsets), fsl
            )
        else:
            inner = fsl

    # Wrap remaining outer levels
    for off in reversed(offsets_float[:-1]):
        inner = ak.contents.ListOffsetArray(
            ak.index.Index32(off),
            inner,
        )
    return inner


# ===========================================================================
# Op functions — all pure-numba operations
#
# Pattern: extract values + offsets (on whatever device the data is on),
# determine xp = array_module(values), allocate output on the same device,
# then dispatch to the CPU (alg.*) or GPU (alg_gpu.launch_*) kernel.
# ===========================================================================

# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _op_length(layout):
    """Length for depth-1 geometries (Line / Ring / MultiPoint)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_length_map1(values, off0, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.length_map1(values, off0, result, missing)
    return ak.contents.NumpyArray(result)


def _op_length2(layout):
    """Length for depth-2 geometries (MultiLine / Polygon perimeter)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_length_map2(values, off0, off1, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.length_map2(values, off0, off1, result, missing)
    return ak.contents.NumpyArray(result)


def _op_length3(layout):
    """Length for depth-3 geometries (MultiPolygon perimeter)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1, off2 = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_length_map3(values, off0, off1, off2, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.length_map3(values, off0, off1, off2, result, missing)
    return ak.contents.NumpyArray(result)


def _op_area(layout):
    """Area for depth-2 geometries (Polygon / MultiLine)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_area_map2(values, off0, off1, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.area_map2(values, off0, off1, result, missing)
    return ak.contents.NumpyArray(result)


def _op_area3(layout):
    """Area for depth-3 geometries (MultiPolygon)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1, off2 = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_area_map3(values, off0, off1, off2, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.area_map3(values, off0, off1, off2, result, missing)
    return ak.contents.NumpyArray(result)


def _op_bounds(layout):
    """Bounding box for any geometry — returns record{xmin,ymin,xmax,ymax}."""
    values, offsets, geo = extract_offsets_and_values(layout)
    scaled = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(scaled[0]) - 1
    result = xp.full((n, 4), xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        gpu = _alg_gpu()
        if len(scaled) == 1:
            gpu.launch_bounds_map1(values, scaled[0], result)
        elif len(scaled) == 2:
            gpu.launch_bounds_map2(values, scaled[0], scaled[1], result)
        else:
            gpu.launch_bounds_map3(values, scaled[0], scaled[1], scaled[2], result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        if len(scaled) == 1:
            alg.bounds_map1(values, scaled[0], result, missing)
        elif len(scaled) == 2:
            alg.bounds_map2(values, scaled[0], scaled[1], result, missing)
        else:
            alg.bounds_map3(values, scaled[0], scaled[1], scaled[2], result, missing)
    return ak.contents.RecordArray(
        [ak.contents.NumpyArray(xp.ascontiguousarray(result[:, 0])),
         ak.contents.NumpyArray(xp.ascontiguousarray(result[:, 1])),
         ak.contents.NumpyArray(xp.ascontiguousarray(result[:, 2])),
         ak.contents.NumpyArray(xp.ascontiguousarray(result[:, 3]))],
        ["xmin", "ymin", "xmax", "ymax"],
    )


def _op_centroid1(layout):
    """Centroid (mean of coords) for depth-1 geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result_x = xp.full(n, xp.nan, dtype=xp.float64)
    result_y = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_centroid_map1(values, off0, result_x, result_y)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.centroid_map1(values, off0, result_x, result_y, missing)
    return ak.contents.RecordArray(
        [ak.contents.NumpyArray(result_x), ak.contents.NumpyArray(result_y)],
        ["x", "y"],
    )


def _op_centroid2(layout):
    """Area-weighted centroid for depth-2 polygon geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result_x = xp.full(n, xp.nan, dtype=xp.float64)
    result_y = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_centroid_map2(values, off0, off1, result_x, result_y)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.centroid_map2(values, off0, off1, result_x, result_y, missing)
    return ak.contents.RecordArray(
        [ak.contents.NumpyArray(result_x), ak.contents.NumpyArray(result_y)],
        ["x", "y"],
    )


def _op_intersects_bounds(layout, x0, y0, x1, y1):
    """Boolean: does each depth-1 geometry intersect bounding box?"""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.zeros(n, dtype=xp.bool_)
    # intersects_bounds has no GPU kernel yet — fall back to CPU via .get()
    if is_gpu_array(values):
        from akimbo_geo._compat import gpu_array_to_numpy
        cpu_vals = gpu_array_to_numpy(values)
        cpu_off0 = gpu_array_to_numpy(off0)
        cpu_res  = np.zeros(n, dtype=np.bool_)
        missing  = np.zeros(n, dtype=np.bool_)
        alg.intersects_bounds_map1(float(x0), float(y0), float(x1), float(y1),
                                    cpu_vals, cpu_off0, cpu_res, missing)
        result = xp.asarray(cpu_res)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.intersects_bounds_map1(float(x0), float(y0), float(x1), float(y1),
                                    values, off0, result, missing)
    return ak.contents.NumpyArray(result)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def _op_count_coords(layout):
    """Number of coordinate points per depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.zeros(n, dtype=xp.int64)
    if is_gpu_array(values):
        _alg_gpu().launch_count_coords_map1(off0, geo.n_dims, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.count_coords_map1(off0, geo.n_dims, result, missing)
    return ak.contents.NumpyArray(result)


def _op_count_geoms(layout):
    """Number of sub-geometries per depth-2 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    outer_offsets = offsets[0]
    xp = array_module(outer_offsets)
    n = len(outer_offsets) - 1
    result = xp.zeros(n, dtype=xp.int64)
    if is_gpu_array(outer_offsets):
        _alg_gpu().launch_count_geoms_map2(outer_offsets, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.count_geoms_map2(outer_offsets, result, missing)
    return ak.contents.NumpyArray(result)


def _op_count_interior_rings(layout):
    """Number of interior rings (holes) per depth-2 polygon geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    outer_offsets = offsets[0]
    xp = array_module(outer_offsets)
    n = len(outer_offsets) - 1
    result = xp.zeros(n, dtype=xp.int64)
    if is_gpu_array(outer_offsets):
        _alg_gpu().launch_count_interior_rings_map2(outer_offsets, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.count_interior_rings_map2(outer_offsets, result, missing)
    return ak.contents.NumpyArray(result)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _op_is_closed(layout):
    """True if first coord == last coord for each depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.zeros(n, dtype=xp.bool_)
    if is_gpu_array(values):
        _alg_gpu().launch_is_closed_map1(values, off0, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.is_closed_map1(values, off0, result, missing)
    return ak.contents.NumpyArray(result)


def _op_is_ring(layout):
    """True if each depth-1 geometry is a closed ring with >= 4 points."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.zeros(n, dtype=xp.bool_)
    if is_gpu_array(values):
        _alg_gpu().launch_is_ring_map1(values, off0, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.is_ring_map1(values, off0, result, missing)
    return ak.contents.NumpyArray(result)


def _op_is_ccw(layout):
    """True if exterior ring is CCW for each depth-2 polygon geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, off1 = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(off0) - 1
    result = xp.zeros(n, dtype=xp.bool_)
    if is_gpu_array(values):
        _alg_gpu().launch_is_ccw_map2(values, off0, off1, result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        alg.is_ccw_map2(values, off0, off1, result, missing)
    return ak.contents.NumpyArray(result)


def _op_has_z(layout):
    """True if the geometry has a Z dimension (n_dims >= 3)."""
    geo_desc = _geo_layout_of(_unwrap(layout))
    n_dims = geo_desc.n_dims if geo_desc is not None else 2
    values, offsets, geo = extract_offsets_and_values(layout)
    xp = array_module(values)
    n = len(offsets[0]) - 1
    result = xp.full(n, n_dims >= 3, dtype=xp.bool_)
    return ak.contents.NumpyArray(result)


def _op_has_m(layout):
    """True if the geometry has an M dimension (n_dims >= 4)."""
    geo_desc = _geo_layout_of(_unwrap(layout))
    n_dims = geo_desc.n_dims if geo_desc is not None else 2
    values, offsets, geo = extract_offsets_and_values(layout)
    xp = array_module(values)
    n = len(offsets[0]) - 1
    result = xp.full(n, n_dims >= 4, dtype=xp.bool_)
    return ak.contents.NumpyArray(result)


# ---------------------------------------------------------------------------
# Coordinate extraction (Point / depth-0)
# ---------------------------------------------------------------------------

def _op_get_x(layout):
    """Extract x coordinate for each depth-0 Point geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    xp = array_module(values)
    n = len(values) // geo.n_dims
    result = xp.empty(n, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_get_x_map0(values, geo.n_dims, result)
    else:
        alg.get_x_map0(values, geo.n_dims, result)
    return ak.contents.NumpyArray(result)


def _op_get_y(layout):
    """Extract y coordinate for each depth-0 Point geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    xp = array_module(values)
    n = len(values) // geo.n_dims
    result = xp.empty(n, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_get_y_map0(values, geo.n_dims, result)
    else:
        alg.get_y_map0(values, geo.n_dims, result)
    return ak.contents.NumpyArray(result)


def _op_get_z(layout):
    """Extract z coordinate for each depth-0 Point geometry (NaN if 2D)."""
    values, offsets, geo = extract_offsets_and_values(layout)
    xp = array_module(values)
    n = len(values) // geo.n_dims
    result = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_get_z_map0(values, geo.n_dims, result)
    else:
        alg.get_z_map0(values, geo.n_dims, result)
    return ak.contents.NumpyArray(result)


def _op_get_coordinates(layout):
    """Return structured {x, y} record array for each depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n_floats = int(off0[-1])
    xs = values[:n_floats:2]
    ys = values[1:n_floats:2]
    pt_offsets = off0 // 2
    return ak.contents.ListOffsetArray(
        ak.index.Index32(pt_offsets),
        ak.contents.RecordArray(
            [ak.contents.NumpyArray(xs), ak.contents.NumpyArray(ys)],
            ["x", "y"],
        ),
    )


# ---------------------------------------------------------------------------
# Affine transformations
# ---------------------------------------------------------------------------

def _op_translate(layout, xoff, yoff):
    """Translate (add xoff, yoff to every coordinate) for depth-1 geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    result = xp.empty_like(values)
    if is_gpu_array(values):
        _alg_gpu().launch_translate_map(values, off0, float(xoff), float(yoff), result)
    else:
        alg.translate_map(values, off0, float(xoff), float(yoff), result)
    return _rebuild_list1(result, off0, geo)


def _op_scale(layout, xfact, yfact, origin):
    """Scale coordinates around an origin point for depth-1 geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    ox, oy = float(origin[0]), float(origin[1])
    result = xp.empty_like(values)
    if is_gpu_array(values):
        _alg_gpu().launch_scale_map(values, off0, float(xfact), float(yfact),
                                     ox, oy, result)
    else:
        alg.scale_map(values, off0, float(xfact), float(yfact), ox, oy, result)
    return _rebuild_list1(result, off0, geo)


def _op_affine_transform(layout, matrix):
    """Apply a 2D affine transform to depth-1 geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    a, b, d, e, xoff, yoff = (float(v) for v in matrix)
    result = xp.empty_like(values)
    if is_gpu_array(values):
        _alg_gpu().launch_affine_transform_map(values, off0, a, b, d, e,
                                                xoff, yoff, result)
    else:
        alg.affine_transform_map(values, off0, a, b, d, e, xoff, yoff, result)
    return _rebuild_list1(result, off0, geo)


# ---------------------------------------------------------------------------
# Coordinate manipulation
# ---------------------------------------------------------------------------

def _op_reverse(layout):
    """Reverse vertex order for each depth-1 geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    xp = array_module(values)
    result = xp.empty_like(values)
    if is_gpu_array(values):
        _alg_gpu().launch_reverse_map1(values, off0, result)
    else:
        alg.reverse_map1(values, off0, result)
    return _rebuild_list1(result, off0, geo)


def _op_force_2d(layout):
    """Drop Z/M coordinates, keeping only X and Y, for any geometry depth."""
    values, offsets, geo = extract_offsets_and_values(layout)
    if geo.n_dims == 2:
        return layout
    xp = array_module(values)
    n_pts = len(values) // geo.n_dims
    result_2d = xp.empty(n_pts * 2, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_force_2d_map(values, geo.n_dims, result_2d)
    else:
        alg.force_2d_map(values, geo.n_dims, result_2d)
    new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, 2, geo.list_depth)
    scaled = _scale_offsets(offsets, geo)
    new_offsets = tuple(o * 2 // geo.n_dims for o in scaled)
    return _rebuild_depth(result_2d, new_offsets, new_geo)


def _op_force_3d(layout, z_val):
    """Add a constant Z coordinate to 2D geometries."""
    values, offsets, geo = extract_offsets_and_values(layout)
    if geo.n_dims >= 3:
        return layout
    xp = array_module(values)
    n_pts = len(values) // geo.n_dims
    result_3d = xp.empty(n_pts * 3, dtype=xp.float64)
    if is_gpu_array(values):
        _alg_gpu().launch_force_3d_map(values, float(z_val), result_3d)
    else:
        alg.force_3d_map(values, float(z_val), result_3d)
    new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, 3, geo.list_depth)
    scaled = _scale_offsets(offsets, geo)
    new_offsets = tuple(o * 3 // geo.n_dims for o in scaled)
    return _rebuild_depth(result_3d, new_offsets, new_geo)


def _op_minimum_bounding_radius(layout):
    """Approximate minimum bounding radius (half bbox diagonal) per geometry."""
    values, offsets, geo = extract_offsets_and_values(layout)
    scaled = _scale_offsets(offsets, geo)
    xp = array_module(values)
    n = len(scaled[0]) - 1
    result = xp.full(n, xp.nan, dtype=xp.float64)
    if is_gpu_array(values):
        gpu = _alg_gpu()
        if len(scaled) == 1:
            gpu.launch_minimum_bounding_radius_map1(values, scaled[0], result)
        else:
            gpu.launch_minimum_bounding_radius_map2(values, scaled[0], scaled[1], result)
    else:
        missing = xp.zeros(n, dtype=xp.bool_)
        if len(scaled) == 1:
            alg.minimum_bounding_radius_map1(values, scaled[0], result, missing)
        else:
            alg.minimum_bounding_radius_map2(values, scaled[0], scaled[1], result, missing)
    return ak.contents.NumpyArray(result)


def _op_segmentize(layout, max_segment_length):
    """Insert intermediate points so no edge exceeds max_segment_length.

    segmentize uses a sequential two-pass algorithm that cannot run in a
    simple CUDA thread-per-geometry pattern (pass-1 must complete for all
    geometries before pass-2 can be sized and launched).  For GPU data we
    fall back to CPU by transferring the coordinate and offset arrays.
    """
    values, offsets, geo = extract_offsets_and_values(layout)
    off0, = _scale_offsets(offsets, geo)
    if is_gpu_array(values):
        from akimbo_geo._compat import gpu_array_to_numpy
        values = gpu_array_to_numpy(values)
        off0   = gpu_array_to_numpy(off0)
    max_len = float(max_segment_length)
    n = len(off0) - 1
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
# Op functions — Tier 1 (pure numba, no shapely)
# ===========================================================================

# ===========================================================================
# Shapely bridge helpers
# ===========================================================================

def _layout_to_shapely(layout):
    """Convert an ak layout node to a numpy array of shapely Geometry objects.

    Uses ``shapely.from_ragged_array`` with our flat coordinate buffer and
    the correctly ordered point-unit offsets — no WKB serialisation.

    Raises ``TypeError`` if the data lives on a GPU device, because GEOS
    operates on CPU memory only.  Call ``.to_backend('cpu')`` on the array
    first, or use a pure-numba operation instead.

    Our ``extract_offsets_and_values`` returns offsets in *outermost-first*
    order (off0 = geom→ring/line, off1 = ring/line→float, ...).  Shapely's
    ``from_ragged_array`` expects them in *innermost-first* order
    (ring→point, geom→ring) and in *point* units for the innermost level:

      depth-1: shapely (vertex_per_geom,)        ← (off0 // n_dims,)
      depth-2: shapely (ring_pts, geom_rings)    ← (off1 // n_dims, off0)
      depth-3: shapely (ring_pts, poly_rings, geom_polys) ← (off2//n, off1, off0)

    Geometry type is inferred from list_depth:
      depth 0 → Point
      depth 1 → LineString
      depth 2 → Polygon
      depth 3 → MultiPolygon
    """
    shapely = require_shapely()
    from shapely import GeometryType

    values, offsets, geo = extract_offsets_and_values(layout)

    # Shapely / GEOS runs on CPU only.  Reject GPU data with a clear message.
    if is_gpu_array(values):
        raise TypeError(
            "Shapely-backed operations require CPU data; this array lives on "
            "the GPU.  Transfer to CPU first with arr.to_backend('cpu'), or "
            "use a pure-numba operation (area, length, bounds, translate, …)."
        )

    coords = values.reshape(-1, geo.n_dims)  # (N_pts, n_dims) — zero-copy view

    # offsets are float-unit for FLAT, point-unit for FSL/STRUCT.
    if geo.coord_kind == CoordKind.INTERLEAVED_FLAT:
        # Scale the innermost offset (last in our tuple) to point units,
        # then reverse the tuple for shapely's expected order.
        if len(offsets) == 0:
            pt_offsets = None
        elif len(offsets) == 1:
            pt_offsets = (offsets[0] // geo.n_dims,)
        else:
            # Scale only the last (innermost = ring→float) offset
            scaled_inner = offsets[-1] // geo.n_dims
            # Reverse: shapely wants (inner_pt, ..., outer)
            pt_offsets = (scaled_inner,) + tuple(reversed(offsets[:-1]))
    else:
        # FSL / STRUCT offsets are already in point units; just reverse.
        if len(offsets) == 0:
            pt_offsets = None
        elif len(offsets) == 1:
            pt_offsets = offsets
        else:
            pt_offsets = (offsets[-1],) + tuple(reversed(offsets[:-1]))

    _depth_to_type = {
        0: GeometryType.POINT,
        1: GeometryType.LINESTRING,
        2: GeometryType.POLYGON,
        3: GeometryType.MULTIPOLYGON,
    }
    geom_type = _depth_to_type.get(geo.list_depth, GeometryType.GEOMETRYCOLLECTION)
    return shapely.from_ragged_array(geom_type, coords, pt_offsets)


def _shapely_to_layout(geoms, reference_geo: GeoLayout | None = None):
    """Convert a numpy array of shapely Geometry objects back to an ak layout.

    Uses ``shapely.to_ragged_array``.  The output is always INTERLEAVED_FLAT.

    ``to_ragged_array`` returns offsets in *innermost-first* order
    (ring→point, geom→ring, ...) while our layout convention is
    *outermost-first* (geom→ring, ring→float, ...).  We reverse the tuple
    and scale the first element (formerly innermost ring→point) to float units.

    Parameters
    ----------
    geoms : np.ndarray of shapely.Geometry
    reference_geo : optional GeoLayout from the source layout; used only to
        preserve n_dims when the GEOS operation keeps the same dimensionality.
    """
    shapely = require_shapely()

    include_z = (reference_geo is not None and reference_geo.n_dims >= 3) or None

    geom_type_id, coords, pt_offsets_shapely = shapely.to_ragged_array(
        geoms, include_z=include_z
    )

    # coords is (N_pts, n_dims); flatten to interleaved
    values = np.ascontiguousarray(coords).ravel().astype(np.float64)
    n_dims = coords.shape[1] if coords.ndim == 2 and coords.shape[0] > 0 else 2
    depth  = len(pt_offsets_shapely)  # 0=Point,1=Line,2=Poly,3=MPoly

    # Convert shapely's innermost-first offsets to our outermost-first float offsets.
    # shapely: (ring_pts, geom_rings, ...)  innermost first
    # ours:    (off0=outer, ..., offN=ring_floats)
    if depth == 0:
        float_offsets = ()
    elif depth == 1:
        # depth-1: shapely (vertex_per_geom_pts,) → ours (off0_floats,)
        float_offsets = (pt_offsets_shapely[0] * n_dims,)
    else:
        # depth≥2: reverse; scale only the first shapely offset (innermost ring pts)
        reversed_offsets = tuple(reversed(pt_offsets_shapely))
        # reversed_offsets[0] is now our outermost (geom→ring/poly, ring-unit)
        # reversed_offsets[-1] is the innermost (ring→pt, in point units) → scale
        float_offsets = reversed_offsets[:-1] + (reversed_offsets[-1] * n_dims,)

    new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, n_dims, depth)
    return _rebuild_depth(values, float_offsets, new_geo)


# ===========================================================================
# Op functions — WKB / WKT decode (dec()-based, match_wkb / match_wkt)
# ===========================================================================

def _op_from_wkb(layout):
    """Decode a matched bytestring layout node into a geometry layout.

    Called by ``dec()`` for each ``bytestring``-typed node it encounters.
    Receives the raw bytestring layout and returns the decoded coordinate
    layout (same outer structure, geometry leaf replaced by flat floats).
    """
    require_shapely()
    from akimbo_geo.convert import from_wkb as _from_wkb
    result = _from_wkb(ak.Array(layout))
    return result.layout


def _op_from_wkt(layout):
    """Decode a matched string layout node into a geometry layout.

    Called by ``dec()`` for each ``string``-typed node it encounters.
    """
    require_shapely()
    from akimbo_geo.convert import from_wkt as _from_wkt
    result = _from_wkt(ak.Array(layout))
    return result.layout


# ===========================================================================
# Op functions — Tier 2 unary scalar (shapely-backed)
# ===========================================================================

def _shapely_unary_bool(layout, fn_name):
    """Apply a unary boolean shapely ufunc."""
    shapely = require_shapely()
    geoms   = _layout_to_shapely(layout)
    result  = getattr(shapely, fn_name)(geoms).astype(np.bool_)
    return ak.Array(result).layout


def _shapely_unary_int(layout, fn_name):
    """Apply a unary integer-returning shapely ufunc."""
    shapely = require_shapely()
    geoms   = _layout_to_shapely(layout)
    result  = np.asarray(getattr(shapely, fn_name)(geoms), dtype=np.int32)
    return ak.Array(result).layout


def _shapely_unary_float(layout, fn_name):
    """Apply a unary float-returning shapely ufunc."""
    shapely = require_shapely()
    geoms   = _layout_to_shapely(layout)
    result  = np.asarray(getattr(shapely, fn_name)(geoms), dtype=np.float64)
    return ak.Array(result).layout


def _shapely_unary_str(layout, fn_name):
    """Apply a unary string-returning shapely ufunc (e.g. is_valid_reason)."""
    shapely = require_shapely()
    import pyarrow as pa
    geoms   = _layout_to_shapely(layout)
    raw     = getattr(shapely, fn_name)(geoms)
    # raw is a numpy object array of str/None
    return ak.from_arrow(pa.array(raw.tolist(), type=pa.large_string())).layout


def _op_is_valid(layout):
    return _shapely_unary_bool(layout, "is_valid")

def _op_is_simple(layout):
    return _shapely_unary_bool(layout, "is_simple")

def _op_is_empty(layout):
    return _shapely_unary_bool(layout, "is_empty")

def _op_is_valid_reason(layout):
    return _shapely_unary_str(layout, "is_valid_reason")

def _op_geom_type_id(layout):
    return _shapely_unary_int(layout, "get_type_id")

def _op_get_num_coordinates(layout):
    return _shapely_unary_int(layout, "get_num_coordinates")


# ===========================================================================
# Op functions — Tier 2 unary geometry (shapely → new geometry layout)
# ===========================================================================

def _shapely_unary_geom(layout, fn, **kwargs):
    """Apply a unary GEOS constructive operation; return a new geometry layout."""
    shapely  = require_shapely()
    geo_in   = _geo_layout_of(_unwrap(layout))
    geoms    = _layout_to_shapely(layout)
    result   = fn(geoms, **kwargs)
    return _shapely_to_layout(result, reference_geo=geo_in)


def _op_boundary(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.boundary)

def _op_convex_hull(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.convex_hull)

def _op_envelope(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.envelope)

def _op_make_valid(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.make_valid)

def _op_normalize(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.normalize)

def _op_extract_unique_points(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.extract_unique_points)

def _op_representative_point(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.point_on_surface)

def _op_minimum_bounding_circle(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.minimum_bounding_circle)

def _op_minimum_rotated_rectangle(layout):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.minimum_rotated_rectangle)

def _op_simplify(layout, tolerance, preserve_topology=True):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.simplify,
                                tolerance=float(tolerance),
                                preserve_topology=bool(preserve_topology))

def _op_buffer(layout, distance, quad_segs=16, cap_style="round",
               join_style="round", mitre_limit=5.0, single_sided=False):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.buffer,
                                distance=float(distance),
                                quad_segs=int(quad_segs),
                                cap_style=cap_style,
                                join_style=join_style,
                                mitre_limit=float(mitre_limit),
                                single_sided=bool(single_sided))

def _op_concave_hull(layout, ratio=0.0, allow_holes=False):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.concave_hull,
                                ratio=float(ratio),
                                allow_holes=bool(allow_holes))

def _op_offset_curve(layout, distance, quad_segs=16, join_style="round",
                     mitre_limit=5.0):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.offset_curve,
                                distance=float(distance),
                                quad_segs=int(quad_segs),
                                join_style=join_style,
                                mitre_limit=float(mitre_limit))

def _op_remove_repeated_points(layout, tolerance=0.0):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.remove_repeated_points,
                                tolerance=float(tolerance))

def _op_line_merge(layout, directed=False):
    shapely = require_shapely()
    return _shapely_unary_geom(layout, shapely.line_merge,
                                directed=bool(directed))


# ===========================================================================
# Op functions — Tier 2 linear referencing
# ===========================================================================

def _op_interpolate(layout, distance, normalized=False):
    """Return the point at the given distance along each line geometry."""
    shapely = require_shapely()
    geo_in  = _geo_layout_of(_unwrap(layout))
    geoms   = _layout_to_shapely(layout)
    result  = shapely.line_interpolate_point(geoms, float(distance),
                                              normalized=bool(normalized))
    return _shapely_to_layout(result, reference_geo=geo_in)


def _op_project(layout, other_layout, normalized=False):
    """Return the distance along each line to the nearest point of `other`."""
    shapely = require_shapely()
    geoms_a = _layout_to_shapely(layout)
    geoms_b = _layout_to_shapely(other_layout)
    result  = shapely.line_locate_point(geoms_a, geoms_b,
                                         normalized=bool(normalized))
    return ak.Array(np.asarray(result, dtype=np.float64)).layout


def _op_shared_paths(layout, other_layout):
    """Return shared paths between line geometries."""
    shapely = require_shapely()
    geo_in  = _geo_layout_of(_unwrap(layout))
    geoms_a = _layout_to_shapely(layout)
    geoms_b = _layout_to_shapely(other_layout)
    result  = shapely.shared_paths(geoms_a, geoms_b)
    return _shapely_to_layout(result, reference_geo=geo_in)


def _op_shortest_line(layout, other_layout):
    """Return the shortest line between each pair of geometries."""
    shapely = require_shapely()
    geo_in  = _geo_layout_of(_unwrap(layout))
    geoms_a = _layout_to_shapely(layout)
    geoms_b = _layout_to_shapely(other_layout)
    result  = shapely.shortest_line(geoms_a, geoms_b)
    return _shapely_to_layout(result, reference_geo=geo_in)


# ===========================================================================
# Op functions — Tier 2 binary predicates
# ===========================================================================

def _shapely_binary_bool(layout_a, layout_b, fn_name, **kwargs):
    """Apply a binary boolean shapely ufunc between two geometry layouts."""
    shapely = require_shapely()
    geoms_a = _layout_to_shapely(layout_a)
    geoms_b = _layout_to_shapely(layout_b)
    result  = getattr(shapely, fn_name)(geoms_a, geoms_b, **kwargs).astype(np.bool_)
    return ak.Array(result).layout


def _shapely_binary_float(layout_a, layout_b, fn_name, **kwargs):
    """Apply a binary float-returning shapely ufunc."""
    shapely = require_shapely()
    geoms_a = _layout_to_shapely(layout_a)
    geoms_b = _layout_to_shapely(layout_b)
    result  = np.asarray(getattr(shapely, fn_name)(geoms_a, geoms_b, **kwargs),
                          dtype=np.float64)
    return ak.Array(result).layout


# ===========================================================================
# Op functions — Tier 2 set-theoretic (binary geometry → new geometry)
# ===========================================================================

def _shapely_binary_geom(layout_a, layout_b, fn_name, **kwargs):
    """Apply a binary GEOS set-theoretic operation."""
    shapely = require_shapely()
    geo_in  = _geo_layout_of(_unwrap(layout_a))
    geoms_a = _layout_to_shapely(layout_a)
    geoms_b = _layout_to_shapely(layout_b)
    result  = getattr(shapely, fn_name)(geoms_a, geoms_b, **kwargs)
    return _shapely_to_layout(result, reference_geo=geo_in)


# ===========================================================================
# Op functions — Tier 2 aggregating reductions
# ===========================================================================

def _op_union_all(layout):
    """Union all geometries in the array into a single geometry."""
    shapely = require_shapely()
    geo_in  = _geo_layout_of(_unwrap(layout))
    geoms   = _layout_to_shapely(layout)
    result  = np.array([shapely.union_all(geoms)])
    return _shapely_to_layout(result, reference_geo=geo_in)


def _op_intersection_all(layout):
    """Intersection of all geometries in the array."""
    shapely = require_shapely()
    geo_in  = _geo_layout_of(_unwrap(layout))
    geoms   = _layout_to_shapely(layout)
    result  = np.array([shapely.intersection_all(geoms)])
    return _shapely_to_layout(result, reference_geo=geo_in)


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
    # Numba predicates
    "is_closed", "is_ring", "is_ccw", "has_z", "has_m",
    "minimum_bounding_radius",
    # Shapely scalar predicates
    "is_valid", "is_simple", "is_empty", "is_valid_reason", "geom_type_id",
    # Coordinate extraction
    "x", "y", "z", "get_coordinates",
    # Spatial predicates (numba)
    "intersects_bounds",
    # Affine transformations
    "translate", "scale", "affine_transform",
    # Coordinate manipulation (numba)
    "reverse", "force_2d", "force_3d", "segmentize", "orient_polygons",
    # Constructive (shapely)
    "boundary", "convex_hull", "envelope", "make_valid", "normalize",
    "extract_unique_points", "representative_point",
    "minimum_bounding_circle", "minimum_rotated_rectangle",
    "simplify", "buffer", "concave_hull", "offset_curve",
    "remove_repeated_points", "line_merge",
    # Linear referencing (shapely)
    "interpolate", "project", "shared_paths", "shortest_line",
    # Binary predicates (shapely)
    "contains", "within", "intersects", "crosses", "overlaps",
    "touches", "covers", "covered_by", "disjoint",
    "distance", "hausdorff_distance",
    # Set-theoretic (shapely)
    "difference", "intersection", "union", "symmetric_difference",
    "union_all", "intersection_all",
    # WKB / WKT
    "from_wkb", "to_wkb", "from_wkt", "to_wkt",
]


class GeoAccessor:
    """Geometry operations on nested / var-length coordinate columns.

    Accepts any of the three GeoArrow coordinate representations plus the
    spatialpandas interleaved-flat convention, and two heuristic forms.

    Numba operations (no extra dependencies): area, length, bounds, centroid,
    is_closed, is_ring, is_ccw, has_z, has_m, minimum_bounding_radius,
    count_*, translate, scale, affine_transform, reverse, force_2d, force_3d,
    segmentize, orient_polygons, x, y, z, get_coordinates, intersects_bounds.

    Shapely operations (require ``shapely>=2.0``): all topology and GEOS-
    based predicates, constructive operations, binary predicates, and set-
    theoretic operations.  Data is converted via ``from_ragged_array`` /
    ``to_ragged_array`` — no WKB serialisation.
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
    minimum_bounding_radius = staticmethod(dec(_op_minimum_bounding_radius,
                                               match=match_any_geom, inmode="ak"))

    # --- Counting -----------------------------------------------------------

    count_coordinates    = staticmethod(dec(_op_count_coords,          match=match_line,    inmode="ak"))
    count_geometries     = staticmethod(dec(_op_count_geoms,           match=match_polygon, inmode="ak"))
    count_interior_rings = staticmethod(dec(_op_count_interior_rings,  match=match_polygon, inmode="ak"))

    # --- Numba predicates ---------------------------------------------------

    is_closed = staticmethod(dec(_op_is_closed, match=match_line,     inmode="ak"))
    is_ring   = staticmethod(dec(_op_is_ring,   match=match_line,     inmode="ak"))
    is_ccw    = staticmethod(dec(_op_is_ccw,    match=match_polygon,  inmode="ak"))
    has_z     = staticmethod(dec(_op_has_z,     match=match_any_geom, inmode="ak"))
    has_m     = staticmethod(dec(_op_has_m,     match=match_any_geom, inmode="ak"))

    # --- Shapely scalar predicates ------------------------------------------

    is_valid       = staticmethod(dec(_op_is_valid,       match=match_any_geom, inmode="ak"))
    is_simple      = staticmethod(dec(_op_is_simple,      match=match_any_geom, inmode="ak"))
    is_empty       = staticmethod(dec(_op_is_empty,       match=match_any_geom, inmode="ak"))
    is_valid_reason = staticmethod(dec(_op_is_valid_reason, match=match_any_geom, inmode="ak"))
    geom_type_id   = staticmethod(dec(_op_geom_type_id,   match=match_any_geom, inmode="ak"))

    # --- Coordinate extraction (Points / depth-0) ---------------------------

    x               = staticmethod(dec(_op_get_x,           match=match_point, inmode="ak"))
    y               = staticmethod(dec(_op_get_y,           match=match_point, inmode="ak"))
    z               = staticmethod(dec(_op_get_z,           match=match_point, inmode="ak"))
    get_coordinates = staticmethod(dec(_op_get_coordinates, match=match_line,  inmode="ak"))

    # --- Spatial predicates (numba) -----------------------------------------

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
        """Translate every coordinate by (xoff, yoff)."""
        def _op(layout):
            return _op_translate(layout, xoff, yoff)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def scale(arr, xfact=1.0, yfact=1.0, origin=(0.0, 0.0)):
        """Scale coordinates by (xfact, yfact) around an origin point."""
        def _op(layout):
            return _op_scale(layout, xfact, yfact, origin)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def affine_transform(arr, matrix):
        """Apply a 2D affine transform [a, b, d, e, xoff, yoff] to every coordinate."""
        def _op(layout):
            return _op_affine_transform(layout, matrix)
        return dec(_op, match=match_line, inmode="ak")(arr)

    # --- Coordinate manipulation (numba) ------------------------------------

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
        """Promote 2D geometries to 3D by adding a constant Z value."""
        def _op(layout):
            return _op_force_3d(layout, z)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def segmentize(arr, max_segment_length):
        """Insert intermediate points so no edge exceeds max_segment_length."""
        def _op(layout):
            return _op_segmentize(layout, max_segment_length)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def orient_polygons(arr, exterior_cw=False):
        """Enforce ring orientation for polygon geometries."""
        def _op(layout):
            return _op_orient_polygons(layout, exterior_cw)
        return dec(_op, match=match_polygon, inmode="ak")(arr)

    # --- Constructive (shapely) ---------------------------------------------

    boundary                = staticmethod(dec(_op_boundary,                match=match_any_geom, inmode="ak"))
    convex_hull             = staticmethod(dec(_op_convex_hull,             match=match_any_geom, inmode="ak"))
    envelope                = staticmethod(dec(_op_envelope,                match=match_any_geom, inmode="ak"))
    make_valid              = staticmethod(dec(_op_make_valid,              match=match_any_geom, inmode="ak"))
    normalize               = staticmethod(dec(_op_normalize,               match=match_any_geom, inmode="ak"))
    extract_unique_points   = staticmethod(dec(_op_extract_unique_points,   match=match_any_geom, inmode="ak"))
    representative_point    = staticmethod(dec(_op_representative_point,    match=match_any_geom, inmode="ak"))
    minimum_bounding_circle = staticmethod(dec(_op_minimum_bounding_circle, match=match_any_geom, inmode="ak"))
    minimum_rotated_rectangle = staticmethod(dec(_op_minimum_rotated_rectangle,
                                                  match=match_any_geom, inmode="ak"))

    @staticmethod
    def simplify(arr, tolerance, preserve_topology=True):
        """Simplify geometries using the Douglas–Peucker algorithm.

        Requires ``shapely>=2.0``.

        Parameters
        ----------
        tolerance : float
        preserve_topology : bool, default True
        """
        def _op(layout):
            return _op_simplify(layout, tolerance, preserve_topology)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def buffer(arr, distance, quad_segs=16, cap_style="round",
               join_style="round", mitre_limit=5.0, single_sided=False):
        """Buffer geometries by a given distance.

        Requires ``shapely>=2.0``.

        Parameters
        ----------
        distance : float
        quad_segs : int, default 16
        cap_style, join_style : str, default "round"
        mitre_limit : float, default 5.0
        single_sided : bool, default False
        """
        def _op(layout):
            return _op_buffer(layout, distance, quad_segs, cap_style,
                               join_style, mitre_limit, single_sided)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def concave_hull(arr, ratio=0.0, allow_holes=False):
        """Compute the concave hull of each geometry.

        Requires ``shapely>=2.0``.

        Parameters
        ----------
        ratio : float, default 0.0 (convex hull)
        allow_holes : bool, default False
        """
        def _op(layout):
            return _op_concave_hull(layout, ratio, allow_holes)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def offset_curve(arr, distance, quad_segs=16, join_style="round",
                     mitre_limit=5.0):
        """Offset a line geometry to one side.

        Requires ``shapely>=2.0``.

        Parameters
        ----------
        distance : float
            Positive = left side, negative = right side.
        """
        def _op(layout):
            return _op_offset_curve(layout, distance, quad_segs,
                                     join_style, mitre_limit)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def remove_repeated_points(arr, tolerance=0.0):
        """Remove consecutive duplicate points from each geometry.

        Requires ``shapely>=2.0``.
        """
        def _op(layout):
            return _op_remove_repeated_points(layout, tolerance)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def line_merge(arr, directed=False):
        """Merge contiguous line segments into longer lines.

        Requires ``shapely>=2.0``.
        """
        def _op(layout):
            return _op_line_merge(layout, directed)
        return dec(_op, match=match_line, inmode="ak")(arr)

    # --- Linear referencing (shapely) ---------------------------------------

    @staticmethod
    def interpolate(arr, distance, normalized=False):
        """Return the point at ``distance`` along each line geometry.

        Requires ``shapely>=2.0``.

        Parameters
        ----------
        distance : float
        normalized : bool, default False
            If True, ``distance`` is a fraction [0, 1] of total length.
        """
        def _op(layout):
            return _op_interpolate(layout, distance, normalized)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def project(arr, other, normalized=False):
        """Return the distance along each line to the closest point of ``other``.

        Requires ``shapely>=2.0``.

        Parameters
        ----------
        other : ak.Array
            Array of geometries to project onto.
        normalized : bool, default False
        """
        def _op(layout_a):
            # other must be convertible to the same length array
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _op_project(layout_a, other_layout, normalized)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def shared_paths(arr, other):
        """Return shared paths between line geometries.

        Requires ``shapely>=2.0``.
        """
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _op_shared_paths(layout_a, other_layout)
        return dec(_op, match=match_line, inmode="ak")(arr)

    @staticmethod
    def shortest_line(arr, other):
        """Return the shortest line between each geometry and ``other``.

        Requires ``shapely>=2.0``.
        """
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _op_shortest_line(layout_a, other_layout)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    # --- Binary predicates (shapely) ----------------------------------------

    @staticmethod
    def contains(arr, other):
        """True if each geometry contains ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "contains")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def within(arr, other):
        """True if each geometry is within ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "within")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def intersects(arr, other):
        """True if each geometry intersects ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "intersects")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def crosses(arr, other):
        """True if each geometry crosses ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "crosses")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def overlaps(arr, other):
        """True if each geometry overlaps ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "overlaps")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def touches(arr, other):
        """True if each geometry touches ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "touches")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def covers(arr, other):
        """True if each geometry covers ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "covers")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def covered_by(arr, other):
        """True if each geometry is covered by ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "covered_by")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def disjoint(arr, other):
        """True if each geometry is disjoint from ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_bool(layout_a, other_layout, "disjoint")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def distance(arr, other):
        """Euclidean distance to ``other``. Requires ``shapely>=2.0``."""
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_float(layout_a, other_layout, "distance")
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def hausdorff_distance(arr, other, densify=None):
        """Hausdorff distance to ``other``. Requires ``shapely>=2.0``."""
        kw = {} if densify is None else {"densify": float(densify)}
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_float(layout_a, other_layout,
                                          "hausdorff_distance", **kw)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    # --- Set-theoretic (shapely) --------------------------------------------

    @staticmethod
    def difference(arr, other, grid_size=None):
        """Geometric difference with ``other``. Requires ``shapely>=2.0``."""
        kw = {} if grid_size is None else {"grid_size": float(grid_size)}
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_geom(layout_a, other_layout, "difference", **kw)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def intersection(arr, other, grid_size=None):
        """Geometric intersection with ``other``. Requires ``shapely>=2.0``."""
        kw = {} if grid_size is None else {"grid_size": float(grid_size)}
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_geom(layout_a, other_layout, "intersection", **kw)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def union(arr, other, grid_size=None):
        """Geometric union with ``other``. Requires ``shapely>=2.0``."""
        kw = {} if grid_size is None else {"grid_size": float(grid_size)}
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_geom(layout_a, other_layout, "union", **kw)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def symmetric_difference(arr, other, grid_size=None):
        """Symmetric difference with ``other``. Requires ``shapely>=2.0``."""
        kw = {} if grid_size is None else {"grid_size": float(grid_size)}
        def _op(layout_a):
            other_layout = ak.Array(other).layout if not hasattr(other, 'layout') \
                else other.layout
            return _shapely_binary_geom(layout_a, other_layout,
                                         "symmetric_difference", **kw)
        return dec(_op, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def union_all(arr):
        """Union all geometries into a single geometry. Requires ``shapely>=2.0``."""
        return dec(_op_union_all, match=match_any_geom, inmode="ak")(arr)

    @staticmethod
    def intersection_all(arr):
        """Intersection of all geometries. Requires ``shapely>=2.0``."""
        return dec(_op_intersection_all, match=match_any_geom, inmode="ak")(arr)

    # --- WKB / WKT I/O ------------------------------------------------------

    @staticmethod
    def from_wkb(arr):
        """Decode a WKB bytestring column into the canonical coordinate layout.

        Can be called directly on a bytes-typed Series — the ``dec()``
        tree-walker matches any ``bytestring``-typed layout node and calls
        ``shapely.from_wkb`` + ``to_ragged_array`` (both vectorised C, no
        Python loops) to decode it into flat interleaved coordinate arrays.

        Examples
        --------
        >>> # pandas
        >>> df["geometry"] = df["wkb_col"].ak.geo.from_wkb()
        >>> # polars
        >>> df = df.with_columns(df["wkb_col"].ak.geo.from_wkb().alias("geometry"))

        Requires ``shapely>=2.0``.
        """
        return dec(_op_from_wkb, match=match_wkb, inmode="ak")(arr)

    @staticmethod
    def to_wkb(arr):
        """Encode coordinate layout → WKB bytestrings.

        Requires ``shapely>=2.0``.
        """
        from akimbo_geo.convert import to_wkb as _to_wkb
        return _to_wkb(arr)

    @staticmethod
    def from_wkt(arr):
        """Decode a WKT string column into the canonical coordinate layout.

        Can be called directly on a string-typed Series — the ``dec()``
        tree-walker matches any ``string``-typed layout node and calls
        ``shapely.from_wkt`` + ``to_ragged_array`` (both vectorised C).

        Examples
        --------
        >>> df["geometry"] = df["wkt_col"].ak.geo.from_wkt()

        Requires ``shapely>=2.0``.
        """
        return dec(_op_from_wkt, match=match_wkt, inmode="ak")(arr)

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
