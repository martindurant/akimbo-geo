"""accessor.py — GeoAccessor sub-accessor for akimbo.

Registers ``series.ak.geo`` on all akimbo-enabled dataframe backends.

Usage
-----
    import akimbo.pandas   # register .ak
    import akimbo_geo      # register .ak.geo

    # series contains list<list<float>> Polygon geometries (any coord format)
    series.ak.geo.area()
    series.ak.geo.length()
    series.ak.geo.bounds()
    series.ak.geo.centroid()

    # WKB / WKT conversion
    series.ak.geo.to_wkb()
    wkb_series.ak.geo.from_wkb()

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
import awkward as ak

from akimbo.apply_tree import dec
from akimbo.mixin import EagerAccessor, LazyAccessor

from akimbo_geo import algorithms as alg
from akimbo_geo.match import (
    CoordKind,
    GeoLayout,
    extract_offsets_and_values,
    match_any_geom,
    match_line,
    match_multipolygon,
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


# ===========================================================================
# Op functions — called by dec() with a matched ak layout node (inmode="ak")
# Each receives an ak.contents.Content node and returns an ak.Array.layout.
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
# GeoAccessor class
# ===========================================================================

_METHODS = [
    "area",
    "area3",
    "length",
    "length2",
    "length3",
    "bounds",
    "centroid",
    "centroid_polygon",
    "intersects_bounds",
    "from_wkb",
    "to_wkb",
    "from_wkt",
    "to_wkt",
    "total_bounds",
]


class GeoAccessor:
    """Geometry operations on nested / var-length coordinate columns.

    Accepts any of the three GeoArrow coordinate representations plus the
    spatialpandas interleaved-flat convention, and two heuristic forms:

    Coordinate representations
    --------------------------
    - Interleaved flat:  ``list<float>`` — ``[x0,y0,x1,y1,...]``
    - Interleaved FSL:   ``list<FixedSizeList<float>[n]>`` — geoarrow native
    - Separated struct:  ``list<Struct<x:float, y:float, ...>>`` — geoarrow recommended
    - Heuristic:         ``list(2 * float)`` or ``2 * float`` at series level

    Geometry type is determined by List nesting depth above the coord leaf:
    0 = Point, 1 = Line/MultiPoint, 2 = Polygon/MultiLine, 3 = MultiPolygon.

    WKB / WKT are I/O-only formats; in-memory compute never boxes coordinates.
    """

    # --- Measurements -------------------------------------------------------

    length  = staticmethod(dec(_op_length,  match=match_line,         inmode="ak"))
    length2 = staticmethod(dec(_op_length2, match=match_polygon,      inmode="ak"))
    length3 = staticmethod(dec(_op_length3, match=match_multipolygon, inmode="ak"))

    area    = staticmethod(dec(_op_area,    match=match_polygon,      inmode="ak"))
    area3   = staticmethod(dec(_op_area3,   match=match_multipolygon, inmode="ak"))

    bounds          = staticmethod(dec(_op_bounds,   match=match_any_geom, inmode="ak"))
    centroid        = staticmethod(dec(_op_centroid1, match=match_line,    inmode="ak"))
    centroid_polygon = staticmethod(dec(_op_centroid2, match=match_polygon, inmode="ak"))

    # --- Spatial predicates -------------------------------------------------

    @staticmethod
    def intersects_bounds(arr, x0, y0, x1, y1):
        """Return bool array: does each depth-1 geometry intersect the bbox?

        Parameters
        ----------
        arr : ak.Array
        x0, y0, x1, y1 : float
            Query bounding box corners.
        """
        def _op(layout):
            return _op_intersects_bounds(layout, x0, y0, x1, y1)
        return dec(_op, match=match_line, inmode="ak")(arr)

    # --- Aggregate ----------------------------------------------------------

    @staticmethod
    def total_bounds(arr):
        """Return the aggregate (xmin, ymin, xmax, ymax) over the entire array.

        Returns
        -------
        tuple[float, float, float, float]
        """
        flat = ak.ravel(arr)
        # ravel gives the innermost float values; for separated struct this
        # may include z/m components, but for 2D data it's always x,y pairs.
        values = np.asarray(flat).astype(np.float64)
        return alg.total_bounds(values)

    # --- WKB / WKT I/O ------------------------------------------------------

    @staticmethod
    def from_wkb(arr):
        """Decode WKB bytestring column → interleaved coordinate list-of-floats.

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
        """Decode WKT string column → interleaved coordinate list-of-floats.

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
