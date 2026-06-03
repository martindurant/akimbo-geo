"""accessor.py — GeoAccessor sub-accessor for akimbo.

Registers ``series.ak.geo`` on all akimbo-enabled dataframe backends.

Usage
-----
    import akimbo.pandas   # register .ak
    import akimbo_geo      # register .ak.geo

    # series contains list<list<float>> Polygon geometries
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
"""

from __future__ import annotations

import numpy as np
import awkward as ak

from akimbo.apply_tree import dec
from akimbo.mixin import EagerAccessor, LazyAccessor

from akimbo_geo import algorithms as alg
from akimbo_geo.match import (
    extract_offsets_and_values,
    match_any_geom,
    match_line,
    match_multipolygon,
    match_polygon,
)


# ===========================================================================
# Op functions — called by dec() with a matched ak layout node
# Each receives an ak.contents.Content node (inmode="ak") and returns an
# ak.Array or ak.contents.Content.
# ===========================================================================

def _op_length(layout):
    """Length for list<float> (Line / Ring / MultiPoint)."""
    values, (off0,) = extract_offsets_and_values(layout)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.length_map1(values, off0, result, missing)
    return ak.Array(result).layout


def _op_length2(layout):
    """Length for list<list<float>> (MultiLine / Polygon perimeter)."""
    values, (off0, off1) = extract_offsets_and_values(layout)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.length_map2(values, off0, off1, result, missing)
    return ak.Array(result).layout


def _op_length3(layout):
    """Length for list<list<list<float>>> (MultiPolygon perimeter)."""
    values, (off0, off1, off2) = extract_offsets_and_values(layout)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.length_map3(values, off0, off1, off2, result, missing)
    return ak.Array(result).layout


def _op_area(layout):
    """Area for list<list<float>> (Polygon)."""
    values, (off0, off1) = extract_offsets_and_values(layout)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.area_map2(values, off0, off1, result, missing)
    return ak.Array(result).layout


def _op_area3(layout):
    """Area for list<list<list<float>>> (MultiPolygon)."""
    values, (off0, off1, off2) = extract_offsets_and_values(layout)
    n = len(off0) - 1
    result = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.area_map3(values, off0, off1, off2, result, missing)
    return ak.Array(result).layout


def _op_bounds(layout):
    """Bounding box for any geometry — returns record{xmin,ymin,xmax,ymax}."""
    values, offsets = extract_offsets_and_values(layout)
    n = len(offsets[0]) - 1
    result = np.full((n, 4), np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    if len(offsets) == 1:
        alg.bounds_map1(values, offsets[0], result, missing)
    elif len(offsets) == 2:
        alg.bounds_map2(values, offsets[0], offsets[1], result, missing)
    else:
        alg.bounds_map3(values, offsets[0], offsets[1], offsets[2], result, missing)
    return ak.Array(
        {"xmin": result[:, 0], "ymin": result[:, 1],
         "xmax": result[:, 2], "ymax": result[:, 3]}
    ).layout


def _op_centroid1(layout):
    """Centroid (mean of coords) for list<float> geometries."""
    values, (off0,) = extract_offsets_and_values(layout)
    n = len(off0) - 1
    result_x = np.full(n, np.nan, dtype=np.float64)
    result_y = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.centroid_map1(values, off0, result_x, result_y, missing)
    return ak.Array({"x": result_x, "y": result_y}).layout


def _op_centroid2(layout):
    """Area-weighted centroid for list<list<float>> polygon geometries."""
    values, (off0, off1) = extract_offsets_and_values(layout)
    n = len(off0) - 1
    result_x = np.full(n, np.nan, dtype=np.float64)
    result_y = np.full(n, np.nan, dtype=np.float64)
    missing = np.zeros(n, dtype=np.bool_)
    alg.centroid_map2(values, off0, off1, result_x, result_y, missing)
    return ak.Array({"x": result_x, "y": result_y}).layout


def _op_intersects_bounds(layout, x0, y0, x1, y1):
    """Boolean: does each list<float> geometry intersect bounding box?"""
    values, (off0,) = extract_offsets_and_values(layout)
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
    "length",
    "bounds",
    "centroid",
    "intersects_bounds",
    "from_wkb",
    "to_wkb",
    "from_wkt",
    "to_wkt",
    "total_bounds",
]


class GeoAccessor:
    """Geometry operations on nested / var-length coordinate columns.

    The canonical storage format is **interleaved flat float lists**,
    matching the spatialpandas Arrow representation:

    - ``list<float>``              — Line, Ring, MultiPoint
    - ``list<list<float>>``        — Polygon, MultiLine
    - ``list<list<list<float>>>``  — MultiPolygon

    All coordinates are interleaved: ``[x0, y0, x1, y1, ...]``.

    WKB / WKT are supported as import/export formats only; the in-memory
    representation never boxes coordinates into geometry objects.
    """

    # --- Measurements -------------------------------------------------------

    # Length dispatches over all three nesting levels with separate ops so
    # that the correct numba kernel is called for each geometry type.
    length = staticmethod(dec(_op_length,  match=match_line,         inmode="ak"))
    # For 2/3-level nesting the user calls .length on the inner structures;
    # provide convenience aliases that match the deeper layouts too.
    length2 = staticmethod(dec(_op_length2, match=match_polygon,     inmode="ak"))
    length3 = staticmethod(dec(_op_length3, match=match_multipolygon, inmode="ak"))

    # Area only meaningful for polygon types.
    area  = staticmethod(dec(_op_area,  match=match_polygon,      inmode="ak"))
    area3 = staticmethod(dec(_op_area3, match=match_multipolygon, inmode="ak"))

    # Bounds and centroid work across all geometry types.
    bounds   = staticmethod(dec(_op_bounds,   match=match_any_geom, inmode="ak"))
    centroid = staticmethod(dec(_op_centroid1, match=match_line,    inmode="ak"))
    centroid_polygon = staticmethod(
        dec(_op_centroid2, match=match_polygon, inmode="ak")
    )

    # --- Spatial predicates -------------------------------------------------

    @staticmethod
    def intersects_bounds(arr, x0, y0, x1, y1):
        """Return bool array: does each list<float> geometry intersect the bbox?

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

        Parameters
        ----------
        arr : ak.Array

        Returns
        -------
        tuple[float, float, float, float]
        """
        # Flatten all coordinate values and call the scalar kernel.
        flat = ak.ravel(arr)
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
        """Encode interleaved coordinate list-of-floats → WKB bytestrings.

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
        """Encode interleaved coordinate list-of-floats → WKT strings.

        Requires ``shapely>=2.0``.
        """
        from akimbo_geo.convert import to_wkt as _to_wkt
        return _to_wkt(arr)

    def __dir__(self):
        return _METHODS


# ---------------------------------------------------------------------------
# Registration — same pattern as akimbo.strings / akimbo.datetimes
# ---------------------------------------------------------------------------
EagerAccessor.register_accessor("geo", GeoAccessor)
LazyAccessor.register_accessor("geo", GeoAccessor)
