"""convert.py — WKB/WKT I/O and framework interoperability bridges.

All functions convert *to or from* the canonical internal representation:
interleaved flat float lists stored in nested Arrow list arrays, exactly
as spatialpandas stores them.

WKB / WKT are the *only* ways geometry objects (shapely, WKB bytes) enter
or leave the system.  In-memory compute always operates on the raw numeric
buffers.

Optional dependencies
---------------------
- ``shapely>=2.0`` — required for from_wkb / to_wkb / from_wkt / to_wkt
- ``spatialpandas``— required for from_spatialpandas / to_spatialpandas
- ``geopandas``    — required for from_geopandas / to_geopandas
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import awkward as ak


# ===========================================================================
# Internal helpers
# ===========================================================================

def _require_shapely():
    try:
        import shapely
        return shapely
    except ImportError as exc:
        raise ImportError(
            "shapely>=2.0 is required for WKB/WKT conversion. "
            "Install it with: pip install 'akimbo-geo[convert]'"
        ) from exc


def _shapely_to_arrow_list(geoms):
    """Convert a 1-D array of shapely geometries to a pa.Array of the
    appropriate nested list<float> Arrow type.

    Dispatches on geometry type:
      Point           → list<float64>  length 2
      LineString/Ring → list<float64>
      Polygon         → list<list<float64>>
      MultiPoint      → list<float64>
      MultiLineString → list<list<float64>>
      MultiPolygon    → list<list<list<float64>>>
    """
    import shapely
    import shapely.geometry as sg

    if len(geoms) == 0:
        return pa.array([], type=pa.list_(pa.float64()))

    # Use shapely's vectorised coordinate access where possible.
    # Determine the dominant type from the first non-null entry.
    first = None
    for g in geoms:
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            first = g
            break

    if first is None:
        return pa.array([None] * len(geoms), type=pa.list_(pa.float64()))

    def _coords_flat(g):
        """Flat interleaved coords for a single geometry."""
        if g is None or (isinstance(g, float) and np.isnan(g)):
            return None
        return np.asarray(g.coords).ravel().tolist()

    if isinstance(first, (sg.Point,)):
        # list<float64>
        rows = [_coords_flat(g) for g in geoms]
        return pa.array(rows, type=pa.list_(pa.float64()))

    elif isinstance(first, (sg.LineString, sg.LinearRing)):
        # list<float64>
        rows = [_coords_flat(g) if g is not None else None for g in geoms]
        return pa.array(rows, type=pa.list_(pa.float64()))

    elif isinstance(first, sg.MultiPoint):
        # list<float64>  (flat interleaved, same as LineString)
        def _mp_coords(g):
            if g is None:
                return None
            return np.array([[pt.x, pt.y] for pt in g.geoms], dtype=np.float64).ravel().tolist()
        rows = [_mp_coords(g) for g in geoms]
        return pa.array(rows, type=pa.list_(pa.float64()))

    elif isinstance(first, sg.Polygon):
        # list<list<float64>>  [exterior, hole0, hole1, ...]
        def _poly_coords(g):
            if g is None:
                return None
            rings = [np.asarray(g.exterior.coords).ravel().tolist()]
            for hole in g.interiors:
                rings.append(np.asarray(hole.coords).ravel().tolist())
            return rings
        rows = [_poly_coords(g) for g in geoms]
        return pa.array(rows, type=pa.list_(pa.list_(pa.float64())))

    elif isinstance(first, sg.MultiLineString):
        # list<list<float64>>
        def _mls_coords(g):
            if g is None:
                return None
            return [np.asarray(line.coords).ravel().tolist() for line in g.geoms]
        rows = [_mls_coords(g) for g in geoms]
        return pa.array(rows, type=pa.list_(pa.list_(pa.float64())))

    elif isinstance(first, sg.MultiPolygon):
        # list<list<list<float64>>>
        def _mpoly_coords(g):
            if g is None:
                return None
            polys = []
            for poly in g.geoms:
                rings = [np.asarray(poly.exterior.coords).ravel().tolist()]
                for hole in poly.interiors:
                    rings.append(np.asarray(hole.coords).ravel().tolist())
                polys.append(rings)
            return polys
        rows = [_mpoly_coords(g) for g in geoms]
        return pa.array(rows, type=pa.list_(pa.list_(pa.list_(pa.float64()))))

    else:
        raise TypeError(
            f"Unsupported geometry type for conversion: {type(first).__name__}"
        )


def _arrow_list_to_shapely(arr):
    """Convert an ak.Array of nested list<float> to shapely geometries.

    Parameters
    ----------
    arr : ak.Array
        Nested list-of-float geometry array.

    Returns
    -------
    numpy object array of shapely geometry instances.
    """
    import shapely.geometry as sg

    py_list = ak.to_list(arr)

    def _depth_of(item):
        """Determine nesting depth by inspecting the first non-None element."""
        if item is None:
            return None
        if isinstance(item, list):
            if len(item) == 0:
                return 1
            inner = _depth_of(item[0])
            if inner is None:
                return 1
            return 1 + inner
        return 0  # leaf (float)

    # Find depth from first non-None entry
    depth = None
    for v in py_list:
        if v is not None:
            depth = _depth_of(v)
            break
    if depth is None:
        return np.array([None] * len(py_list), dtype=object)

    def _to_geom(val):
        if val is None:
            return None
        if depth == 1:
            # list<float>: LineString or Point
            coords = np.array(val, dtype=np.float64).reshape(-1, 2)
            if len(coords) == 1:
                return sg.Point(coords[0])
            return sg.LineString(coords)
        elif depth == 2:
            # list<list<float>>: Polygon (first ring = exterior, rest = holes)
            if len(val) == 0:
                return None
            rings = [np.array(r, dtype=np.float64).reshape(-1, 2) for r in val]
            return sg.Polygon(shell=rings[0], holes=rings[1:])
        else:
            # depth == 3: list<list<list<float>>>: MultiPolygon
            polys = []
            for poly_rings in val:
                rings = [np.array(r, dtype=np.float64).reshape(-1, 2) for r in poly_rings]
                polys.append(sg.Polygon(shell=rings[0], holes=rings[1:]))
            return sg.MultiPolygon(polys)

    return np.array([_to_geom(v) for v in py_list], dtype=object)


# ===========================================================================
# WKB
# ===========================================================================

def from_wkb(arr: ak.Array) -> ak.Array:
    """Decode a column of WKB bytestrings into the canonical coordinate layout.

    The input ``arr`` should contain WKB bytes (an ``ak.Array`` of
    bytestrings, or a nested structure thereof).  The output has the same
    outer structure but with each leaf WKB value replaced by the appropriate
    nested list-of-float structure.

    Requires ``shapely>=2.0``.
    """
    shapely = _require_shapely()

    # Convert to Python list of bytes objects for shapely
    py = ak.to_list(arr)

    def _decode(item):
        if item is None:
            return None
        if isinstance(item, (bytes, bytearray)):
            g = shapely.from_wkb(item)
            pa_arr = _shapely_to_arrow_list(np.array([g], dtype=object))
            return pa_arr.to_pylist()[0]
        if isinstance(item, list):
            return [_decode(x) for x in item]
        raise TypeError(f"Expected bytes, got {type(item)}")

    decoded = [_decode(x) for x in py]
    # Rebuild as an awkward array via pyarrow
    # Infer the arrow type from the first non-null decoded entry.
    first = next((x for x in decoded if x is not None), None)
    if first is None:
        return ak.from_arrow(pa.array(decoded, type=pa.list_(pa.float64())))
    pa_arr = pa.array(decoded)
    return ak.from_arrow(pa_arr)


def to_wkb(arr: ak.Array) -> ak.Array:
    """Encode canonical coordinate layout → WKB bytestrings.

    Requires ``shapely>=2.0``.
    """
    shapely = _require_shapely()

    geoms  = _arrow_list_to_shapely(arr)
    wkb    = np.array(
        [shapely.to_wkb(g) if g is not None else None for g in geoms],
        dtype=object,
    )
    pa_out = pa.array(wkb.tolist(), type=pa.large_binary())
    return ak.from_arrow(pa_out)


# ===========================================================================
# WKT
# ===========================================================================

def from_wkt(arr: ak.Array) -> ak.Array:
    """Decode a column of WKT strings into the canonical coordinate layout.

    Requires ``shapely>=2.0``.
    """
    shapely = _require_shapely()

    py = ak.to_list(arr)

    def _decode(item):
        if item is None:
            return None
        if isinstance(item, str):
            g = shapely.from_wkt(item)
            pa_arr = _shapely_to_arrow_list(np.array([g], dtype=object))
            return pa_arr.to_pylist()[0]
        if isinstance(item, list):
            return [_decode(x) for x in item]
        raise TypeError(f"Expected str, got {type(item)}")

    decoded = [_decode(x) for x in py]
    pa_arr  = pa.array(decoded)
    return ak.from_arrow(pa_arr)


def to_wkt(arr: ak.Array) -> ak.Array:
    """Encode canonical coordinate layout → WKT strings.

    Requires ``shapely>=2.0``.
    """
    shapely = _require_shapely()

    geoms  = _arrow_list_to_shapely(arr)
    wkt    = np.array(
        [shapely.to_wkt(g) if g is not None else None for g in geoms],
        dtype=object,
    )
    pa_out = pa.array(wkt.tolist(), type=pa.large_string())
    return ak.from_arrow(pa_out)


# ===========================================================================
# spatialpandas bridge
# ===========================================================================

def _nesting_depth(arr: ak.Array) -> int:
    """Return the number of list-nesting levels in arr's layout."""
    from akimbo_geo.match import _unwrap
    depth = 0
    layout = _unwrap(arr.layout)
    while layout.is_list:
        depth += 1
        layout = _unwrap(layout.content)
    return depth


def from_spatialpandas(geom_array) -> ak.Array:
    """Convert a spatialpandas ``GeometryListArray`` to an ``ak.Array``.

    This is a **zero-copy** operation: spatialpandas stores geometry data in
    ``geom_array.data`` which is already a ``pa.Array``.  We wrap it directly
    with ``ak.from_arrow``.

    Parameters
    ----------
    geom_array : spatialpandas GeometryListArray (or GeoSeries.values)

    Returns
    -------
    ak.Array with the same nested list<float> structure.
    """
    # GeometryArray.data is a pa.Array (or pa.ChunkedArray after concat)
    pa_data = geom_array.data
    if isinstance(pa_data, pa.ChunkedArray):
        pa_data = pa_data.combine_chunks()
    return ak.from_arrow(pa_data)


def to_spatialpandas(arr: ak.Array, geom_class=None):
    """Convert an ``ak.Array`` of geometry coordinates to a spatialpandas array.

    Parameters
    ----------
    arr : ak.Array
        Must contain a nested list<float> layout matching one of the
        spatialpandas geometry types.
    geom_class : spatialpandas GeometryListArray subclass, optional
        If None, the class is inferred from the nesting depth of ``arr``.

    Returns
    -------
    spatialpandas GeometryListArray subclass instance.
    """
    try:
        import spatialpandas.geometry as spg
    except ImportError as exc:
        raise ImportError(
            "spatialpandas is required for to_spatialpandas. "
            "Install it with: pip install 'akimbo-geo[spatialpandas]'"
        ) from exc

    # Use extensionarray=False to get a plain pa.Array that spatialpandas
    # can consume without the awkward extension type wrapper.
    pa_arr = ak.to_arrow(arr, extensionarray=False)

    if geom_class is None:
        depth = _nesting_depth(arr)
        _depth_to_class = {
            1: spg.LineArray,
            2: spg.PolygonArray,
            3: spg.MultiPolygonArray,
        }
        geom_class = _depth_to_class.get(depth)
        if geom_class is None:
            raise ValueError(
                f"Cannot infer spatialpandas type from nesting depth {depth}. "
                "Pass geom_class explicitly."
            )

    return geom_class(pa_arr)


# ===========================================================================
# geopandas bridge
# ===========================================================================

def from_geopandas(geoseries) -> ak.Array:
    """Convert a geopandas ``GeoSeries`` to an ``ak.Array`` of coordinates.

    Requires ``geopandas`` and ``shapely>=2.0``.
    """
    _require_shapely()
    try:
        import geopandas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "geopandas is required. "
            "Install it with: pip install 'akimbo-geo[geopandas]'"
        ) from exc

    geoms  = np.asarray(geoseries, dtype=object)
    pa_arr = _shapely_to_arrow_list(geoms)
    return ak.from_arrow(pa_arr)


def to_geopandas(arr: ak.Array):
    """Convert an ``ak.Array`` of geometry coordinates to a geopandas ``GeoSeries``.

    Requires ``geopandas`` and ``shapely>=2.0``.
    """
    _require_shapely()
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required. "
            "Install it with: pip install 'akimbo-geo[geopandas]'"
        ) from exc

    pa_arr = ak.to_arrow(arr)
    geoms  = _arrow_list_to_shapely(pa_arr)
    return gpd.GeoSeries(geoms)
