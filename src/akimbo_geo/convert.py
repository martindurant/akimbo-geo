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

from akimbo_geo._compat import require_shapely, require_geopandas


# ===========================================================================
# Internal helpers
# ===========================================================================

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

    elif isinstance(first, (sg.Polygon, sg.MultiPolygon)):
        # list<list<float64>> — each element is one ring
        # MultiPolygon is flattened: all rings from all sub-polygons concatenated.
        # Exterior rings are stored first, then holes, preserving ring structure
        # within each sub-polygon.  This is consistent with how spatialpandas
        # stores MultiPolygon at depth-2 (all rings in one flat ring list).
        def _poly_rings(g):
            if g is None:
                return None
            if isinstance(g, sg.Polygon):
                rings = [np.asarray(g.exterior.coords).ravel().tolist()]
                for hole in g.interiors:
                    rings.append(np.asarray(hole.coords).ravel().tolist())
                return rings
            else:  # MultiPolygon — flatten all rings
                rings = []
                for poly in g.geoms:
                    rings.append(np.asarray(poly.exterior.coords).ravel().tolist())
                    for hole in poly.interiors:
                        rings.append(np.asarray(hole.coords).ravel().tolist())
                return rings
        rows = [_poly_rings(g) for g in geoms]
        return pa.array(rows, type=pa.list_(pa.list_(pa.float64())))

    elif isinstance(first, sg.MultiLineString):
        # list<list<float64>>
        def _mls_coords(g):
            if g is None:
                return None
            return [np.asarray(line.coords).ravel().tolist() for line in g.geoms]
        rows = [_mls_coords(g) for g in geoms]
        return pa.array(rows, type=pa.list_(pa.list_(pa.float64())))

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
    """Decode a flat column of WKB bytestrings into the canonical coordinate layout.

    Uses ``shapely.from_wkb`` (vectorised C, no Python loops) to parse the
    bytes, then ``shapely.to_ragged_array`` to extract the flat coordinate
    buffers and offset arrays that akimbo-geo works on natively.

    The column is assumed to be **homogeneous** — all rows contain the same
    geometry type (or a promotable mix such as Polygon + MultiPolygon).
    Shapely automatically promotes mixed types to the highest-dimensional
    type (e.g. Polygon + MultiPolygon → MultiPolygon at depth-3).  The
    GeoParquet ``geometry_types`` metadata field, if present, documents the
    expected types.

    Parameters
    ----------
    arr : ak.Array
        1-D array of WKB bytestrings (``large_binary`` or ``binary`` Arrow
        type).  Nulls are preserved as missing values in the output.

    Returns
    -------
    ak.Array
        Nested list-of-float geometry array in the canonical interleaved
        coordinate format.  Depth depends on geometry type:
        - Point       → depth-0  (bare float buffer)
        - Line/Ring   → depth-1  list<float>
        - Polygon     → depth-2  list<list<float>>
        - MultiPolygon→ depth-3  list<list<list<float>>>

    Requires ``shapely>=2.0``.
    """
    shapely = require_shapely()

    # shapely.from_wkb is fully vectorised (C extension, GIL released).
    # We pass a numpy object array of bytes objects — no Python iteration.
    wkb_np = np.array(ak.to_list(arr), dtype=object)
    geoms   = shapely.from_wkb(wkb_np)

    # Guard: if all geometries are null, to_ragged_array raises.
    # Return a flat null float array in that case.
    non_null = geoms[geoms != None]  # noqa: E711
    if len(non_null) == 0:
        return ak.from_arrow(pa.array([None] * len(geoms), type=pa.list_(pa.float64())))

    # to_ragged_array extracts coords + offset arrays at the C level.
    geom_type_id, coords, pt_offsets = shapely.to_ragged_array(geoms)

    # coords is (N_pts, n_dims) float64 — ravel to interleaved flat buffer
    values    = np.ascontiguousarray(coords).ravel().astype(np.float64)
    n_dims    = coords.shape[1] if coords.ndim == 2 and len(coords) > 0 else 2
    list_depth = len(pt_offsets)   # 0=Point, 1=Line, 2=Polygon, 3=MultiPolygon

    # pt_offsets from to_ragged_array are in *innermost-first* (shapely) order.
    # _rebuild_depth expects *outermost-first* float-unit offsets.
    # Reverse the tuple and scale the first element (innermost ring→point) to
    # float units — identical to the logic in _shapely_to_layout().
    from akimbo_geo.accessor import _rebuild_depth, GeoLayout, CoordKind
    if list_depth == 0:
        float_offsets = ()
    elif list_depth == 1:
        float_offsets = (pt_offsets[0] * n_dims,)
    else:
        reversed_offsets = tuple(reversed(pt_offsets))
        float_offsets = reversed_offsets[:-1] + (reversed_offsets[-1] * n_dims,)

    new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, n_dims, list_depth)
    layout  = _rebuild_depth(values, float_offsets, new_geo)
    return ak.Array(layout)


def to_wkb(arr: ak.Array) -> ak.Array:
    """Encode canonical coordinate layout → WKB bytestrings.

    Uses ``shapely.to_wkb`` (vectorised C).  The input must be a flat
    (1-D) geometry array.

    Requires ``shapely>=2.0``.
    """
    shapely = require_shapely()
    from akimbo_geo.accessor import _layout_to_shapely
    geoms  = _layout_to_shapely(arr.layout)
    wkb_np = shapely.to_wkb(geoms)
    return ak.from_arrow(pa.array(wkb_np.tolist(), type=pa.large_binary()))


# ===========================================================================
# WKT
# ===========================================================================

def from_wkt(arr: ak.Array) -> ak.Array:
    """Decode a flat column of WKT strings into the canonical coordinate layout.

    Uses ``shapely.from_wkt`` (vectorised C) then ``to_ragged_array``, with
    the same homogeneity assumption as :func:`from_wkb`.

    Requires ``shapely>=2.0``.
    """
    shapely = require_shapely()

    wkt_np = np.array(ak.to_list(arr), dtype=object)
    geoms   = shapely.from_wkt(wkt_np)

    non_null = geoms[geoms != None]  # noqa: E711
    if len(non_null) == 0:
        return ak.from_arrow(pa.array([None] * len(geoms), type=pa.list_(pa.float64())))

    geom_type_id, coords, pt_offsets = shapely.to_ragged_array(geoms)

    values    = np.ascontiguousarray(coords).ravel().astype(np.float64)
    n_dims    = coords.shape[1] if coords.ndim == 2 and len(coords) > 0 else 2
    list_depth = len(pt_offsets)

    from akimbo_geo.accessor import _rebuild_depth, GeoLayout, CoordKind
    if list_depth == 0:
        float_offsets = ()
    elif list_depth == 1:
        float_offsets = (pt_offsets[0] * n_dims,)
    else:
        reversed_offsets = tuple(reversed(pt_offsets))
        float_offsets = reversed_offsets[:-1] + (reversed_offsets[-1] * n_dims,)

    new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, n_dims, list_depth)
    layout  = _rebuild_depth(values, float_offsets, new_geo)
    return ak.Array(layout)


def to_wkt(arr: ak.Array) -> ak.Array:
    """Encode canonical coordinate layout → WKT strings.

    Requires ``shapely>=2.0``.
    """
    shapely = require_shapely()

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
    from akimbo_geo._compat import require_spatialpandas
    spg = require_spatialpandas().geometry

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
    require_shapely()
    require_geopandas()

    geoms  = np.asarray(geoseries, dtype=object)
    pa_arr = _shapely_to_arrow_list(geoms)
    return ak.from_arrow(pa_arr)


def to_geopandas(arr: ak.Array):
    """Convert an ``ak.Array`` of geometry coordinates to a geopandas ``GeoSeries``.

    Requires ``geopandas`` and ``shapely>=2.0``.
    """
    require_shapely()
    gpd = require_geopandas()

    geoms = _arrow_list_to_shapely(arr)
    return gpd.GeoSeries(geoms)


# ===========================================================================
# Parquet I/O — geometry-aware loading
# ===========================================================================

def read_parquet(
    path: str,
    geometry_col: str = "geometry",
    columns=None,
    **kwargs,
) -> "pd.DataFrame":
    """Read a (Geo)Parquet file into a plain pandas DataFrame.

    Handles both encoding conventions automatically:

    **Native geoarrow** (``encoding`` in ``["geoarrow.point", "geoarrow.linestring",
    "geoarrow.polygon", "geoarrow.multipolygon", ...]``):
        The geometry column is already stored as nested list-of-float arrays.
        It is loaded as a ``pd.ArrowDtype`` column and returned as-is — no
        conversion, no shapely, zero overhead.

    **WKB** (``encoding == "WKB"`` or no ``geo`` metadata):
        The geometry column is decoded from WKB bytestrings using
        ``shapely.from_wkb`` + ``shapely.to_ragged_array`` (both vectorised
        C calls) and stored as a ``pd.ArrowDtype`` column of interleaved
        coordinate arrays.  Shapely handles mixed geometry types (e.g.
        Polygon + MultiPolygon) by promoting to the highest type.

    Parameters
    ----------
    path : str
        Path to a Parquet file.
    geometry_col : str, default ``"geometry"``
        Name of the geometry column in the file.
    columns : list[str] | None
        Columns to read; if None, all columns are read.
    **kwargs
        Passed through to ``pyarrow.parquet.read_table``.

    Returns
    -------
    pd.DataFrame
        A plain pandas DataFrame with ``pd.ArrowDtype`` columns.  The
        geometry column contains akimbo-geo's native interleaved coordinate
        format and supports ``.ak.geo`` directly — no further conversion
        step is needed.

    Examples
    --------
    >>> import akimbo.pandas, akimbo_geo
    >>> from akimbo_geo.convert import read_parquet
    >>>
    >>> # Works for native geoarrow AND WKB parquet files:
    >>> df = read_parquet("buildings.parquet")
    >>> df["area"] = df["geometry"].ak.geo.area()
    """
    import json
    import pyarrow.parquet as pq
    import pandas as pd

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    # Read all requested columns
    if columns is not None and geometry_col not in columns:
        columns = list(columns) + [geometry_col]
    table = pq.read_table(path, columns=columns, **kwargs)

    # Detect geometry encoding from GeoParquet metadata
    raw_meta = schema.metadata or {}
    geo_meta = raw_meta.get(b"geo")
    encoding = None
    if geo_meta:
        try:
            geo = json.loads(geo_meta)
            col_info = geo.get("columns", {}).get(geometry_col, {})
            encoding = col_info.get("encoding", "WKB")
        except (json.JSONDecodeError, AttributeError):
            encoding = "WKB"

    geom_col = table.column(geometry_col)

    if encoding and encoding.lower() != "wkb":
        # Native geoarrow: column is already list-of-float arrays.
        # Just wrap as ArrowDtype — zero conversion.
        geom_series = pd.array(geom_col, dtype=pd.ArrowDtype(geom_col.type))
    else:
        # WKB: decode via shapely (vectorised C) → flat coord arrays
        require_shapely()
        import shapely as _shapely
        from akimbo_geo.accessor import _rebuild_depth, GeoLayout, CoordKind

        wkb_np    = np.array([x.as_py() for x in geom_col], dtype=object)
        geoms     = _shapely.from_wkb(wkb_np)
        _, coords, pt_offsets = _shapely.to_ragged_array(geoms)

        values     = np.ascontiguousarray(coords).ravel().astype(np.float64)
        n_dims     = coords.shape[1] if coords.ndim == 2 and len(coords) > 0 else 2
        list_depth = len(pt_offsets)

        if list_depth == 0:
            float_offsets = ()
        elif list_depth == 1:
            float_offsets = (pt_offsets[0] * n_dims,)
        else:
            rev = tuple(reversed(pt_offsets))
            float_offsets = rev[:-1] + (rev[-1] * n_dims,)

        new_geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, n_dims, list_depth)
        layout  = _rebuild_depth(values, float_offsets, new_geo)
        geom_ak = ak.Array(layout)
        geom_pa = ak.to_arrow(geom_ak, extensionarray=False)
        geom_series = pd.array(geom_pa, dtype=pd.ArrowDtype(geom_pa.type))

    # Build the output DataFrame with ArrowDtype columns
    other_cols = [c for c in table.column_names if c != geometry_col]
    df = table.select(other_cols).to_pandas(
        types_mapper=pd.ArrowDtype,
    )
    df[geometry_col] = geom_series
    # Move geometry column to position it had in the original table
    orig_pos = table.column_names.index(geometry_col)
    cols = list(df.columns)
    cols.remove(geometry_col)
    cols.insert(orig_pos, geometry_col)
    return df[cols]
