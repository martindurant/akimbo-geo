"""match.py — layout-matching predicates for geometry node identification.

We support three distinct coordinate leaf representations, corresponding to
the GeoArrow specification and spatialpandas, plus heuristic inference:

1. **Interleaved flat** (spatialpandas style):
   ``list<float>`` — flat buffer ``[x0,y0,x1,y1,...]``.
   The outer list provides per-geometry offsets; the flat values are all
   coordinate components interleaved.  This is what spatialpandas stores
   and what the existing numba kernels operate on.

2. **Interleaved FixedSizeList** (GeoArrow native interleaved):
   ``FixedSizeList<double>[n_dim]`` — each element is one coordinate point
   stored as a fixed-size list of n_dim floats.  Arrow type name:
   ``list<item: fixed_size_list<xy: double>[2]>``.

3. **Separated struct** (GeoArrow native separated, recommended):
   ``Struct<x:double, y:double[, z:double, m:double]>`` — each element is
   one coordinate point stored as a record with named float fields.

In all three cases, **geometry type** (LineString / Polygon / MultiPolygon)
is determined by the number of additional ``List`` nesting levels *above*
the coordinate leaf:

   +---------------------------+-----+-------+-------+--------------+
   | Nesting above coord leaf  |  0  |   1   |   2   |      3       |
   +---------------------------+-----+-------+-------+--------------+
   | Type                      | Pt  | Line/ | Poly/ | MultiPolygon |
   |                           |     | MPoint| MLine |              |
   +---------------------------+-----+-------+-------+--------------+

Additionally we support two **heuristic** cases where the data may not
follow the full GeoArrow spec but can still be reasonably interpreted:

- ``series(list(2 * float))`` — a series where each element is a variable-
  length list of 2-element fixed-size arrays.  This is the standard output
  of ``ak.Array([[[x0,y0],[x1,y1]], ...])`` and can be treated as a
  LineString in the GeoArrow interleaved FixedSizeList sense.

- ``series(2 * float)`` or ``series(N * float)`` — a series where each
  row *is* a coordinate pair (or more generally an N-component vector that
  could be a Point). This is ``RegularArray(size=2)`` at the outer level.

These match functions follow the ``match`` protocol of
``akimbo.apply_tree.dec``: they accept one or more ``ak.contents.Content``
layout nodes and return a bool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import awkward as ak


# ===========================================================================
# Coordinate leaf kind
# ===========================================================================

class CoordKind(Enum):
    """How coordinate values are stored at the innermost level."""
    INTERLEAVED_FLAT = auto()   # list<float> — spatialpandas, [x,y,x,y,...]
    INTERLEAVED_FSL  = auto()   # FixedSizeList<float>[n] — geoarrow native
    SEPARATED_STRUCT = auto()   # Struct<x:float, y:float, ...> — geoarrow recommended


@dataclass(frozen=True)
class GeoLayout:
    """Describes the geometry structure of a matched layout node.

    Attributes
    ----------
    coord_kind : CoordKind
        How individual coordinate values are stored.
    n_dims : int
        Number of spatial dimensions (2 for XY, 3 for XYZ, etc.).
    list_depth : int
        Number of List nesting levels above the coordinate leaf.
        0 = Point, 1 = LineString/MultiPoint, 2 = Polygon/MultiLine,
        3 = MultiPolygon.
    """
    coord_kind: CoordKind
    n_dims: int
    list_depth: int


# ===========================================================================
# Internal layout inspection helpers
# ===========================================================================

def _unwrap(layout):
    """Strip an UnmaskedArray wrapper if present.

    ``ak.from_arrow`` inserts an ``UnmaskedArray`` between each
    ``ListOffsetArray`` and its content. We strip it so that structure
    checks see the true node type.
    """
    if isinstance(layout, ak.contents.UnmaskedArray):
        return layout.content
    return layout


def _is_float_numpy(layout) -> bool:
    """True for a plain NumpyArray of float/int dtype."""
    layout = _unwrap(layout)
    return layout.is_leaf and getattr(layout, "dtype", None) is not None \
        and layout.dtype.kind in ("f", "i", "u")


def _coord_kind_of(layout) -> GeoLayout | None:
    """Return the GeoLayout for a coordinate-leaf node, or None if not a geom leaf.

    A "coordinate leaf" is the innermost non-List node that holds actual
    coordinate numeric data in one of the three supported representations.
    """
    layout = _unwrap(layout)

    # --- Separated struct: Struct<x:float, y:float[, ...]> ------------------
    if isinstance(layout, ak.contents.RecordArray):
        fields = layout.fields
        if (len(fields) >= 2
                and fields[0] in ("x", "X", "lon", "longitude")
                and fields[1] in ("y", "Y", "lat", "latitude")
                and all(_is_float_numpy(c) for c in layout.contents)):
            return GeoLayout(CoordKind.SEPARATED_STRUCT, len(fields), 0)
        return None

    # --- Interleaved FixedSizeList: FixedSizeList<float>[n_dim] --------------
    if isinstance(layout, ak.contents.RegularArray):
        n = layout.size
        if n in (2, 3, 4) and _is_float_numpy(layout.content):
            return GeoLayout(CoordKind.INTERLEAVED_FSL, n, 0)
        return None

    # --- Interleaved flat: plain NumpyArray (used inside list<float>) --------
    # We do NOT return a GeoLayout for a bare NumpyArray here;
    # _geo_layout_of() handles the flat-list case at the list level.
    return None


def _geo_layout_of(layout) -> GeoLayout | None:
    """Return the ``GeoLayout`` for a layout node, or ``None`` if not a geometry.

    Counts List nesting levels above the coordinate leaf.  Accepts:
    - ``list<float>`` (spatialpandas interleaved flat)
    - ``list<FixedSizeList<float>[n]>`` (geoarrow interleaved)
    - ``list<Struct<x,y,...>>`` (geoarrow separated)
    - ``FixedSizeList<float>[n]`` or ``Struct<x,y>`` at depth-0 (Point)
    - ``RegularArray(size=2, NumpyArray)`` — heuristic 2*float coord pair
    - Nested lists of the above (for Polygon, MultiPolygon, etc.)
    """
    layout = _unwrap(layout)

    # Depth-0: is this node itself a coordinate leaf?
    leaf = _coord_kind_of(layout)
    if leaf is not None:
        return leaf

    # Heuristic: RegularArray(size=2, float) at the outer level — a series
    # of 2D coordinate pairs, treat as Point.
    if isinstance(layout, ak.contents.RegularArray):
        if layout.size in (2, 3, 4) and _is_float_numpy(layout.content):
            return GeoLayout(CoordKind.INTERLEAVED_FSL, layout.size, 0)

    # List-type node: unwrap one List level and recurse.
    if not layout.is_list:
        return None

    # Exclude string / bytestring lists
    array_param = layout.parameter("__array__")
    if array_param in ("string", "bytestring", "char", "byte"):
        return None

    inner = _unwrap(layout.content)

    # Special case: list<float> — spatialpandas interleaved flat
    if _is_float_numpy(inner):
        return GeoLayout(CoordKind.INTERLEAVED_FLAT, 2, 1)

    # Recurse for deeper nesting
    inner_geo = _geo_layout_of(inner)
    if inner_geo is None:
        return None
    return GeoLayout(inner_geo.coord_kind, inner_geo.n_dims, inner_geo.list_depth + 1)


# ===========================================================================
# Public match functions — passed as ``match=`` to dec()
# ===========================================================================

def _match_at_depth(layout, depth: int) -> bool:
    """Return True if the layout is a geometry node with the given list_depth."""
    geo = _geo_layout_of(layout)
    return geo is not None and geo.list_depth == depth


def match_point(*layouts, **_) -> bool:
    """Match Point: coordinate leaf at depth 0.

    Accepts FixedSizeList<double>[2/3/4], Struct<x,y,...>,
    and heuristic 2*float RegularArray.
    """
    return _match_at_depth(layouts[0], 0)


def match_line(*layouts, **_) -> bool:
    """Match LineString / MultiPoint / Ring: list<Coordinate> (depth 1).

    Accepts all three coordinate representations, plus the spatialpandas
    ``list<float>`` interleaved flat form.
    """
    return _match_at_depth(layouts[0], 1)


def match_polygon(*layouts, **_) -> bool:
    """Match Polygon / MultiLineString: list<list<Coordinate>> (depth 2)."""
    return _match_at_depth(layouts[0], 2)


def match_multipolygon(*layouts, **_) -> bool:
    """Match MultiPolygon: list<list<list<Coordinate>>> (depth 3)."""
    return _match_at_depth(layouts[0], 3)


def match_any_geom(*layouts, **_) -> bool:
    """Match any geometry layout (depth 0–3, any coordinate representation)."""
    return _geo_layout_of(layouts[0]) is not None


# ===========================================================================
# Buffer extraction — normalise all representations to interleaved flat
# ===========================================================================

def extract_interleaved(layout) -> tuple:
    """Extract ``(values, offsets)`` from a geometry layout node, normalising
    all coordinate representations to **interleaved flat float64**.

    For separated struct coordinates, the x and y (and optionally z/m) arrays
    are interleaved in-memory to produce a flat ``[x0,y0,x1,y1,...]`` buffer.
    For FixedSizeList coordinates, the existing flat buffer is reused directly.
    For spatialpandas-style flat lists, the buffer is used as-is.

    Returns
    -------
    values : numpy.ndarray  shape (N_coords * n_dims,)
        Flat interleaved coordinate buffer.
    offsets : tuple[numpy.ndarray, ...]
        One int32 offset array per List nesting level, outermost first.
        Offsets index into the *coordinate* (not raw float) dimension,
        i.e. ``values[offsets[i]*n_dims : offsets[i+1]*n_dims]`` is
        geometry ``i`` when ``list_depth == 1``.

        For INTERLEAVED_FLAT the offsets index directly into the float
        buffer as before (each stride covers one float, not one point),
        so existing kernels remain valid unchanged.
    geo : GeoLayout
        The detected geometry layout descriptor.
    """
    from akimbo_geo._compat import array_module

    geo = _geo_layout_of(layout)
    if geo is None:
        raise ValueError(f"Layout {type(layout).__name__} is not a recognised geometry")

    offsets = []
    current = _unwrap(layout)

    # Peel off List nesting levels, collecting offset arrays.
    # We use the raw .data attribute of the awkward Index object and cast to
    # int32 *on the same device* — array_module detects numpy vs cupy.
    for _ in range(geo.list_depth):
        raw = current.offsets.data          # numpy.ndarray or cupy.ndarray
        xp = array_module(raw)
        offsets.append(xp.asarray(raw, dtype=xp.int32))
        current = _unwrap(current.content)

    # current is now the coordinate leaf node
    values = _extract_coord_values(current, geo.coord_kind, geo.n_dims)
    return values, tuple(offsets), geo


def _extract_coord_values(layout, coord_kind: CoordKind, n_dims: int):
    """Extract a flat interleaved float64 array from a coordinate leaf.

    The returned array lives on the *same device* as the input data — it is
    a numpy array for CPU layouts and a cupy array for GPU layouts.  The
    calling code must not assume numpy; use ``_compat.array_module(values)``
    to branch on device type.
    """
    from akimbo_geo._compat import array_module

    layout = _unwrap(layout)

    if coord_kind == CoordKind.INTERLEAVED_FLAT:
        # NumpyArray (or cupy equivalent) — flat buffer, already interleaved
        raw = layout.data
        xp = array_module(raw)
        return xp.asarray(raw, dtype=xp.float64)

    if coord_kind == CoordKind.INTERLEAVED_FSL:
        # RegularArray(size=n_dims, NumpyArray) — flat buffer, already interleaved
        leaf = _unwrap(layout.content)
        raw  = leaf.data
        xp   = array_module(raw)
        return xp.asarray(raw, dtype=xp.float64)

    if coord_kind == CoordKind.SEPARATED_STRUCT:
        # RecordArray with separate per-dimension arrays.
        # Interleave: [x0,y0,x1,y1,...] or [x0,y0,z0,x1,y1,z1,...]
        # All component arrays must live on the same device.
        raw_arrays = [_unwrap(c).data for c in layout.contents[:n_dims]]
        xp = array_module(raw_arrays[0])
        arrays = [xp.asarray(a, dtype=xp.float64) for a in raw_arrays]
        n_pts = len(arrays[0])
        out = xp.empty(n_pts * n_dims, dtype=xp.float64)
        for i, arr in enumerate(arrays):
            out[i::n_dims] = arr
        return out

    raise ValueError(f"Unknown CoordKind: {coord_kind}")


# ===========================================================================
# Legacy alias for backward compatibility with existing op functions
# ===========================================================================

def extract_offsets_and_values(layout):
    """Return ``(buffer_values, buffer_offsets)`` from a geometry layout node.

    This is the primary entry point used by the op functions in
    ``accessor.py``.  All coordinate representations are normalised to a
    flat interleaved float64 buffer so the numba kernels can operate without
    modification.

    For INTERLEAVED_FLAT (spatialpandas), the offset values are direct float-
    buffer indices (stride 1 per float, i.e. 2 floats = 1 coord).

    For INTERLEAVED_FSL and SEPARATED_STRUCT, the offset values are *point*
    indices (stride 1 per coordinate point).  The returned values buffer is
    interleaved, so ``values[off*n_dims : (off+1)*n_dims]`` is one point.
    The op functions use ``n_dims`` from the GeoLayout to scale accordingly.

    Returns
    -------
    values : numpy.ndarray
    offsets : tuple[numpy.ndarray, ...]
    geo : GeoLayout
    """
    return extract_interleaved(layout)


def flat_offsets_into_values(offsets_tuple):
    """Flatten a tuple of offset arrays into direct offsets into buffer_values."""
    flat = offsets_tuple[0]
    for inner in offsets_tuple[1:]:
        flat = inner[flat]
    return flat
