"""match.py — layout-matching predicates for geometry node identification.

Geometry types are stored as nested Arrow list arrays of interleaved float
coordinates, following the spatialpandas convention:

    list<float>              — Line, Ring, MultiPoint  (1 nesting level)
    list<list<float>>        — Polygon, MultiLine       (2 nesting levels)
    list<list<list<float>>>  — MultiPolygon             (3 nesting levels)

All leaf values are interleaved (x0, y0, x1, y1, ...).

These functions follow the ``match`` protocol expected by
``akimbo.apply_tree.dec``: they receive one or more
``ak.contents.Content`` layout nodes and return a bool.
"""

from __future__ import annotations

import awkward as ak


def _unwrap(layout):
    """Strip a single UnmaskedArray wrapper if present.

    When Arrow arrays are converted to awkward via ``ak.from_arrow``, each
    list nesting level gets an ``UnmaskedArray`` between the ``ListOffsetArray``
    and its content.  The match predicates need to see through this wrapper
    to check the true content type.
    """
    if isinstance(layout, ak.contents.UnmaskedArray):
        return layout.content
    return layout


def _is_float_leaf(layout) -> bool:
    """True for a numeric (float/int) leaf array, stripping UnmaskedArray."""
    layout = _unwrap(layout)
    return layout.is_leaf and layout.dtype.kind in ("f", "i", "u")


def _is_float_list(layout) -> bool:
    """True for list<numeric> — the innermost coordinate list.

    Explicitly excludes string/bytestring lists, which also appear as
    ``list<uint8>`` in the awkward layout tree.
    """
    layout = _unwrap(layout)
    if not layout.is_list:
        return False
    # Exclude string / char / bytestring arrays — they share list<uint8> structure
    array_param = layout.parameter("__array__")
    if array_param in ("string", "bytestring"):
        return False
    return _is_float_leaf(layout.content)


def _is_float_list2(layout) -> bool:
    """True for list<list<numeric>>."""
    layout = _unwrap(layout)
    return layout.is_list and _is_float_list(layout.content)


def _is_float_list3(layout) -> bool:
    """True for list<list<list<numeric>>>."""
    layout = _unwrap(layout)
    return layout.is_list and _is_float_list2(layout.content)


# ---------------------------------------------------------------------------
# Public match functions — passed as ``match=`` to dec()
# ---------------------------------------------------------------------------

def match_line(*layouts, **_) -> bool:
    """Match Line / Ring / MultiPoint: list<float> (1 nesting level).

    Corresponds to spatialpandas ``_nesting_levels = 1`` arrays:
    ``LineArray``, ``RingArray``, ``MultiPointArray``.
    """
    return _is_float_list(layouts[0])


def match_polygon(*layouts, **_) -> bool:
    """Match Polygon / MultiLine: list<list<float>> (2 nesting levels).

    Corresponds to spatialpandas ``_nesting_levels = 2`` arrays:
    ``PolygonArray``, ``MultiLineArray``.
    """
    return _is_float_list2(layouts[0])


def match_multipolygon(*layouts, **_) -> bool:
    """Match MultiPolygon: list<list<list<float>>> (3 nesting levels).

    Corresponds to spatialpandas ``_nesting_levels = 3`` arrays:
    ``MultiPolygonArray``.
    """
    return _is_float_list3(layouts[0])


def match_any_geom(*layouts, **_) -> bool:
    """Match any geometry layout (1, 2, or 3 nesting levels of float lists)."""
    layout = layouts[0]
    return _is_float_list(layout) or _is_float_list2(layout) or _is_float_list3(layout)


# ---------------------------------------------------------------------------
# Helpers used by ops to extract raw buffers from a matched layout node
# ---------------------------------------------------------------------------

def extract_offsets_and_values(layout):
    """Return (buffer_values, buffer_offsets) from an ak layout node.

    Traverses the awkward layout tree directly (no Arrow round-trip),
    stripping ``UnmaskedArray`` wrappers that appear when data comes from
    ``ak.from_arrow``.

    Returns
    -------
    buffer_values : numpy.ndarray
        Flat 1-D float64 array of interleaved coordinates.
    buffer_offsets : tuple[numpy.ndarray, ...]
        One int32 offset array per nesting level; outermost first.
    """
    import numpy as np

    offsets = []
    current = _unwrap(layout)

    # Walk list nesting levels
    while current.is_list:
        offsets.append(np.asarray(current.offsets.data).astype(np.int32))
        current = _unwrap(current.content)

    # current is now a NumpyArray leaf
    values = np.asarray(current.data).astype(np.float64)
    return values, tuple(offsets)


def flat_offsets_into_values(offsets_tuple):
    """Flatten a tuple of offset arrays into direct offsets into ``buffer_values``.

    For 1-level: returns offsets_tuple[0].
    For 2-level: returns offsets1[offsets0].
    For 3-level: returns offsets2[offsets1[offsets0]].

    This gives ``buffer_outer_offsets`` in spatialpandas terminology —
    direct start/stop indices into the flat coordinate buffer per top-level
    geometry.
    """
    flat = offsets_tuple[0]
    for inner in offsets_tuple[1:]:
        flat = inner[flat]
    return flat
