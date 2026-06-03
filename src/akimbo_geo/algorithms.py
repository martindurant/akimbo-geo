"""algorithms.py — numba-jitted geometry kernels.

All kernels operate on the raw flat coordinate buffer and offset arrays
extracted from the Arrow list structure, following the same conventions as
spatialpandas:

- Coordinates are **interleaved**: ``values[2k] = x``, ``values[2k+1] = y``.
- ``value_offsets`` is a 1-D array of start/stop indices; geometry ``i``
  occupies ``values[value_offsets[i] : value_offsets[i+1]]``.
- For multi-level types a tuple of offset arrays is passed; each level
  telescopes into the next.

Scalar kernels (decorated with ``@ngjit``) operate on a single geometry.
Vectorised map kernels (decorated with ``@ngpjit``) iterate in parallel over
an entire array, filling a pre-allocated result buffer.

These kernels are vendored/adapted from spatialpandas
(https://github.com/holoviz/spatialpandas) with the long-term intention of
replacing that dependency.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from numba import jit, prange

# ---------------------------------------------------------------------------
# JIT decorators — mirrors spatialpandas utils.ngjit / ngpjit
# ---------------------------------------------------------------------------

ngjit  = jit(nopython=True, nogil=True)
ngpjit = jit(nopython=True, nogil=True, parallel=True)


# ===========================================================================
# Scalar kernels — operate on a single geometry's slice of the flat buffer
# ===========================================================================

@ngjit  # pragma: no cover
def compute_line_length(values, value_offsets):
    """Total Euclidean length of a single line / ring / multiline.

    Parameters
    ----------
    values:
        Flat interleaved coordinate array.
    value_offsets:
        1-D array of start/stop indices; each consecutive pair
        ``[value_offsets[i], value_offsets[i+1])`` is one sub-line segment.
        For a plain Line this has length 2; for a MultiLine it has one entry
        per component line.

    Returns
    -------
    float64 total length.
    """
    total = 0.0
    for seg in range(len(value_offsets) - 1):
        start = value_offsets[seg]
        stop  = value_offsets[seg + 1]
        x0 = values[start]
        y0 = values[start + 1]
        for i in range(start + 2, stop, 2):
            x1 = values[i]
            y1 = values[i + 1]
            if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                total += sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            x0 = x1
            y0 = y1
    return total


@ngjit  # pragma: no cover
def compute_area(values, value_offsets):
    """Signed area of a single polygon (exterior + holes) via the shoelace formula.

    Parameters
    ----------
    values:
        Flat interleaved coordinate array.
    value_offsets:
        1-D array of ring start/stop indices.  First ring = exterior shell
        (CCW, positive area); subsequent rings = holes (CW, negative area).

    Returns
    -------
    float64 area (positive for CCW orientation).
    """
    area = 0.0
    for seg in range(len(value_offsets) - 1):
        start = value_offsets[seg]
        stop  = value_offsets[seg + 1]
        poly_length = stop - start
        if poly_length < 6:
            # Degenerate ring — fewer than 3 coordinate pairs
            continue
        for k in range(start, stop - 4, 2):
            i = k + 2
            j = k + 4
            area += values[i] * (values[j + 1] - values[k + 1])
        # Wrap-around term
        area += values[start] * (values[start + 3] - values[stop - 3])
    return area / 2.0


@ngjit  # pragma: no cover
def compute_bounds(values, start, stop):
    """Bounding box of a single geometry's flat coordinate slice.

    Returns
    -------
    (xmin, ymin, xmax, ymax) as float64 scalars.
    """
    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf
    i = start
    while i < stop:
        x = values[i]
        y = values[i + 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
        i += 2
    return xmin, ymin, xmax, ymax


@ngjit  # pragma: no cover
def compute_centroid_line(values, start, stop):
    """Centroid of a point set / line as the simple mean of coordinates.

    Returns
    -------
    (cx, cy) float64 pair.
    """
    n = (stop - start) // 2
    if n == 0:
        return np.nan, np.nan
    cx = 0.0
    cy = 0.0
    i = start
    while i < stop:
        cx += values[i]
        cy += values[i + 1]
        i += 2
    return cx / n, cy / n


@ngjit  # pragma: no cover
def compute_centroid_polygon(values, value_offsets):
    """Area-weighted centroid of a polygon (exterior ring only).

    Uses the standard shoelace-based centroid formula.

    Returns
    -------
    (cx, cy) float64 pair.
    """
    # Use only the exterior ring (first ring, value_offsets[0:2])
    start = value_offsets[0]
    stop  = value_offsets[1]
    area  = 0.0
    cx    = 0.0
    cy    = 0.0
    n = stop - start
    if n < 6:
        return np.nan, np.nan
    for k in range(start, stop - 2, 2):
        x0 = values[k]
        y0 = values[k + 1]
        x1 = values[k + 2]
        y1 = values[k + 3]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx   += (x0 + x1) * cross
        cy   += (y0 + y1) * cross
    # Close the ring
    x0 = values[stop - 2]
    y0 = values[stop - 1]
    x1 = values[start]
    y1 = values[start + 1]
    cross = x0 * y1 - x1 * y0
    area += cross
    cx   += (x0 + x1) * cross
    cy   += (y0 + y1) * cross
    if area == 0.0:
        return np.nan, np.nan
    area *= 3.0
    return cx / area, cy / area


# ===========================================================================
# Vectorised map kernels — one entry per geometry, parallel across rows
# ===========================================================================

# --- 1-level nesting: list<float> (Line / Ring / MultiPoint) ---------------

@ngpjit  # pragma: no cover
def length_map1(values, offsets0, result, missing):
    """Compute length for each list<float> geometry in parallel.

    Parameters
    ----------
    values:   flat float64 coordinate buffer.
    offsets0: (N+1,) int32 offsets into values.
    result:   (N,) float64 output array (pre-allocated, NaN-filled).
    missing:  (N,) bool array; True → skip (leave NaN).
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            seg_offsets = offsets0[i:i + 2]
            result[i] = compute_line_length(values, seg_offsets)


@ngpjit  # pragma: no cover
def bounds_map1(values, offsets0, result, missing):
    """Bounding box for each list<float> geometry.

    result shape: (N, 4) — columns are xmin, ymin, xmax, ymax.
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            xmin, ymin, xmax, ymax = compute_bounds(values, offsets0[i], offsets0[i + 1])
            result[i, 0] = xmin
            result[i, 1] = ymin
            result[i, 2] = xmax
            result[i, 3] = ymax


@ngpjit  # pragma: no cover
def centroid_map1(values, offsets0, result_x, result_y, missing):
    """Centroid (mean of coords) for each list<float> geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            cx, cy = compute_centroid_line(values, offsets0[i], offsets0[i + 1])
            result_x[i] = cx
            result_y[i] = cy


# --- 2-level nesting: list<list<float>> (Polygon / MultiLine) ---------------

@ngpjit  # pragma: no cover
def length_map2(values, offsets0, offsets1, result, missing):
    """Compute length for each list<list<float>> geometry in parallel.

    offsets0 : (N+1,) — outer list (polygons or multilines)
    offsets1 : variable length — inner list (rings or lines)
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            result[i] = compute_line_length(values, inner)


@ngpjit  # pragma: no cover
def area_map2(values, offsets0, offsets1, result, missing):
    """Compute area for each list<list<float>> geometry in parallel.

    For Polygon: offsets0 separates polygons; offsets1 separates rings.
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            result[i] = compute_area(values, inner)


@ngpjit  # pragma: no cover
def bounds_map2(values, offsets0, offsets1, result, missing):
    """Bounding box for each list<list<float>> geometry.

    result shape: (N, 4).
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = offsets1[offsets0[i]]
            stop  = offsets1[offsets0[i + 1]]
            xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
            result[i, 0] = xmin
            result[i, 1] = ymin
            result[i, 2] = xmax
            result[i, 3] = ymax


@ngpjit  # pragma: no cover
def centroid_map2(values, offsets0, offsets1, result_x, result_y, missing):
    """Area-weighted centroid for each list<list<float>> polygon geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            cx, cy = compute_centroid_polygon(values, inner)
            result_x[i] = cx
            result_y[i] = cy


# --- 3-level nesting: list<list<list<float>>> (MultiPolygon) ----------------

@ngpjit  # pragma: no cover
def length_map3(values, offsets0, offsets1, offsets2, result, missing):
    """Compute length for each list<list<list<float>>> geometry in parallel."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner1 = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            start2 = inner1[0]
            stop2  = inner1[-1]
            inner2 = offsets2[start2:stop2 + 1]
            result[i] = compute_line_length(values, inner2)


@ngpjit  # pragma: no cover
def area_map3(values, offsets0, offsets1, offsets2, result, missing):
    """Compute area for each list<list<list<float>>> geometry in parallel."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner1 = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            start2 = inner1[0]
            stop2  = inner1[-1]
            inner2 = offsets2[start2:stop2 + 1]
            result[i] = compute_area(values, inner2)


@ngpjit  # pragma: no cover
def bounds_map3(values, offsets0, offsets1, offsets2, result, missing):
    """Bounding box for each list<list<list<float>>> geometry.

    result shape: (N, 4).
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner1 = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            start  = offsets2[inner1[0]]
            stop   = offsets2[inner1[-1]]
            xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
            result[i, 0] = xmin
            result[i, 1] = ymin
            result[i, 2] = xmax
            result[i, 3] = ymax


# ===========================================================================
# Aggregate (whole-column) helpers
# ===========================================================================

@ngjit  # pragma: no cover
def total_bounds(values):
    """Return the aggregate bounding box of all coordinates in values.

    Returns
    -------
    (xmin, ymin, xmax, ymax) float64 tuple.
    """
    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf
    i = 0
    while i < len(values):
        x = values[i]
        y = values[i + 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
        i += 2
    return xmin, ymin, xmax, ymax


# ===========================================================================
# Intersection / predicate kernels (ported from spatialpandas)
# ===========================================================================

@ngjit  # pragma: no cover
def _segments_intersect(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    """Return True if line segment A intersects line segment B."""
    def _orient(px, py, qx, qy, rx, ry):
        val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        if val > 0.0:
            return 1
        elif val < 0.0:
            return -1
        return 0

    def _on_segment(px, py, qx, qy, rx, ry):
        return (min(px, rx) <= qx <= max(px, rx) and
                min(py, ry) <= qy <= max(py, ry))

    o1 = _orient(ax0, ay0, ax1, ay1, bx0, by0)
    o2 = _orient(ax0, ay0, ax1, ay1, bx1, by1)
    o3 = _orient(bx0, by0, bx1, by1, ax0, ay0)
    o4 = _orient(bx0, by0, bx1, by1, ax1, ay1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(ax0, ay0, bx0, by0, ax1, ay1):
        return True
    if o2 == 0 and _on_segment(ax0, ay0, bx1, by1, ax1, ay1):
        return True
    if o3 == 0 and _on_segment(bx0, by0, ax0, ay0, bx1, by1):
        return True
    if o4 == 0 and _on_segment(bx0, by0, ax1, ay1, bx1, by1):
        return True
    return False


@ngjit  # pragma: no cover
def _point_in_ring(px, py, values, ring_start, ring_stop):
    """Winding-number point-in-polygon test for a single ring."""
    winding = 0
    n = (ring_stop - ring_start) // 2
    for k in range(n - 1):
        idx = ring_start + k * 2
        x0 = values[idx]
        y0 = values[idx + 1]
        x1 = values[idx + 2]
        y1 = values[idx + 3]
        if y0 <= py:
            if y1 > py:
                if (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0) > 0:
                    winding += 1
        else:
            if y1 <= py:
                if (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0) < 0:
                    winding -= 1
    return winding != 0


@ngpjit  # pragma: no cover
def intersects_bounds_map1(x0, y0, x1, y1, values, offsets0, result, missing):
    """Test whether each list<float> geometry intersects the given bounding box."""
    n = len(offsets0) - 1
    for i in prange(n):
        if missing[i]:
            continue
        gx_min, gy_min, gx_max, gy_max = compute_bounds(
            values, offsets0[i], offsets0[i + 1]
        )
        # Quick bbox reject
        if gx_max < x0 or gx_min > x1 or gy_max < y0 or gy_min > y1:
            result[i] = False
            continue
        # At least one vertex inside the query box
        found = False
        j = offsets0[i]
        while j < offsets0[i + 1]:
            vx = values[j]
            vy = values[j + 1]
            if x0 <= vx <= x1 and y0 <= vy <= y1:
                found = True
                break
            j += 2
        if found:
            result[i] = True
            continue
        # Check whether any edge crosses the query bbox boundary
        qx0, qy0, qx1, qy1 = x0, y0, x1, y1
        j = offsets0[i]
        while j < offsets0[i + 1] - 2:
            ax = values[j];   ay = values[j + 1]
            bx = values[j + 2]; by = values[j + 3]
            # Check against each of the 4 bbox edges
            if (_segments_intersect(ax, ay, bx, by, qx0, qy0, qx1, qy0) or
                    _segments_intersect(ax, ay, bx, by, qx1, qy0, qx1, qy1) or
                    _segments_intersect(ax, ay, bx, by, qx1, qy1, qx0, qy1) or
                    _segments_intersect(ax, ay, bx, by, qx0, qy1, qx0, qy0)):
                found = True
                break
            j += 2
        result[i] = found
