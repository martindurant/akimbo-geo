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

Design: shared scalar implementations
--------------------------------------
Each scalar helper is first written as a plain Python function (``_*_impl``)
using only ``math`` module constants (``math.inf``, ``math.nan``,
``math.isfinite``) so it contains no ``numpy``-specific calls.  The CPU
version is then created with ``ngjit(_*_impl)`` and the GPU device version
(in ``algorithms_gpu.py``) with ``cuda.jit(device=True)(_*_impl)``.  This
guarantees the two flavours are compiled from the **same source** with no
duplication.

Exceptions: ``compute_line_length`` cannot be shared because the CPU map
kernels pass a slice of the offsets array (``offsets0[i:i+2]``) while CUDA
kernels must use explicit integer indices — the two have different call
signatures.  ``segmentize_*``, ``orient_*``, and ``reverse_*`` are
sequential kernels not ported to GPU.

These kernels are vendored/adapted from spatialpandas
(https://github.com/holoviz/spatialpandas) with the long-term intention of
replacing that dependency.
"""

from __future__ import annotations

from math import inf, isfinite, nan, sqrt

import numpy as np
from numba import jit, prange

# ---------------------------------------------------------------------------
# JIT decorators — mirrors spatialpandas utils.ngjit / ngpjit
# ---------------------------------------------------------------------------

ngjit  = jit(nopython=True, nogil=True)
ngpjit = jit(nopython=True, nogil=True, parallel=True)


# ===========================================================================
# Plain Python _impl functions — compiled to both CPU and GPU flavours
# ===========================================================================

def _compute_area_impl(values, value_offsets):
    """Signed area of a single polygon via shoelace.  CPU+GPU shared."""
    area = 0.0
    for seg in range(len(value_offsets) - 1):
        start = value_offsets[seg]
        stop  = value_offsets[seg + 1]
        poly_length = stop - start
        if poly_length < 6:
            continue
        for k in range(start, stop - 4, 2):
            i = k + 2
            j = k + 4
            area += values[i] * (values[j + 1] - values[k + 1])
        area += values[start] * (values[start + 3] - values[stop - 3])
    return area / 2.0


def _compute_bounds_impl(values, start, stop):
    """Bounding box of a coordinate slice.  CPU+GPU shared."""
    xmin =  inf;  ymin =  inf
    xmax = -inf;  ymax = -inf
    i = start
    while i < stop:
        x = values[i];  y = values[i + 1]
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
        i += 2
    return xmin, ymin, xmax, ymax


def _compute_centroid_line_impl(values, start, stop):
    """Mean-of-coordinates centroid.  CPU+GPU shared."""
    n = (stop - start) // 2
    if n == 0:
        return nan, nan
    cx = 0.0;  cy = 0.0
    i = start
    while i < stop:
        cx += values[i];  cy += values[i + 1]
        i += 2
    return cx / n, cy / n


def _compute_centroid_polygon_impl(values, value_offsets):
    """Shoelace centroid (exterior ring only).  CPU+GPU shared."""
    start = value_offsets[0]
    stop  = value_offsets[1]
    n = stop - start
    if n < 6:
        return nan, nan
    area = 0.0;  cx = 0.0;  cy = 0.0
    for k in range(start, stop - 2, 2):
        x0 = values[k];     y0 = values[k + 1]
        x1 = values[k + 2]; y1 = values[k + 3]
        cross = x0 * y1 - x1 * y0
        area += cross;  cx += (x0 + x1) * cross;  cy += (y0 + y1) * cross
    x0 = values[stop - 2];  y0 = values[stop - 1]
    x1 = values[start];     y1 = values[start + 1]
    cross = x0 * y1 - x1 * y0
    area += cross;  cx += (x0 + x1) * cross;  cy += (y0 + y1) * cross
    if area == 0.0:
        return nan, nan
    area *= 3.0
    return cx / area, cy / area


def _total_bounds_impl(values):
    """Aggregate bounding box of all coordinates.  CPU+GPU shared."""
    xmin =  inf;  ymin =  inf
    xmax = -inf;  ymax = -inf
    i = 0
    while i < len(values):
        x = values[i];  y = values[i + 1]
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
        i += 2
    return xmin, ymin, xmax, ymax


def _orient_ring_impl(values, start, stop):
    """Signed area of a single ring.  CPU+GPU shared."""
    n_pts = (stop - start) // 2
    if n_pts < 3:
        return 0.0
    area = 0.0
    for k in range(n_pts - 1):
        i = start + k * 2
        x0 = values[i];     y0 = values[i + 1]
        x1 = values[i + 2]; y1 = values[i + 3]
        area += x0 * y1 - x1 * y0
    x0 = values[stop - 2];  y0 = values[stop - 1]
    x1 = values[start];     y1 = values[start + 1]
    area += x0 * y1 - x1 * y0
    return area / 2.0


# ===========================================================================
# CPU scalar kernels — plain _impl wrapped with ngjit
# ===========================================================================

compute_area             = ngjit(_compute_area_impl)             # pragma: no cover
compute_bounds           = ngjit(_compute_bounds_impl)           # pragma: no cover
compute_centroid_line    = ngjit(_compute_centroid_line_impl)    # pragma: no cover
compute_centroid_polygon = ngjit(_compute_centroid_polygon_impl) # pragma: no cover
total_bounds             = ngjit(_total_bounds_impl)             # pragma: no cover
orient_ring              = ngjit(_orient_ring_impl)              # pragma: no cover


@ngjit  # pragma: no cover
def compute_line_length(values, value_offsets):
    """Total Euclidean length of a single line / ring / multiline.

    Takes a slice of the offsets array (CPU map-kernel convention).
    GPU kernels use a different call signature (explicit start/stop).
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
            if isfinite(x0) and isfinite(y0) and isfinite(x1) and isfinite(y1):
                total += sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            x0 = x1
            y0 = y1
    return total


# Legacy alias kept for any code that imports compute_centroid_line directly
compute_centroid_line_scalar = compute_centroid_line  # same object


# ===========================================================================
# Vectorised map kernels — one entry per geometry, parallel across rows
# ===========================================================================

# --- 1-level nesting: list<float> (Line / Ring / MultiPoint) ---------------

@ngpjit  # pragma: no cover
def length_map1(values, offsets0, result, missing):
    """Compute length for each list<float> geometry in parallel."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            seg_offsets = offsets0[i:i + 2]
            result[i] = compute_line_length(values, seg_offsets)


@ngpjit  # pragma: no cover
def bounds_map1(values, offsets0, result, missing):
    """Bounding box for each list<float> geometry. result shape: (N, 4)."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            xmin, ymin, xmax, ymax = compute_bounds(values, offsets0[i], offsets0[i + 1])
            result[i, 0] = xmin; result[i, 1] = ymin
            result[i, 2] = xmax; result[i, 3] = ymax


@ngpjit  # pragma: no cover
def centroid_map1(values, offsets0, result_x, result_y, missing):
    """Centroid (mean of coords) for each list<float> geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            cx, cy = compute_centroid_line(values, offsets0[i], offsets0[i + 1])
            result_x[i] = cx; result_y[i] = cy


# --- 2-level nesting: list<list<float>> (Polygon / MultiLine) ---------------

@ngpjit  # pragma: no cover
def length_map2(values, offsets0, offsets1, result, missing):
    """Compute length for each list<list<float>> geometry in parallel."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            result[i] = compute_line_length(values, inner)


@ngpjit  # pragma: no cover
def area_map2(values, offsets0, offsets1, result, missing):
    """Compute area for each list<list<float>> geometry in parallel."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            result[i] = compute_area(values, inner)


@ngpjit  # pragma: no cover
def bounds_map2(values, offsets0, offsets1, result, missing):
    """Bounding box for each list<list<float>> geometry. result shape: (N, 4)."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = offsets1[offsets0[i]]
            stop  = offsets1[offsets0[i + 1]]
            xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
            result[i, 0] = xmin; result[i, 1] = ymin
            result[i, 2] = xmax; result[i, 3] = ymax


@ngpjit  # pragma: no cover
def centroid_map2(values, offsets0, offsets1, result_x, result_y, missing):
    """Area-weighted centroid for each list<list<float>> polygon geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            cx, cy = compute_centroid_polygon(values, inner)
            result_x[i] = cx; result_y[i] = cy


# --- 3-level nesting: list<list<list<float>>> (MultiPolygon) ----------------

@ngpjit  # pragma: no cover
def length_map3(values, offsets0, offsets1, offsets2, result, missing):
    """Compute length for each list<list<list<float>>> geometry in parallel."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner1 = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            inner2 = offsets2[inner1[0]:inner1[-1] + 1]
            result[i] = compute_line_length(values, inner2)


@ngpjit  # pragma: no cover
def area_map3(values, offsets0, offsets1, offsets2, result, missing):
    """Compute area for each list<list<list<float>>> geometry in parallel."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner1 = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            inner2 = offsets2[inner1[0]:inner1[-1] + 1]
            result[i] = compute_area(values, inner2)


@ngpjit  # pragma: no cover
def bounds_map3(values, offsets0, offsets1, offsets2, result, missing):
    """Bounding box for each list<list<list<float>>> geometry. result shape: (N, 4)."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            inner1 = offsets1[offsets0[i]:offsets0[i + 1] + 1]
            start  = offsets2[inner1[0]]
            stop   = offsets2[inner1[-1]]
            xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
            result[i, 0] = xmin; result[i, 1] = ymin
            result[i, 2] = xmax; result[i, 3] = ymax


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
        x0 = values[idx];     y0 = values[idx + 1]
        x1 = values[idx + 2]; y1 = values[idx + 3]
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
        if gx_max < x0 or gx_min > x1 or gy_max < y0 or gy_min > y1:
            result[i] = False
            continue
        found = False
        j = offsets0[i]
        while j < offsets0[i + 1]:
            vx = values[j];  vy = values[j + 1]
            if x0 <= vx <= x1 and y0 <= vy <= y1:
                found = True
                break
            j += 2
        if found:
            result[i] = True
            continue
        qx0, qy0, qx1, qy1 = x0, y0, x1, y1
        j = offsets0[i]
        while j < offsets0[i + 1] - 2:
            ax = values[j];   ay = values[j + 1]
            bx = values[j + 2]; by = values[j + 3]
            if (_segments_intersect(ax, ay, bx, by, qx0, qy0, qx1, qy0) or
                    _segments_intersect(ax, ay, bx, by, qx1, qy0, qx1, qy1) or
                    _segments_intersect(ax, ay, bx, by, qx1, qy1, qx0, qy1) or
                    _segments_intersect(ax, ay, bx, by, qx0, qy1, qx0, qy0)):
                found = True
                break
            j += 2
        result[i] = found


# ===========================================================================
# Counting kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def count_coords_map1(offsets0, n_dims, result, missing):
    """Number of coordinate points per depth-1 geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            result[i] = (offsets0[i + 1] - offsets0[i]) // n_dims


@ngpjit  # pragma: no cover
def count_geoms_map2(offsets0, result, missing):
    """Number of sub-geometries per depth-2 geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            result[i] = offsets0[i + 1] - offsets0[i]


@ngpjit  # pragma: no cover
def count_interior_rings_map2(offsets0, result, missing):
    """Number of interior rings (holes) per polygon."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            n_rings = offsets0[i + 1] - offsets0[i]
            result[i] = n_rings - 1 if n_rings > 0 else 0


# ===========================================================================
# Predicate kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def is_closed_map1(values, offsets0, result, missing):
    """True if first coordinate == last coordinate for each depth-1 geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = offsets0[i];  stop = offsets0[i + 1]
            if stop - start < 4:
                result[i] = False
            else:
                result[i] = (values[start]     == values[stop - 2] and
                             values[start + 1] == values[stop - 1])


@ngpjit  # pragma: no cover
def is_ccw_map2(values, offsets0, offsets1, result, missing):
    """True if exterior ring of each depth-2 polygon is counter-clockwise."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            ring_start_idx = offsets0[i]
            inner = offsets1[ring_start_idx:ring_start_idx + 2]
            area = compute_area(values, inner)
            result[i] = area > 0.0


# ===========================================================================
# Coordinate extraction kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def get_x_map0(values, n_dims, result):
    """Extract x coordinate for each depth-0 Point geometry."""
    n = len(values) // n_dims
    for i in prange(n):
        result[i] = values[i * n_dims]


@ngpjit  # pragma: no cover
def get_y_map0(values, n_dims, result):
    """Extract y coordinate for each depth-0 Point geometry."""
    n = len(values) // n_dims
    for i in prange(n):
        result[i] = values[i * n_dims + 1]


@ngpjit  # pragma: no cover
def get_z_map0(values, n_dims, result):
    """Extract z coordinate for each depth-0 Point geometry (NaN if 2D)."""
    n = len(values) // n_dims
    for i in prange(n):
        if n_dims >= 3:
            result[i] = values[i * n_dims + 2]
        else:
            result[i] = nan


# ===========================================================================
# Affine transformation kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def translate_map(values, offsets0, xoff, yoff, result):
    """Add (xoff, yoff) to every coordinate in each depth-1 geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i];  stop = offsets0[i + 1]
        j = start
        while j < stop:
            result[j]     = values[j]     + xoff
            result[j + 1] = values[j + 1] + yoff
            j += 2


@ngpjit  # pragma: no cover
def scale_map(values, offsets0, xfact, yfact, ox, oy, result):
    """Scale each coordinate around origin (ox, oy)."""
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i];  stop = offsets0[i + 1]
        j = start
        while j < stop:
            result[j]     = (values[j]     - ox) * xfact + ox
            result[j + 1] = (values[j + 1] - oy) * yfact + oy
            j += 2


@ngpjit  # pragma: no cover
def affine_transform_map(values, offsets0, a, b, d, e, xoff, yoff, result):
    """Apply a 2D affine transform to every coordinate."""
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i];  stop = offsets0[i + 1]
        j = start
        while j < stop:
            x = values[j];  y = values[j + 1]
            result[j]     = a * x + b * y + xoff
            result[j + 1] = d * x + e * y + yoff
            j += 2


# ===========================================================================
# Coordinate manipulation kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def reverse_map1(values, offsets0, result):
    """Reverse the vertex order of each depth-1 geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i];  stop = offsets0[i + 1]
        n_pts = (stop - start) // 2
        for k in range(n_pts):
            src = start + k * 2
            dst = stop  - (k + 1) * 2
            result[src]     = values[dst]
            result[src + 1] = values[dst + 1]


@ngpjit  # pragma: no cover
def force_2d_map(values_nd, n_dims, result_2d):
    """Copy only (x, y) from an n-dimensional flat buffer."""
    n_pts = len(values_nd) // n_dims
    for i in prange(n_pts):
        result_2d[i * 2]     = values_nd[i * n_dims]
        result_2d[i * 2 + 1] = values_nd[i * n_dims + 1]


@ngpjit  # pragma: no cover
def force_3d_map(values_2d, z_val, result_3d):
    """Promote a 2D flat buffer to 3D by inserting a constant z value."""
    n_pts = len(values_2d) // 2
    for i in prange(n_pts):
        result_3d[i * 3]     = values_2d[i * 2]
        result_3d[i * 3 + 1] = values_2d[i * 2 + 1]
        result_3d[i * 3 + 2] = z_val


@ngjit  # pragma: no cover
def segmentize_count(values, offsets0, max_len):
    """Count the total number of output points after segmentizing."""
    total = np.int64(0)
    n = len(offsets0) - 1
    for i in range(n):
        start = offsets0[i];  stop = offsets0[i + 1]
        total += 1
        j = start
        while j < stop - 2:
            dx = values[j + 2] - values[j]
            dy = values[j + 3] - values[j + 1]
            edge_len = sqrt(dx * dx + dy * dy)
            if edge_len > max_len:
                n_segs = int(edge_len / max_len)
                total += n_segs
            else:
                total += 1
            j += 2
    return total


@ngjit  # pragma: no cover
def segmentize_map1(values, offsets0, max_len, result, new_offsets):
    """Segmentize depth-1 geometries: insert points along edges > max_len."""
    n = len(offsets0) - 1
    out_pos = np.int64(0)
    new_offsets[0] = 0
    for i in range(n):
        start = offsets0[i];  stop = offsets0[i + 1]
        result[out_pos * 2]     = values[start]
        result[out_pos * 2 + 1] = values[start + 1]
        out_pos += 1
        j = start
        while j < stop - 2:
            x0 = values[j];     y0 = values[j + 1]
            x1 = values[j + 2]; y1 = values[j + 3]
            dx = x1 - x0;  dy = y1 - y0
            edge_len = sqrt(dx * dx + dy * dy)
            if edge_len > max_len:
                n_segs = int(edge_len / max_len)
                for k in range(1, n_segs + 1):
                    t = k / n_segs
                    result[out_pos * 2]     = x0 + t * dx
                    result[out_pos * 2 + 1] = y0 + t * dy
                    out_pos += 1
            else:
                result[out_pos * 2]     = x1
                result[out_pos * 2 + 1] = y1
                out_pos += 1
            j += 2
        new_offsets[i + 1] = out_pos * 2


@ngjit  # pragma: no cover
def reverse_ring_inplace(values, start, stop):
    """Reverse a ring's vertex order in-place."""
    n_pts = (stop - start) // 2
    for k in range(n_pts // 2):
        a = start + k * 2
        b = stop  - (k + 1) * 2
        tx = values[a];  ty = values[a + 1]
        values[a]     = values[b];  values[a + 1] = values[b + 1]
        values[b]     = tx;         values[b + 1] = ty


@ngjit  # pragma: no cover
def orient_polygons_map2(values, offsets0, offsets1, exterior_cw):
    """Enforce ring orientation for each depth-2 polygon, in-place."""
    n = len(offsets0) - 1
    for i in range(n):
        ring_start_idx = offsets0[i]
        ring_stop_idx  = offsets0[i + 1]
        for r in range(ring_start_idx, ring_stop_idx):
            rs = offsets1[r];  re = offsets1[r + 1]
            area = orient_ring(values, rs, re)
            is_exterior = (r == ring_start_idx)
            if is_exterior:
                want_positive = not exterior_cw
                if (want_positive and area < 0.0) or (not want_positive and area > 0.0):
                    reverse_ring_inplace(values, rs, re)
            else:
                want_positive = exterior_cw
                if (want_positive and area < 0.0) or (not want_positive and area > 0.0):
                    reverse_ring_inplace(values, rs, re)


# ===========================================================================
# Tier-1 predicates
# ===========================================================================

@ngpjit  # pragma: no cover
def is_ring_map1(values, offsets0, result, missing):
    """True if each depth-1 geometry is a closed ring with >= 4 points."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = offsets0[i];  stop = offsets0[i + 1]
            if stop - start < 8:
                result[i] = False
            else:
                result[i] = (values[start]     == values[stop - 2] and
                             values[start + 1] == values[stop - 1])


@ngpjit  # pragma: no cover
def minimum_bounding_radius_map1(values, offsets0, result, missing):
    """Approximate minimum bounding radius for each depth-1 geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            xmin, ymin, xmax, ymax = compute_bounds(values, offsets0[i], offsets0[i + 1])
            dx = xmax - xmin;  dy = ymax - ymin
            result[i] = sqrt(dx * dx + dy * dy) / 2.0


@ngpjit  # pragma: no cover
def minimum_bounding_radius_map2(values, offsets0, offsets1, result, missing):
    """Approximate minimum bounding radius for each depth-2 geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = offsets1[offsets0[i]]
            stop  = offsets1[offsets0[i + 1]]
            xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
            dx = xmax - xmin;  dy = ymax - ymin
            result[i] = sqrt(dx * dx + dy * dy) / 2.0
