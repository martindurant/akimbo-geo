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


# ===========================================================================
# Counting kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def count_coords_map1(offsets0, n_dims, result, missing):
    """Number of coordinate *points* per depth-1 geometry.

    For INTERLEAVED_FLAT offsets index floats, so divide by n_dims.
    For FSL/STRUCT offsets already index points; n_dims still passed for
    uniformity but the division is identical either way because the op
    functions pass float-unit offsets in all cases.
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            result[i] = (offsets0[i + 1] - offsets0[i]) // n_dims


@ngpjit  # pragma: no cover
def count_geoms_map2(offsets0, result, missing):
    """Number of sub-geometries per depth-2 geometry (rings per polygon, etc.)."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            result[i] = offsets0[i + 1] - offsets0[i]


@ngpjit  # pragma: no cover
def count_interior_rings_map2(offsets0, result, missing):
    """Number of interior rings (holes) per polygon.

    First ring is the exterior; all subsequent are holes.
    """
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
            start = offsets0[i]
            stop  = offsets0[i + 1]
            if stop - start < 4:
                # Fewer than 2 points — cannot be closed
                result[i] = False
            else:
                result[i] = (values[start]     == values[stop - 2] and
                             values[start + 1] == values[stop - 1])


@ngpjit  # pragma: no cover
def is_ccw_map2(values, offsets0, offsets1, result, missing):
    """True if exterior ring of each depth-2 polygon is counter-clockwise.

    Uses the sign of the shoelace area: positive area → CCW.
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            # Exterior ring is the first ring of this polygon
            ring_start_idx = offsets0[i]
            inner = offsets1[ring_start_idx:ring_start_idx + 2]
            area = compute_area(values, inner)
            result[i] = area > 0.0


# ===========================================================================
# Coordinate extraction kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def get_x_map0(values, n_dims, result):
    """Extract x coordinate for each depth-0 Point geometry.

    values  : flat float buffer (one point per entry-pair or entry-triple etc.)
    n_dims  : stride (2 for XY, 3 for XYZ, ...)
    result  : (N,) output array
    """
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
            result[i] = np.nan


# ===========================================================================
# Affine transformation kernels
# ===========================================================================

@ngpjit  # pragma: no cover
def translate_map(values, offsets0, xoff, yoff, result):
    """Add (xoff, yoff) to every coordinate in each depth-1 geometry.

    Stride is always 2 (interleaved flat; FSL/STRUCT have been normalised).
    result must be pre-allocated to the same size as values.
    """
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i]
        stop  = offsets0[i + 1]
        j = start
        while j < stop:
            result[j]     = values[j]     + xoff
            result[j + 1] = values[j + 1] + yoff
            j += 2


@ngpjit  # pragma: no cover
def scale_map(values, offsets0, xfact, yfact, ox, oy, result):
    """Scale each coordinate around origin (ox, oy).

    result[x] = (values[x] - ox) * xfact + ox
    result[y] = (values[y] - oy) * yfact + oy
    """
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i]
        stop  = offsets0[i + 1]
        j = start
        while j < stop:
            result[j]     = (values[j]     - ox) * xfact + ox
            result[j + 1] = (values[j + 1] - oy) * yfact + oy
            j += 2


@ngpjit  # pragma: no cover
def affine_transform_map(values, offsets0, a, b, d, e, xoff, yoff, result):
    """Apply a 2D affine transform to every coordinate.

    Matrix form:  [x']   [a  b] [x]   [xoff]
                  [y'] = [d  e] [y] + [yoff]

    Parameters match the geopandas / shapely affine_transform 6-element
    convention: (a, b, d, e, xoff, yoff).
    """
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i]
        stop  = offsets0[i + 1]
        j = start
        while j < stop:
            x = values[j]
            y = values[j + 1]
            result[j]     = a * x + b * y + xoff
            result[j + 1] = d * x + e * y + yoff
            j += 2


# ===========================================================================
# Coordinate manipulation kernels (produce new geometry arrays)
# ===========================================================================

@ngpjit  # pragma: no cover
def reverse_map1(values, offsets0, result):
    """Reverse the vertex order of each depth-1 geometry.

    result is a new flat buffer of the same size as values.
    Coordinate *pairs* are reversed (not individual floats), preserving
    interleaved XY structure.
    """
    n = len(offsets0) - 1
    for i in prange(n):
        start = offsets0[i]
        stop  = offsets0[i + 1]
        n_pts = (stop - start) // 2
        for k in range(n_pts):
            src = start + k * 2
            dst = stop  - (k + 1) * 2
            result[src]     = values[dst]
            result[src + 1] = values[dst + 1]


@ngpjit  # pragma: no cover
def force_2d_map(values_nd, n_dims, result_2d):
    """Copy only (x, y) from an n-dimensional flat buffer into a 2D buffer.

    values_nd : flat buffer with n_dims floats per point
    result_2d : pre-allocated flat buffer with 2 floats per point
    """
    n_pts = len(values_nd) // n_dims
    for i in prange(n_pts):
        result_2d[i * 2]     = values_nd[i * n_dims]
        result_2d[i * 2 + 1] = values_nd[i * n_dims + 1]


@ngpjit  # pragma: no cover
def force_3d_map(values_2d, z_val, result_3d):
    """Promote a 2D flat buffer to 3D by inserting a constant z value.

    result_3d : pre-allocated flat buffer with 3 floats per point
    """
    n_pts = len(values_2d) // 2
    for i in prange(n_pts):
        result_3d[i * 3]     = values_2d[i * 2]
        result_3d[i * 3 + 1] = values_2d[i * 2 + 1]
        result_3d[i * 3 + 2] = z_val


@ngjit  # pragma: no cover
def segmentize_count(values, offsets0, max_len):
    """Count the total number of output points after segmentizing.

    For each edge longer than max_len we insert floor(edge_length/max_len)
    intermediate points.  This first pass determines the new buffer size.

    Returns the new total number of coordinate *points* (not floats).
    """
    total = np.int64(0)
    n = len(offsets0) - 1
    for i in range(n):
        start = offsets0[i]
        stop  = offsets0[i + 1]
        # Always keep the first point of this geometry
        total += 1
        j = start
        while j < stop - 2:
            dx = values[j + 2] - values[j]
            dy = values[j + 3] - values[j + 1]
            edge_len = sqrt(dx * dx + dy * dy)
            # Number of *new* segments = ceil(edge_len / max_len)
            # = floor((edge_len - eps) / max_len) + 1
            if edge_len > max_len:
                n_segs = int(edge_len / max_len)
                # intermediate points = n_segs - 1, plus the endpoint
                total += n_segs
            else:
                total += 1
            j += 2
    return total


@ngjit  # pragma: no cover
def segmentize_map1(values, offsets0, max_len, result, new_offsets):
    """Segmentize depth-1 geometries: insert points along edges > max_len.

    result      : pre-allocated flat output float buffer (sized by segmentize_count)
    new_offsets : (N+1,) int64 output offset array
    """
    n = len(offsets0) - 1
    out_pos = np.int64(0)
    new_offsets[0] = 0
    for i in range(n):
        start = offsets0[i]
        stop  = offsets0[i + 1]
        # Copy first point
        result[out_pos * 2]     = values[start]
        result[out_pos * 2 + 1] = values[start + 1]
        out_pos += 1
        j = start
        while j < stop - 2:
            x0 = values[j];     y0 = values[j + 1]
            x1 = values[j + 2]; y1 = values[j + 3]
            dx = x1 - x0
            dy = y1 - y0
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
def orient_ring(values, start, stop):
    """Return the signed area of a ring; used to check/enforce orientation.

    Positive → CCW (exterior convention), negative → CW (hole convention).
    """
    area = 0.0
    n_pts = (stop - start) // 2
    if n_pts < 3:
        return 0.0
    for k in range(n_pts - 1):
        i = start + k * 2
        x0 = values[i];     y0 = values[i + 1]
        x1 = values[i + 2]; y1 = values[i + 3]
        area += x0 * y1 - x1 * y0
    # Close
    x0 = values[stop - 2]; y0 = values[stop - 1]
    x1 = values[start];    y1 = values[start + 1]
    area += x0 * y1 - x1 * y0
    return area / 2.0


@ngjit  # pragma: no cover
def reverse_ring_inplace(values, start, stop):
    """Reverse a ring's vertex order in-place (swap pairs around mid-point)."""
    n_pts = (stop - start) // 2
    for k in range(n_pts // 2):
        a = start + k * 2
        b = stop  - (k + 1) * 2
        tx = values[a];     ty = values[a + 1]
        values[a]     = values[b];     values[a + 1] = values[b + 1]
        values[b]     = tx;            values[b + 1] = ty


@ngjit  # pragma: no cover
def orient_polygons_map2(values, offsets0, offsets1, exterior_cw):
    """Enforce ring orientation for each depth-2 polygon, in-place.

    exterior_cw : if True, exterior ring is forced CW (area < 0);
                  if False (default / geopandas convention), exterior is CCW.
    Hole rings are always forced to the opposite orientation.
    """
    n = len(offsets0) - 1
    for i in range(n):          # sequential: modifies values in-place
        ring_start_idx = offsets0[i]
        ring_stop_idx  = offsets0[i + 1]
        for r in range(ring_start_idx, ring_stop_idx):
            rs = offsets1[r]
            re = offsets1[r + 1]
            area = orient_ring(values, rs, re)
            is_exterior = (r == ring_start_idx)
            if is_exterior:
                # Exterior: want CCW (area > 0) unless exterior_cw requested
                want_positive = not exterior_cw
                if (want_positive and area < 0.0) or (not want_positive and area > 0.0):
                    reverse_ring_inplace(values, rs, re)
            else:
                # Hole: want CW (area < 0) unless exterior_cw requested
                want_positive = exterior_cw
                if (want_positive and area < 0.0) or (not want_positive and area > 0.0):
                    reverse_ring_inplace(values, rs, re)


# ===========================================================================
# Tier-1 pure-numba additions (no shapely required)
# ===========================================================================

@ngpjit  # pragma: no cover
def is_ring_map1(values, offsets0, result, missing):
    """True if each depth-1 geometry is a valid ring (closed, >= 4 points).

    A ring must be:
    - closed: first coord == last coord
    - have at least 4 coordinate *pairs* (3 distinct vertices + closing repeat)

    This is the minimum necessary check without a self-intersection test.
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = offsets0[i]
            stop  = offsets0[i + 1]
            n_floats = stop - start
            if n_floats < 8:           # < 4 coordinate pairs
                result[i] = False
            else:
                result[i] = (values[start]     == values[stop - 2] and
                             values[start + 1] == values[stop - 1])


@ngpjit  # pragma: no cover
def minimum_bounding_radius_map1(values, offsets0, result, missing):
    """Approximate minimum bounding radius for each depth-1 geometry.

    Returns half the diagonal of the bounding box, which is an upper bound
    on the true minimum bounding circle radius.  Exact computation requires
    Welzl's algorithm (O(n) expected) which is available via shapely; this
    fast approximation is useful for quick spatial filtering.

    result[i] = sqrt((xmax-xmin)^2 + (ymax-ymin)^2) / 2
    """
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            xmin, ymin, xmax, ymax = compute_bounds(
                values, offsets0[i], offsets0[i + 1]
            )
            dx = xmax - xmin
            dy = ymax - ymin
            result[i] = sqrt(dx * dx + dy * dy) / 2.0


@ngpjit  # pragma: no cover
def minimum_bounding_radius_map2(values, offsets0, offsets1, result, missing):
    """Approximate minimum bounding radius for each depth-2 (polygon) geometry."""
    n = len(offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = offsets1[offsets0[i]]
            stop  = offsets1[offsets0[i + 1]]
            xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
            dx = xmax - xmin
            dy = ymax - ymin
            result[i] = sqrt(dx * dx + dy * dy) / 2.0
