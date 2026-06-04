"""algorithms_gpu.py — CUDA GPU kernels for geometry operations.

Scalar device functions are created by applying ``cuda.jit(device=True)``
to the *same* plain Python ``_*_impl`` functions defined in ``algorithms.py``.
This guarantees CPU and GPU flavours are compiled from **identical source**
with no duplication.

Map kernels (one thread per geometry) are defined here using the CUDA
``cuda.grid(1)`` idiom and call the shared device functions.  Each kernel
has a ``launch_*`` Python wrapper that calculates grid/block dimensions
and invokes the kernel.

Sequential kernels (``segmentize``, ``orient_polygons``) are not ported here;
the accessor falls back to CPU for those.

Requirements
------------
- ``numba.cuda`` — part of the standard ``numba`` package
- A CUDA-capable GPU + CUDA toolkit
- ``cupy`` for device-side array allocation

Lazy import
-----------
This module is imported only when ``_compat.is_gpu_array(values)`` returns
True; it is never loaded on CPU-only systems.
"""

from __future__ import annotations

from math import sqrt

from numba import cuda

# Import the shared plain-Python implementations from algorithms.py
from akimbo_geo.algorithms import (
    _compute_area_impl,
    _compute_bounds_impl,
    _compute_centroid_line_impl,
    _compute_centroid_polygon_impl,
    _orient_ring_impl,
    _total_bounds_impl,
)


# ===========================================================================
# Scalar device functions — same logic as CPU, different decorator
# ===========================================================================

compute_area             = cuda.jit(device=True)(_compute_area_impl)             # pragma: no cover
compute_bounds           = cuda.jit(device=True)(_compute_bounds_impl)           # pragma: no cover
compute_centroid_line    = cuda.jit(device=True)(_compute_centroid_line_impl)    # pragma: no cover
compute_centroid_polygon = cuda.jit(device=True)(_compute_centroid_polygon_impl) # pragma: no cover
orient_ring              = cuda.jit(device=True)(_orient_ring_impl)              # pragma: no cover


# compute_line_length has a different GPU call signature (explicit start/stop
# instead of an offsets slice), so it is defined directly here.
@cuda.jit(device=True)
def compute_line_length(values, start, stop):  # pragma: no cover
    """Euclidean length of a single line segment [start, stop)."""
    total = 0.0
    j = start
    while j < stop - 2:
        dx = values[j + 2] - values[j]
        dy = values[j + 3] - values[j + 1]
        total += sqrt(dx * dx + dy * dy)
        j += 2
    return total


# ===========================================================================
# CUDA grid kernels — depth-1 (list<float>)
# ===========================================================================

@cuda.jit
def _length_map1_kernel(values, offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        result[i] = compute_line_length(values, offsets0[i], offsets0[i + 1])


@cuda.jit
def _bounds_map1_kernel(values, offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        xmin, ymin, xmax, ymax = compute_bounds(values, offsets0[i], offsets0[i + 1])
        result[i, 0] = xmin; result[i, 1] = ymin
        result[i, 2] = xmax; result[i, 3] = ymax


@cuda.jit
def _centroid_map1_kernel(values, offsets0, result_x, result_y, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        cx, cy = compute_centroid_line(values, offsets0[i], offsets0[i + 1])
        result_x[i] = cx; result_y[i] = cy


@cuda.jit
def _is_closed_map1_kernel(values, offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets0[i]; stop = offsets0[i + 1]
        if stop - start < 4:
            result[i] = False
        else:
            result[i] = (values[start] == values[stop - 2] and
                         values[start + 1] == values[stop - 1])


@cuda.jit
def _is_ring_map1_kernel(values, offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets0[i]; stop = offsets0[i + 1]
        if stop - start < 8:
            result[i] = False
        else:
            result[i] = (values[start] == values[stop - 2] and
                         values[start + 1] == values[stop - 1])


@cuda.jit
def _count_coords_map1_kernel(offsets0, n_dims, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        result[i] = (offsets0[i + 1] - offsets0[i]) // n_dims


@cuda.jit
def _translate_map_kernel(values, offsets0, xoff, yoff, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets0[i]; stop = offsets0[i + 1]
        j = start
        while j < stop:
            result[j]     = values[j]     + xoff
            result[j + 1] = values[j + 1] + yoff
            j += 2


@cuda.jit
def _scale_map_kernel(values, offsets0, xfact, yfact, ox, oy, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets0[i]; stop = offsets0[i + 1]
        j = start
        while j < stop:
            result[j]     = (values[j]     - ox) * xfact + ox
            result[j + 1] = (values[j + 1] - oy) * yfact + oy
            j += 2


@cuda.jit
def _affine_transform_map_kernel(values, offsets0, a, b, d, e, xoff, yoff, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets0[i]; stop = offsets0[i + 1]
        j = start
        while j < stop:
            x = values[j]; y = values[j + 1]
            result[j]     = a * x + b * y + xoff
            result[j + 1] = d * x + e * y + yoff
            j += 2


@cuda.jit
def _reverse_map1_kernel(values, offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets0[i]; stop = offsets0[i + 1]
        n_pts = (stop - start) // 2
        for k in range(n_pts):
            src = start + k * 2
            dst = stop  - (k + 1) * 2
            result[src]     = values[dst]
            result[src + 1] = values[dst + 1]


@cuda.jit
def _minimum_bounding_radius_map1_kernel(values, offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        xmin, ymin, xmax, ymax = compute_bounds(values, offsets0[i], offsets0[i + 1])
        dx = xmax - xmin; dy = ymax - ymin
        result[i] = sqrt(dx * dx + dy * dy) / 2.0


@cuda.jit
def _force_2d_map_kernel(values_nd, n_dims, result_2d, n_pts):  # pragma: no cover
    i = cuda.grid(1)
    if i < n_pts:
        result_2d[i * 2]     = values_nd[i * n_dims]
        result_2d[i * 2 + 1] = values_nd[i * n_dims + 1]


@cuda.jit
def _force_3d_map_kernel(values_2d, z_val, result_3d, n_pts):  # pragma: no cover
    i = cuda.grid(1)
    if i < n_pts:
        result_3d[i * 3]     = values_2d[i * 2]
        result_3d[i * 3 + 1] = values_2d[i * 2 + 1]
        result_3d[i * 3 + 2] = z_val


@cuda.jit
def _get_x_map0_kernel(values, n_dims, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        result[i] = values[i * n_dims]


@cuda.jit
def _get_y_map0_kernel(values, n_dims, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        result[i] = values[i * n_dims + 1]


@cuda.jit
def _get_z_map0_kernel(values, n_dims, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        if n_dims >= 3:
            result[i] = values[i * n_dims + 2]
        # else leave as NaN (result pre-filled by caller)


# ===========================================================================
# CUDA grid kernels — depth-2 (list<list<float>>)
# ===========================================================================

@cuda.jit
def _area_map2_kernel(values, offsets0, offsets1, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        area = 0.0
        for r in range(offsets0[i], offsets0[i + 1]):
            area += compute_area(values, offsets1[r:r + 2])
        result[i] = area


@cuda.jit
def _length_map2_kernel(values, offsets0, offsets1, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets1[offsets0[i]]
        stop  = offsets1[offsets0[i + 1]]
        result[i] = compute_line_length(values, start, stop)


@cuda.jit
def _bounds_map2_kernel(values, offsets0, offsets1, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets1[offsets0[i]]
        stop  = offsets1[offsets0[i + 1]]
        xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
        result[i, 0] = xmin; result[i, 1] = ymin
        result[i, 2] = xmax; result[i, 3] = ymax


@cuda.jit
def _centroid_map2_kernel(values, offsets0, offsets1, result_x, result_y, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        # exterior ring only: first ring of this polygon
        ring_start = offsets1[offsets0[i]]
        ring_stop  = offsets1[offsets0[i] + 1]
        cx, cy = compute_centroid_polygon(values, offsets1[offsets0[i]:offsets0[i] + 2])
        result_x[i] = cx; result_y[i] = cy


@cuda.jit
def _is_ccw_map2_kernel(values, offsets0, offsets1, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        area = compute_area(values, offsets1[offsets0[i]:offsets0[i] + 2])
        result[i] = area > 0.0


@cuda.jit
def _count_geoms_map2_kernel(offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        result[i] = offsets0[i + 1] - offsets0[i]


@cuda.jit
def _count_interior_rings_map2_kernel(offsets0, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        nr = offsets0[i + 1] - offsets0[i]
        result[i] = nr - 1 if nr > 0 else 0


@cuda.jit
def _minimum_bounding_radius_map2_kernel(values, offsets0, offsets1, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets1[offsets0[i]]
        stop  = offsets1[offsets0[i + 1]]
        xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
        dx = xmax - xmin; dy = ymax - ymin
        result[i] = sqrt(dx * dx + dy * dy) / 2.0


# ===========================================================================
# CUDA grid kernels — depth-3 (list<list<list<float>>>)
# ===========================================================================

@cuda.jit
def _area_map3_kernel(values, offsets0, offsets1, offsets2, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        area = 0.0
        for p in range(offsets1[offsets0[i]], offsets1[offsets0[i + 1]]):
            area += compute_area(values, offsets2[p:p + 2])
        result[i] = area


@cuda.jit
def _length_map3_kernel(values, offsets0, offsets1, offsets2, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets2[offsets1[offsets0[i]]]
        stop  = offsets2[offsets1[offsets0[i + 1]]]
        result[i] = compute_line_length(values, start, stop)


@cuda.jit
def _bounds_map3_kernel(values, offsets0, offsets1, offsets2, result, n):  # pragma: no cover
    i = cuda.grid(1)
    if i < n:
        start = offsets2[offsets1[offsets0[i]]]
        stop  = offsets2[offsets1[offsets0[i + 1]]]
        xmin, ymin, xmax, ymax = compute_bounds(values, start, stop)
        result[i, 0] = xmin; result[i, 1] = ymin
        result[i, 2] = xmax; result[i, 3] = ymax


# ===========================================================================
# Launch helpers — Python-callable wrappers
# ===========================================================================

_THREADS = 256


def _grid(n: int):
    return (n + _THREADS - 1) // _THREADS, _THREADS


# --- depth-1 ---

def launch_length_map1(values, offsets0, result):
    n = len(offsets0) - 1
    _length_map1_kernel[_grid(n)](values, offsets0, result, n)

def launch_bounds_map1(values, offsets0, result):
    n = len(offsets0) - 1
    _bounds_map1_kernel[_grid(n)](values, offsets0, result, n)

def launch_centroid_map1(values, offsets0, result_x, result_y):
    n = len(offsets0) - 1
    _centroid_map1_kernel[_grid(n)](values, offsets0, result_x, result_y, n)

def launch_count_coords_map1(offsets0, n_dims, result):
    n = len(offsets0) - 1
    _count_coords_map1_kernel[_grid(n)](offsets0, n_dims, result, n)

def launch_is_closed_map1(values, offsets0, result):
    n = len(offsets0) - 1
    _is_closed_map1_kernel[_grid(n)](values, offsets0, result, n)

def launch_is_ring_map1(values, offsets0, result):
    n = len(offsets0) - 1
    _is_ring_map1_kernel[_grid(n)](values, offsets0, result, n)

def launch_translate_map(values, offsets0, xoff, yoff, result):
    n = len(offsets0) - 1
    _translate_map_kernel[_grid(n)](values, offsets0, xoff, yoff, result, n)

def launch_scale_map(values, offsets0, xfact, yfact, ox, oy, result):
    n = len(offsets0) - 1
    _scale_map_kernel[_grid(n)](values, offsets0, xfact, yfact, ox, oy, result, n)

def launch_affine_transform_map(values, offsets0, a, b, d, e, xoff, yoff, result):
    n = len(offsets0) - 1
    _affine_transform_map_kernel[_grid(n)](
        values, offsets0, a, b, d, e, xoff, yoff, result, n
    )

def launch_reverse_map1(values, offsets0, result):
    n = len(offsets0) - 1
    _reverse_map1_kernel[_grid(n)](values, offsets0, result, n)

def launch_minimum_bounding_radius_map1(values, offsets0, result):
    n = len(offsets0) - 1
    _minimum_bounding_radius_map1_kernel[_grid(n)](values, offsets0, result, n)

def launch_get_x_map0(values, n_dims, result):
    n = len(values) // n_dims
    _get_x_map0_kernel[_grid(n)](values, n_dims, result, n)

def launch_get_y_map0(values, n_dims, result):
    n = len(values) // n_dims
    _get_y_map0_kernel[_grid(n)](values, n_dims, result, n)

def launch_get_z_map0(values, n_dims, result):
    n = len(values) // n_dims
    _get_z_map0_kernel[_grid(n)](values, n_dims, result, n)

def launch_force_2d_map(values_nd, n_dims, result_2d):
    n_pts = len(values_nd) // n_dims
    _force_2d_map_kernel[_grid(n_pts)](values_nd, n_dims, result_2d, n_pts)

def launch_force_3d_map(values_2d, z_val, result_3d):
    n_pts = len(values_2d) // 2
    _force_3d_map_kernel[_grid(n_pts)](values_2d, z_val, result_3d, n_pts)

# --- depth-2 ---

def launch_area_map2(values, offsets0, offsets1, result):
    n = len(offsets0) - 1
    _area_map2_kernel[_grid(n)](values, offsets0, offsets1, result, n)

def launch_length_map2(values, offsets0, offsets1, result):
    n = len(offsets0) - 1
    _length_map2_kernel[_grid(n)](values, offsets0, offsets1, result, n)

def launch_bounds_map2(values, offsets0, offsets1, result):
    n = len(offsets0) - 1
    _bounds_map2_kernel[_grid(n)](values, offsets0, offsets1, result, n)

def launch_centroid_map2(values, offsets0, offsets1, result_x, result_y):
    n = len(offsets0) - 1
    _centroid_map2_kernel[_grid(n)](values, offsets0, offsets1, result_x, result_y, n)

def launch_is_ccw_map2(values, offsets0, offsets1, result):
    n = len(offsets0) - 1
    _is_ccw_map2_kernel[_grid(n)](values, offsets0, offsets1, result, n)

def launch_count_geoms_map2(offsets0, result):
    n = len(offsets0) - 1
    _count_geoms_map2_kernel[_grid(n)](offsets0, result, n)

def launch_count_interior_rings_map2(offsets0, result):
    n = len(offsets0) - 1
    _count_interior_rings_map2_kernel[_grid(n)](offsets0, result, n)

def launch_minimum_bounding_radius_map2(values, offsets0, offsets1, result):
    n = len(offsets0) - 1
    _minimum_bounding_radius_map2_kernel[_grid(n)](values, offsets0, offsets1, result, n)

# --- depth-3 ---

def launch_area_map3(values, offsets0, offsets1, offsets2, result):
    n = len(offsets0) - 1
    _area_map3_kernel[_grid(n)](values, offsets0, offsets1, offsets2, result, n)

def launch_length_map3(values, offsets0, offsets1, offsets2, result):
    n = len(offsets0) - 1
    _length_map3_kernel[_grid(n)](values, offsets0, offsets1, offsets2, result, n)

def launch_bounds_map3(values, offsets0, offsets1, offsets2, result):
    n = len(offsets0) - 1
    _bounds_map3_kernel[_grid(n)](values, offsets0, offsets1, offsets2, result, n)
