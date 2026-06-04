"""test_impl_functions.py — direct tests of the plain-Python _impl functions.

The ``_*_impl`` functions in ``algorithms.py`` are pre-JIT plain Python
functions.  Because they have no numba decorator they are fully traceable by
the coverage tool, unlike the ``@ngjit``/``@ngpjit``-decorated wrappers.

Testing them directly also validates the geometry math independently of the
JIT compilation pipeline.
"""

from math import isclose, inf, nan, sqrt, isnan, isinf

import numpy as np
import pytest


# ===========================================================================
# _compute_area_impl
# ===========================================================================

class TestComputeAreaImpl:

    def _fn(self):
        from akimbo_geo.algorithms import _compute_area_impl
        return _compute_area_impl

    def test_unit_square_ccw(self):
        # CCW unit square: area should be +1.0
        values  = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        offsets = np.array([0, 10], dtype=np.int32)
        result  = self._fn()(values, offsets)
        assert isclose(result, 1.0)

    def test_unit_square_cw(self):
        # CW unit square: area should be -1.0
        values  = np.array([0., 0., 0., 1., 1., 1., 1., 0., 0., 0.])
        offsets = np.array([0, 10], dtype=np.int32)
        result  = self._fn()(values, offsets)
        assert isclose(result, -1.0)

    def test_degenerate_fewer_than_3_points(self):
        # Only 2 points (4 floats) — degenerate, area = 0
        values  = np.array([0., 0., 1., 1.])
        offsets = np.array([0, 4], dtype=np.int32)
        assert self._fn()(values, offsets) == 0.0

    def test_polygon_with_hole(self):
        # 2×2 square (exterior) + 0.5×0.5 interior hole
        # exterior CCW: area +4; hole CW: area -0.25; net +3.75
        exterior = [0., 0., 2., 0., 2., 2., 0., 2., 0., 0.]
        hole     = [0.5, 0.5, 0.5, 1., 1., 1., 1., 0.5, 0.5, 0.5]
        values   = np.array(exterior + hole)
        offsets  = np.array([0, 10, 20], dtype=np.int32)
        result   = self._fn()(values, offsets)
        assert isclose(result, 3.75)

    def test_multiple_rings_empty_first(self):
        # First "ring" has < 6 floats (degenerate, skipped), second is valid.
        # offsets [0,2,10]: ring0 = values[0:2] (1 pt, skip), ring1 = values[2:10]
        # values[2:10] = [1,0, 1,1, 0,1, 0,0] — right-angled triangle, area = 0.5
        values  = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        offsets = np.array([0, 2, 10], dtype=np.int32)
        result  = self._fn()(values, offsets)
        assert isclose(result, 0.5)


# ===========================================================================
# _compute_bounds_impl
# ===========================================================================

class TestComputeBoundsImpl:

    def _fn(self):
        from akimbo_geo.algorithms import _compute_bounds_impl
        return _compute_bounds_impl

    def test_basic(self):
        values = np.array([1., 5., 3., 2., 0., 4.])
        xmin, ymin, xmax, ymax = self._fn()(values, 0, 6)
        assert xmin == 0. and ymin == 2.
        assert xmax == 3. and ymax == 5.

    def test_single_point(self):
        values = np.array([7., 3.])
        xmin, ymin, xmax, ymax = self._fn()(values, 0, 2)
        assert xmin == xmax == 7.
        assert ymin == ymax == 3.

    def test_negative_coords(self):
        values = np.array([-3., -1., 2., 4.])
        xmin, ymin, xmax, ymax = self._fn()(values, 0, 4)
        assert xmin == -3. and xmax == 2.
        assert ymin == -1. and ymax == 4.

    def test_start_stop_slice(self):
        # Only test a sub-range of a larger buffer
        values = np.array([100., 100., 1., 5., 3., 2., 100., 100.])
        xmin, ymin, xmax, ymax = self._fn()(values, 2, 6)
        assert xmin == 1. and xmax == 3.
        assert ymin == 2. and ymax == 5.

    def test_all_same_coords(self):
        values = np.array([2., 2., 2., 2., 2., 2.])
        xmin, ymin, xmax, ymax = self._fn()(values, 0, 6)
        assert xmin == xmax == 2.
        assert ymin == ymax == 2.


# ===========================================================================
# _compute_centroid_line_impl
# ===========================================================================

class TestComputeCentroidLineImpl:

    def _fn(self):
        from akimbo_geo.algorithms import _compute_centroid_line_impl
        return _compute_centroid_line_impl

    def test_two_points(self):
        values = np.array([0., 0., 2., 4.])
        cx, cy = self._fn()(values, 0, 4)
        assert isclose(cx, 1.0) and isclose(cy, 2.0)

    def test_three_points(self):
        # (0,0), (1,0), (1,1) — mean = (2/3, 1/3)
        values = np.array([0., 0., 1., 0., 1., 1.])
        cx, cy = self._fn()(values, 0, 6)
        assert isclose(cx, 2/3, rel_tol=1e-9)
        assert isclose(cy, 1/3, rel_tol=1e-9)

    def test_empty_range_returns_nan(self):
        values = np.array([0., 0.])
        cx, cy = self._fn()(values, 0, 0)
        assert isnan(cx) and isnan(cy)

    def test_sub_range(self):
        # Only use the last two points
        values = np.array([99., 99., 0., 0., 4., 2.])
        cx, cy = self._fn()(values, 2, 6)
        assert isclose(cx, 2.0) and isclose(cy, 1.0)


# ===========================================================================
# _compute_centroid_polygon_impl
# ===========================================================================

class TestComputeCentroidPolygonImpl:

    def _fn(self):
        from akimbo_geo.algorithms import _compute_centroid_polygon_impl
        return _compute_centroid_polygon_impl

    def test_unit_square(self):
        values  = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        offsets = np.array([0, 10], dtype=np.int32)
        cx, cy  = self._fn()(values, offsets)
        assert isclose(cx, 0.5, rel_tol=1e-6)
        assert isclose(cy, 0.5, rel_tol=1e-6)

    def test_right_triangle(self):
        # (0,0),(2,0),(0,2),(0,0): centroid = (2/3, 2/3)
        values  = np.array([0., 0., 2., 0., 0., 2., 0., 0.])
        offsets = np.array([0, 8], dtype=np.int32)
        cx, cy  = self._fn()(values, offsets)
        assert isclose(cx, 2/3, rel_tol=1e-5)
        assert isclose(cy, 2/3, rel_tol=1e-5)

    def test_degenerate_too_few_points_returns_nan(self):
        values  = np.array([0., 0., 1., 1.])  # only 2 points
        offsets = np.array([0, 4], dtype=np.int32)
        cx, cy  = self._fn()(values, offsets)
        assert isnan(cx) and isnan(cy)

    def test_zero_area_collinear_returns_nan(self):
        # All points on x-axis — zero area
        values  = np.array([0., 0., 1., 0., 2., 0., 0., 0.])
        offsets = np.array([0, 8], dtype=np.int32)
        cx, cy  = self._fn()(values, offsets)
        assert isnan(cx) and isnan(cy)

    def test_uses_only_exterior_ring(self):
        # offsets has two rings; centroid must use only the first (exterior)
        exterior = [0., 0., 2., 0., 2., 2., 0., 2., 0., 0.]  # 2×2 square, centroid (1,1)
        hole     = [0.5, 0.5, 0.5, 1., 1., 1., 1., 0.5, 0.5, 0.5]
        values   = np.array(exterior + hole)
        offsets  = np.array([0, 10, 20], dtype=np.int32)
        cx, cy   = self._fn()(values, offsets)
        # Centroid of exterior ring (2×2 square) = (1.0, 1.0)
        assert isclose(cx, 1.0, rel_tol=1e-6)
        assert isclose(cy, 1.0, rel_tol=1e-6)


# ===========================================================================
# _total_bounds_impl
# ===========================================================================

class TestTotalBoundsImpl:

    def _fn(self):
        from akimbo_geo.algorithms import _total_bounds_impl
        return _total_bounds_impl

    def test_basic(self):
        values = np.array([1., 5., 3., 2., 0., 4.])
        xmin, ymin, xmax, ymax = self._fn()(values)
        assert xmin == 0. and ymin == 2.
        assert xmax == 3. and ymax == 5.

    def test_single_point(self):
        values = np.array([7., 3.])
        xmin, ymin, xmax, ymax = self._fn()(values)
        assert xmin == xmax == 7.
        assert ymin == ymax == 3.

    def test_multiple_geometries_flat(self):
        # Simulates two geometries laid end-to-end
        values = np.array([-1., 2., 5., -3., 0., 0., 4., 1.])
        xmin, ymin, xmax, ymax = self._fn()(values)
        assert xmin == -1. and ymin == -3.
        assert xmax == 5.  and ymax == 2.


# ===========================================================================
# _orient_ring_impl
# ===========================================================================

class TestOrientRingImpl:

    def _fn(self):
        from akimbo_geo.algorithms import _orient_ring_impl
        return _orient_ring_impl

    def test_ccw_ring_positive_area(self):
        # CCW unit square: area > 0
        values = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        result = self._fn()(values, 0, 10)
        assert result > 0.0
        assert isclose(result, 1.0)

    def test_cw_ring_negative_area(self):
        # CW unit square: area < 0
        values = np.array([0., 0., 0., 1., 1., 1., 1., 0., 0., 0.])
        result = self._fn()(values, 0, 10)
        assert result < 0.0
        assert isclose(result, -1.0)

    def test_degenerate_less_than_3_points(self):
        values = np.array([0., 0., 1., 1.])
        assert self._fn()(values, 0, 4) == 0.0

    def test_sub_range(self):
        # Ring occupies only part of the buffer
        padding = [99., 99., 99., 99.]
        ring    = [0., 0., 1., 0., 1., 1., 0., 1., 0., 0.]
        values  = np.array(padding + ring + padding)
        start   = 4
        stop    = 4 + 10
        result  = self._fn()(values, start, stop)
        assert isclose(result, 1.0)


# ===========================================================================
# Confirm _impl and JIT wrapper give the same answer
# ===========================================================================

class TestImplMatchesJit:
    """Spot-check that the plain _impl and the ngjit wrapper agree numerically."""

    def test_compute_bounds_agrees(self):
        from akimbo_geo.algorithms import _compute_bounds_impl, compute_bounds
        values = np.array([1., 5., 3., 2., 0., 4.])
        impl_result = _compute_bounds_impl(values, 0, 6)
        jit_result  = compute_bounds(values, 0, 6)
        assert impl_result == jit_result

    def test_compute_area_agrees(self):
        from akimbo_geo.algorithms import _compute_area_impl, compute_area
        values  = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        offsets = np.array([0, 10], dtype=np.int32)
        assert isclose(_compute_area_impl(values, offsets),
                       compute_area(values, offsets))

    def test_compute_centroid_line_agrees(self):
        from akimbo_geo.algorithms import _compute_centroid_line_impl, compute_centroid_line
        values = np.array([0., 0., 2., 4.])
        impl_result = _compute_centroid_line_impl(values, 0, 4)
        jit_result  = compute_centroid_line(values, 0, 4)
        assert impl_result == jit_result

    def test_total_bounds_agrees(self):
        from akimbo_geo.algorithms import _total_bounds_impl, total_bounds
        values = np.array([1., 5., 3., 2., 0., 4.])
        assert _total_bounds_impl(values) == total_bounds(values)

    def test_orient_ring_agrees(self):
        from akimbo_geo.algorithms import _orient_ring_impl, orient_ring
        values = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        assert isclose(_orient_ring_impl(values, 0, 10),
                       orient_ring(values, 0, 10))
