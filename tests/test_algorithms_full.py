"""test_algorithms_full.py — full coverage of algorithms.py numba kernels.

Tests the map2/map3 vectorised kernels, centroid kernels,
intersects_bounds, _segments_intersect, _point_in_ring, and total_bounds.
"""

from math import isclose, sqrt

import numpy as np
import pytest


# ===========================================================================
# 2-level map kernels (Polygon / MultiLine)
# ===========================================================================

class TestMap2Kernels:
    """Test the *_map2 parallel kernels directly."""

    def _unit_square_inputs(self):
        """Return (values, offsets0, offsets1) for one unit-square polygon."""
        # CCW unit square: (0,0),(1,0),(1,1),(0,1),(0,0)
        values  = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        offsets0 = np.array([0, 1], dtype=np.int32)   # 1 polygon → 1 ring slice
        offsets1 = np.array([0, 10], dtype=np.int32)  # ring: all 10 floats
        return values, offsets0, offsets1

    def test_area_map2_unit_square(self):
        from akimbo_geo.algorithms import area_map2
        values, off0, off1 = self._unit_square_inputs()
        result  = np.full(1, np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        area_map2(values, off0, off1, result, missing)
        assert isclose(abs(result[0]), 1.0)

    def test_area_map2_missing_skipped(self):
        from akimbo_geo.algorithms import area_map2
        values, off0, off1 = self._unit_square_inputs()
        result  = np.full(1, np.nan)
        missing = np.ones(1, dtype=np.bool_)   # all missing
        area_map2(values, off0, off1, result, missing)
        assert np.isnan(result[0])

    def test_area_map2_two_polygons(self):
        """Two polygons: unit square and 2×2 square."""
        from akimbo_geo.algorithms import area_map2
        # 2×2 square: (0,0),(2,0),(2,2),(0,2),(0,0)
        sq2 = np.array([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0])
        sq1 = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        values  = np.concatenate([sq1, sq2])
        offsets0 = np.array([0, 1, 2], dtype=np.int32)
        offsets1 = np.array([0, 10, 20], dtype=np.int32)
        result  = np.full(2, np.nan)
        missing = np.zeros(2, dtype=np.bool_)
        area_map2(values, offsets0, offsets1, result, missing)
        assert isclose(abs(result[0]), 1.0)
        assert isclose(abs(result[1]), 4.0)

    def test_length_map2_multiline(self):
        """MultiLine: two line segments per geometry."""
        from akimbo_geo.algorithms import length_map2
        # (0,0)→(1,0) len=1; (0,0)→(3,4) len=5
        values  = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0])
        offsets0 = np.array([0, 2], dtype=np.int32)    # 1 multiline → 2 sub-lines
        offsets1 = np.array([0, 4, 8], dtype=np.int32) # line0: floats 0-3, line1: 4-7
        result  = np.full(1, np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        length_map2(values, offsets0, offsets1, result, missing)
        assert isclose(result[0], 1.0 + 5.0)

    def test_bounds_map2(self):
        from akimbo_geo.algorithms import bounds_map2
        values, off0, off1 = self._unit_square_inputs()
        result  = np.full((1, 4), np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        bounds_map2(values, off0, off1, result, missing)
        assert result[0, 0] == 0.0  # xmin
        assert result[0, 1] == 0.0  # ymin
        assert result[0, 2] == 1.0  # xmax
        assert result[0, 3] == 1.0  # ymax

    def test_centroid_map2_unit_square(self):
        from akimbo_geo.algorithms import centroid_map2
        values, off0, off1 = self._unit_square_inputs()
        result_x = np.full(1, np.nan)
        result_y = np.full(1, np.nan)
        missing  = np.zeros(1, dtype=np.bool_)
        centroid_map2(values, off0, off1, result_x, result_y, missing)
        # Centroid of unit square should be (0.5, 0.5)
        assert isclose(result_x[0], 0.5, rel_tol=1e-6)
        assert isclose(result_y[0], 0.5, rel_tol=1e-6)

    def test_centroid_map2_missing(self):
        from akimbo_geo.algorithms import centroid_map2
        values, off0, off1 = self._unit_square_inputs()
        result_x = np.full(1, np.nan)
        result_y = np.full(1, np.nan)
        missing  = np.ones(1, dtype=np.bool_)
        centroid_map2(values, off0, off1, result_x, result_y, missing)
        assert np.isnan(result_x[0])


# ===========================================================================
# 3-level map kernels (MultiPolygon)
# ===========================================================================

class TestMap3Kernels:
    """Test the *_map3 parallel kernels directly."""

    def _two_unit_squares_inputs(self):
        """MultiPolygon: two unit squares side-by-side, total area = 2."""
        # Square 1: (0,0),(1,0),(1,1),(0,1),(0,0)
        sq1 = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        # Square 2: (2,0),(3,0),(3,1),(2,1),(2,0)
        sq2 = np.array([2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0])
        values   = np.concatenate([sq1, sq2])
        # offsets0: 1 multipolygon → polygons 0 and 1
        offsets0 = np.array([0, 2], dtype=np.int32)
        # offsets1: each polygon → 1 ring
        offsets1 = np.array([0, 1, 2], dtype=np.int32)
        # offsets2: each ring → floats
        offsets2 = np.array([0, 10, 20], dtype=np.int32)
        return values, offsets0, offsets1, offsets2

    def test_area_map3(self):
        from akimbo_geo.algorithms import area_map3
        values, off0, off1, off2 = self._two_unit_squares_inputs()
        result  = np.full(1, np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        area_map3(values, off0, off1, off2, result, missing)
        assert isclose(abs(result[0]), 2.0)

    def test_area_map3_missing(self):
        from akimbo_geo.algorithms import area_map3
        values, off0, off1, off2 = self._two_unit_squares_inputs()
        result  = np.full(1, np.nan)
        missing = np.ones(1, dtype=np.bool_)
        area_map3(values, off0, off1, off2, result, missing)
        assert np.isnan(result[0])

    def test_length_map3(self):
        from akimbo_geo.algorithms import length_map3
        values, off0, off1, off2 = self._two_unit_squares_inputs()
        result  = np.full(1, np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        length_map3(values, off0, off1, off2, result, missing)
        # Each square ring has perimeter 4; total = 8
        assert isclose(result[0], 8.0)

    def test_bounds_map3(self):
        from akimbo_geo.algorithms import bounds_map3
        values, off0, off1, off2 = self._two_unit_squares_inputs()
        result  = np.full((1, 4), np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        bounds_map3(values, off0, off1, off2, result, missing)
        assert result[0, 0] == 0.0  # xmin
        assert result[0, 3] == 1.0  # ymax
        assert result[0, 2] == 3.0  # xmax


# ===========================================================================
# Centroid kernels
# ===========================================================================

class TestCentroidKernels:

    def test_compute_centroid_polygon_unit_square(self):
        from akimbo_geo.algorithms import compute_centroid_polygon
        # CCW unit square
        values  = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        offsets = np.array([0, 10], dtype=np.int32)
        cx, cy  = compute_centroid_polygon(values, offsets)
        assert isclose(cx, 0.5, rel_tol=1e-6)
        assert isclose(cy, 0.5, rel_tol=1e-6)

    def test_compute_centroid_polygon_degenerate(self):
        """Fewer than 3 points → NaN."""
        from akimbo_geo.algorithms import compute_centroid_polygon
        values  = np.array([0.0, 0.0, 1.0, 1.0])  # only 2 points
        offsets = np.array([0, 4], dtype=np.int32)
        cx, cy  = compute_centroid_polygon(values, offsets)
        assert np.isnan(cx) and np.isnan(cy)

    def test_compute_centroid_polygon_zero_area(self):
        """Collinear ring → zero area → NaN centroid."""
        from akimbo_geo.algorithms import compute_centroid_polygon
        # All points on x-axis
        values  = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        offsets = np.array([0, 8], dtype=np.int32)
        cx, cy  = compute_centroid_polygon(values, offsets)
        assert np.isnan(cx) and np.isnan(cy)

    def test_compute_centroid_line_empty(self):
        """Zero points → NaN centroid."""
        from akimbo_geo.algorithms import compute_centroid_line
        values = np.array([], dtype=np.float64)
        cx, cy = compute_centroid_line(values, 0, 0)
        assert np.isnan(cx) and np.isnan(cy)


# ===========================================================================
# Intersection / predicate kernels
# ===========================================================================

class TestIntersectsKernels:

    def test_intersects_bounds_vertex_inside(self):
        """Line with a vertex inside the box → True."""
        from akimbo_geo.algorithms import intersects_bounds_map1
        # Line: (0.5, 0.5) → (10, 10); point (0.5, 0.5) is inside [0,0,1,1]
        values  = np.array([0.5, 0.5, 10.0, 10.0])
        offsets = np.array([0, 4], dtype=np.int32)
        result  = np.zeros(1, dtype=np.bool_)
        missing = np.zeros(1, dtype=np.bool_)
        intersects_bounds_map1(0.0, 0.0, 1.0, 1.0, values, offsets, result, missing)
        assert result[0]

    def test_intersects_bounds_edge_crosses(self):
        """Line that crosses a bbox edge without any vertex inside → True."""
        from akimbo_geo.algorithms import intersects_bounds_map1
        # Horizontal line y=0.5, from x=-1 to x=2; crosses left/right edges of [0,0,1,1]
        values  = np.array([-1.0, 0.5, 2.0, 0.5])
        offsets = np.array([0, 4], dtype=np.int32)
        result  = np.zeros(1, dtype=np.bool_)
        missing = np.zeros(1, dtype=np.bool_)
        intersects_bounds_map1(0.0, 0.0, 1.0, 1.0, values, offsets, result, missing)
        assert result[0]

    def test_intersects_bounds_fully_outside(self):
        """Line entirely outside → False."""
        from akimbo_geo.algorithms import intersects_bounds_map1
        values  = np.array([5.0, 5.0, 10.0, 10.0])
        offsets = np.array([0, 4], dtype=np.int32)
        result  = np.zeros(1, dtype=np.bool_)
        missing = np.zeros(1, dtype=np.bool_)
        intersects_bounds_map1(0.0, 0.0, 1.0, 1.0, values, offsets, result, missing)
        assert not result[0]

    def test_intersects_bounds_missing(self):
        """Missing geometry → result left as initialised (False)."""
        from akimbo_geo.algorithms import intersects_bounds_map1
        values  = np.array([0.5, 0.5, 0.5, 0.5])
        offsets = np.array([0, 4], dtype=np.int32)
        result  = np.zeros(1, dtype=np.bool_)
        missing = np.ones(1, dtype=np.bool_)
        intersects_bounds_map1(0.0, 0.0, 1.0, 1.0, values, offsets, result, missing)
        assert not result[0]

    def test_intersects_bounds_multiple(self):
        """Three lines: first inside, second outside, third crosses."""
        from akimbo_geo.algorithms import intersects_bounds_map1
        values = np.array([
            0.5, 0.5, 0.5, 0.5,       # point inside box
            5.0, 5.0, 6.0, 6.0,       # entirely outside
            -1.0, 0.5, 2.0, 0.5,      # crosses box
        ])
        offsets = np.array([0, 4, 8, 12], dtype=np.int32)
        result  = np.zeros(3, dtype=np.bool_)
        missing = np.zeros(3, dtype=np.bool_)
        intersects_bounds_map1(0.0, 0.0, 1.0, 1.0, values, offsets, result, missing)
        assert result[0]
        assert not result[1]
        assert result[2]

    def test_segments_intersect_crossing(self):
        """Two segments that clearly cross → True."""
        from akimbo_geo.algorithms import _segments_intersect
        # (0,0)→(1,1) crosses (0,1)→(1,0)
        assert _segments_intersect(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0)

    def test_segments_intersect_parallel(self):
        """Two parallel segments → False."""
        from akimbo_geo.algorithms import _segments_intersect
        assert not _segments_intersect(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    def test_segments_intersect_collinear_overlap(self):
        """Collinear overlapping segments → True (endpoint on segment)."""
        from akimbo_geo.algorithms import _segments_intersect
        # (0,0)→(2,0) and (1,0)→(3,0) share (1,0)→(2,0)
        assert _segments_intersect(0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0)

    def test_point_in_ring_inside(self):
        """Point clearly inside a square ring → True."""
        from akimbo_geo.algorithms import _point_in_ring
        # CCW unit square
        values = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        assert _point_in_ring(0.5, 0.5, values, 0, 10)

    def test_point_in_ring_outside(self):
        """Point outside the ring → False."""
        from akimbo_geo.algorithms import _point_in_ring
        values = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        assert not _point_in_ring(2.0, 2.0, values, 0, 10)


# ===========================================================================
# total_bounds
# ===========================================================================

class TestTotalBounds:

    def test_total_bounds_single_point(self):
        from akimbo_geo.algorithms import total_bounds
        values = np.array([3.0, 7.0])
        xmin, ymin, xmax, ymax = total_bounds(values)
        assert xmin == xmax == 3.0
        assert ymin == ymax == 7.0

    def test_total_bounds_multiple(self):
        from akimbo_geo.algorithms import total_bounds
        values = np.array([0.0, 5.0, 3.0, 2.0, -1.0, 4.0])
        xmin, ymin, xmax, ymax = total_bounds(values)
        assert xmin == -1.0
        assert xmax == 3.0
        assert ymin == 2.0
        assert ymax == 5.0
