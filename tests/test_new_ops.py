"""test_new_ops.py — tests for all newly implemented operations.

Covers:
- count_coordinates, count_geometries, count_interior_rings
- is_closed, is_ccw, has_z, has_m
- x, y, z, get_coordinates (Point depth-0)
- translate, scale, affine_transform
- reverse
- force_2d, force_3d
- segmentize
- orient_polygons
"""

from math import isclose, sqrt

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401
import akimbo_geo     # noqa: F401

# ---------------------------------------------------------------------------
# Shared geometry fixtures
# ---------------------------------------------------------------------------

# Flat-interleaved (spatialpandas style) unit-square ring: (0,0),(1,0),(1,1),(0,1),(0,0)
SQUARE_RING = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]

# Two lines (depth-1)
LINE1 = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]   # (0,0)→(1,0)→(1,1)
LINE2 = [0.0, 0.0, 3.0, 4.0]              # (0,0)→(3,4)

def _line_arr():
    return ak.from_arrow(pa.array([LINE1, LINE2], type=pa.list_(pa.float64())))

def _square_poly_arr():
    """One unit square polygon as list<list<float>>."""
    return ak.from_arrow(
        pa.array([[SQUARE_RING]], type=pa.list_(pa.list_(pa.float64())))
    )

def _poly_with_hole_arr():
    """2×2 square with 0.5×0.5 interior hole as list<list<float>>."""
    exterior = [0.0,0.0, 2.0,0.0, 2.0,2.0, 0.0,2.0, 0.0,0.0]
    hole     = [0.5,0.5, 1.0,0.5, 1.0,1.0, 0.5,1.0, 0.5,0.5]
    return ak.from_arrow(
        pa.array([[exterior, hole]], type=pa.list_(pa.list_(pa.float64())))
    )


# ===========================================================================
# Counting
# ===========================================================================

class TestCountCoordinates:

    def test_line_coords(self):
        from akimbo_geo.accessor import _op_count_coords
        result = ak.Array(_op_count_coords(_line_arr().layout)).tolist()
        assert result[0] == 3   # LINE1 has 3 points
        assert result[1] == 2   # LINE2 has 2 points

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.count_coordinates(_line_arr()))
        assert result[0] == 3
        assert result[1] == 2


class TestCountGeometries:

    def test_polygon_rings(self):
        from akimbo_geo.accessor import _op_count_geoms
        result = ak.Array(_op_count_geoms(_square_poly_arr().layout)).tolist()
        assert result[0] == 1   # 1 ring (no holes)

    def test_polygon_with_hole(self):
        from akimbo_geo.accessor import _op_count_geoms
        result = ak.Array(_op_count_geoms(_poly_with_hole_arr().layout)).tolist()
        assert result[0] == 2   # exterior + 1 hole

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.count_geometries(_poly_with_hole_arr()))
        assert result[0] == 2


class TestCountInteriorRings:

    def test_no_holes(self):
        from akimbo_geo.accessor import _op_count_interior_rings
        result = ak.Array(
            _op_count_interior_rings(_square_poly_arr().layout)
        ).tolist()
        assert result[0] == 0

    def test_one_hole(self):
        from akimbo_geo.accessor import _op_count_interior_rings
        result = ak.Array(
            _op_count_interior_rings(_poly_with_hole_arr().layout)
        ).tolist()
        assert result[0] == 1

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(
            GeoAccessor.count_interior_rings(_poly_with_hole_arr())
        )
        assert result[0] == 1


# ===========================================================================
# Predicates
# ===========================================================================

class TestIsClosed:

    def test_open_line(self):
        from akimbo_geo.accessor import _op_is_closed
        result = ak.Array(_op_is_closed(_line_arr().layout)).tolist()
        assert result[0] is False  # (0,0)→(1,0)→(1,1) not closed
        assert result[1] is False

    def test_closed_ring(self):
        from akimbo_geo.accessor import _op_is_closed
        ring = ak.from_arrow(
            pa.array([SQUARE_RING], type=pa.list_(pa.float64()))
        )
        result = ak.Array(_op_is_closed(ring.layout)).tolist()
        assert result[0] is True

    def test_single_point(self):
        """Fewer than 2 points → not closed."""
        from akimbo_geo.accessor import _op_is_closed
        arr = ak.from_arrow(pa.array([[1.0, 2.0]], type=pa.list_(pa.float64())))
        result = ak.Array(_op_is_closed(arr.layout)).tolist()
        assert result[0] is False

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        ring = ak.from_arrow(pa.array([SQUARE_RING], type=pa.list_(pa.float64())))
        result = ak.to_list(GeoAccessor.is_closed(ring))
        assert result[0] is True


class TestIsCCW:

    def test_ccw_square(self):
        """CCW (positive area) exterior ring → True."""
        from akimbo_geo.accessor import _op_is_ccw
        result = ak.Array(_op_is_ccw(_square_poly_arr().layout)).tolist()
        assert result[0] is True

    def test_cw_square(self):
        """CW (negative area) ring → False."""
        from akimbo_geo.accessor import _op_is_ccw
        cw_ring = [0.0,0.0, 0.0,1.0, 1.0,1.0, 1.0,0.0, 0.0,0.0]
        arr = ak.from_arrow(pa.array([[cw_ring]], type=pa.list_(pa.list_(pa.float64()))))
        result = ak.Array(_op_is_ccw(arr.layout)).tolist()
        assert result[0] is False

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.is_ccw(_square_poly_arr()))
        assert result[0] is True


class TestHasZM:

    def test_has_z_false_2d(self):
        from akimbo_geo.accessor import _op_has_z
        result = ak.Array(_op_has_z(_line_arr().layout)).tolist()
        assert all(v is False for v in result)

    def test_has_z_true_3d(self):
        from akimbo_geo.accessor import _op_has_z
        arr = ak.from_arrow(pa.array(
            [[[0.0, 0.0, 10.0], [1.0, 0.0, 20.0]]],
            type=pa.list_(pa.list_(pa.field("xyz", pa.float64()), 3))
        ))
        result = ak.Array(_op_has_z(arr.layout)).tolist()
        assert all(v is True for v in result)

    def test_has_m_false(self):
        from akimbo_geo.accessor import _op_has_m
        result = ak.Array(_op_has_m(_line_arr().layout)).tolist()
        assert all(v is False for v in result)

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.has_z(_line_arr()))
        assert all(v is False for v in result)


# ===========================================================================
# Coordinate extraction (Point depth-0)
# ===========================================================================

class TestXYZ:

    def _point_fsl(self):
        """Three Points as FixedSizeList<double>[2]."""
        return ak.from_arrow(pa.array(
            [[0.0, 0.0], [3.0, 4.0], [1.0, 2.0]],
            type=pa.list_(pa.field("xy", pa.float64()), 2)
        ))

    def _point_3d(self):
        return ak.from_arrow(pa.array(
            [[0.0, 0.0, 10.0], [3.0, 4.0, 20.0]],
            type=pa.list_(pa.field("xyz", pa.float64()), 3)
        ))

    def test_x(self):
        from akimbo_geo.accessor import _op_get_x
        result = ak.Array(_op_get_x(self._point_fsl().layout)).tolist()
        assert isclose(result[0], 0.0)
        assert isclose(result[1], 3.0)
        assert isclose(result[2], 1.0)

    def test_y(self):
        from akimbo_geo.accessor import _op_get_y
        result = ak.Array(_op_get_y(self._point_fsl().layout)).tolist()
        assert isclose(result[0], 0.0)
        assert isclose(result[1], 4.0)

    def test_z_2d_is_nan(self):
        from akimbo_geo.accessor import _op_get_z
        result = ak.Array(_op_get_z(self._point_fsl().layout)).tolist()
        assert all(v is None or (isinstance(v, float) and np.isnan(v))
                   for v in result)

    def test_z_3d(self):
        from akimbo_geo.accessor import _op_get_z
        result = ak.Array(_op_get_z(self._point_3d().layout)).tolist()
        assert isclose(result[0], 10.0)
        assert isclose(result[1], 20.0)

    def test_via_accessor_x(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.x(self._point_fsl()))
        assert isclose(result[1], 3.0)

    def test_via_accessor_y(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.y(self._point_fsl()))
        assert isclose(result[1], 4.0)

    def test_via_accessor_z(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.z(self._point_3d()))
        assert isclose(result[0], 10.0)


class TestGetCoordinates:

    def test_line_coords_structure(self):
        from akimbo_geo.accessor import _op_get_coordinates
        result = ak.Array(_op_get_coordinates(_line_arr().layout)).tolist()
        # Each row is a list of {x, y} dicts
        assert len(result[0]) == 3  # LINE1: 3 points
        assert isclose(result[0][0]["x"], 0.0)
        assert isclose(result[0][0]["y"], 0.0)
        assert isclose(result[0][2]["x"], 1.0)
        assert isclose(result[0][2]["y"], 1.0)

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.get_coordinates(_line_arr()))
        assert len(result[1]) == 2  # LINE2: 2 points
        assert isclose(result[1][1]["x"], 3.0)


# ===========================================================================
# Affine transformations
# ===========================================================================

class TestTranslate:

    def test_shift_by_1_1(self):
        from akimbo_geo.accessor import _op_translate
        result_layout = _op_translate(_line_arr().layout, 1.0, 1.0)
        result = ak.Array(result_layout).tolist()
        # LINE1[0] was (0,0) → should be (1,1)
        assert isclose(result[0][0], 1.0)
        assert isclose(result[0][1], 1.0)
        # LINE1[2] was (1,0) → should be (2,1)
        assert isclose(result[0][2], 2.0)
        assert isclose(result[0][3], 1.0)

    def test_zero_shift_identity(self):
        from akimbo_geo.accessor import _op_translate
        original = ak.to_list(_line_arr())
        result = ak.Array(_op_translate(_line_arr().layout, 0.0, 0.0)).tolist()
        assert result == original

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.translate(_line_arr(), xoff=2.0, yoff=3.0))
        # LINE2[0] was (0,0) → should be (2,3)
        assert isclose(result[1][0], 2.0)
        assert isclose(result[1][1], 3.0)


class TestScale:

    def test_scale_2x(self):
        from akimbo_geo.accessor import _op_scale
        result = ak.Array(
            _op_scale(_line_arr().layout, 2.0, 2.0, (0.0, 0.0))
        ).tolist()
        # LINE2 (0,0)→(3,4) scaled 2× → (0,0)→(6,8)
        assert isclose(result[1][0], 0.0)
        assert isclose(result[1][2], 6.0)
        assert isclose(result[1][3], 8.0)

    def test_scale_with_origin(self):
        from akimbo_geo.accessor import _op_scale
        # Scale by 2 around (1, 0): (0,0)→ (-1,0); (1,0)→(1,0); (1,1)→(1,2)
        result = ak.Array(
            _op_scale(_line_arr().layout, 2.0, 2.0, (1.0, 0.0))
        ).tolist()
        assert isclose(result[0][0], -1.0)  # x: (0-1)*2+1 = -1
        assert isclose(result[0][1], 0.0)   # y: (0-0)*2+0 = 0
        assert isclose(result[0][2], 1.0)   # x: (1-1)*2+1 = 1
        assert isclose(result[0][4], 1.0)   # x: (1-1)*2+1 = 1
        assert isclose(result[0][5], 2.0)   # y: (1-0)*2+0 = 2

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.scale(_line_arr(), xfact=0.5, yfact=0.5))
        assert isclose(result[1][2], 1.5)   # 3.0 * 0.5


class TestAffineTransform:

    def test_identity(self):
        """Identity matrix [1,0,0,1,0,0] leaves coordinates unchanged."""
        from akimbo_geo.accessor import _op_affine_transform
        original = ak.to_list(_line_arr())
        result = ak.Array(
            _op_affine_transform(_line_arr().layout, [1, 0, 0, 1, 0, 0])
        ).tolist()
        np.testing.assert_allclose(result[0], original[0])
        np.testing.assert_allclose(result[1], original[1])

    def test_translate_via_matrix(self):
        """[1,0,0,1,2,3] is pure translation by (2,3)."""
        from akimbo_geo.accessor import _op_affine_transform
        result = ak.Array(
            _op_affine_transform(_line_arr().layout, [1, 0, 0, 1, 2, 3])
        ).tolist()
        assert isclose(result[1][0], 2.0)  # LINE2 (0,0)→(2,3)
        assert isclose(result[1][1], 3.0)

    def test_reflection_y(self):
        """[1,0,0,-1,0,0] mirrors in the x-axis (y → -y)."""
        from akimbo_geo.accessor import _op_affine_transform
        result = ak.Array(
            _op_affine_transform(_line_arr().layout, [1, 0, 0, -1, 0, 0])
        ).tolist()
        # LINE1[2] = (1,0) → (1, 0); LINE1[4] = (1,1) → (1, -1)
        assert isclose(result[0][5], -1.0)

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(
            GeoAccessor.affine_transform(_line_arr(), matrix=[1, 0, 0, 1, 5, 5])
        )
        assert isclose(result[0][0], 5.0)


# ===========================================================================
# Reverse
# ===========================================================================

class TestReverse:

    def test_two_point_line(self):
        """Reversing (0,0)→(3,4) gives (3,4)→(0,0)."""
        from akimbo_geo.accessor import _op_reverse
        arr = ak.from_arrow(pa.array([LINE2], type=pa.list_(pa.float64())))
        result = ak.Array(_op_reverse(arr.layout)).tolist()
        assert isclose(result[0][0], 3.0)
        assert isclose(result[0][1], 4.0)
        assert isclose(result[0][2], 0.0)
        assert isclose(result[0][3], 0.0)

    def test_three_point_line(self):
        """(0,0)→(1,0)→(1,1) reversed → (1,1)→(1,0)→(0,0)."""
        from akimbo_geo.accessor import _op_reverse
        arr = ak.from_arrow(pa.array([LINE1], type=pa.list_(pa.float64())))
        result = ak.Array(_op_reverse(arr.layout)).tolist()
        assert isclose(result[0][0], 1.0) and isclose(result[0][1], 1.0)
        assert isclose(result[0][2], 1.0) and isclose(result[0][3], 0.0)
        assert isclose(result[0][4], 0.0) and isclose(result[0][5], 0.0)

    def test_double_reverse_is_identity(self):
        from akimbo_geo.accessor import _op_reverse
        original = ak.to_list(_line_arr())
        once  = _op_reverse(_line_arr().layout)
        twice = _op_reverse(once)
        result = ak.Array(twice).tolist()
        np.testing.assert_allclose(result[0], original[0])
        np.testing.assert_allclose(result[1], original[1])

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        arr = ak.from_arrow(pa.array([LINE2], type=pa.list_(pa.float64())))
        result = ak.to_list(GeoAccessor.reverse(arr))
        assert isclose(result[0][0], 3.0) and isclose(result[0][1], 4.0)


# ===========================================================================
# force_2d / force_3d
# ===========================================================================

class TestForce2D3D:

    def _line_3d(self):
        """Line with XYZ coordinates."""
        return ak.from_arrow(pa.array(
            [[[0.0, 0.0, 10.0], [1.0, 0.0, 20.0], [1.0, 1.0, 30.0]]],
            type=pa.list_(pa.list_(pa.field("xyz", pa.float64()), 3))
        ))

    def test_force_2d_drops_z(self):
        from akimbo_geo.accessor import _op_force_2d
        arr_3d = self._line_3d()
        result_layout = _op_force_2d(arr_3d.layout)
        result = ak.Array(result_layout).tolist()
        # Should now be [[0,0, 1,0, 1,1]] (6 floats, no z)
        row = result[0]
        assert len(row) == 6
        assert isclose(row[0], 0.0) and isclose(row[1], 0.0)
        assert isclose(row[2], 1.0) and isclose(row[3], 0.0)

    def test_force_2d_already_2d_noop(self):
        """force_2d on a 2D array returns layout unchanged."""
        from akimbo_geo.accessor import _op_force_2d
        original = _line_arr().layout
        result = _op_force_2d(original)
        assert result is original

    def test_force_3d_adds_z(self):
        from akimbo_geo.accessor import _op_force_3d
        result_layout = _op_force_3d(_line_arr().layout, z_val=99.0)
        result = ak.Array(result_layout).tolist()
        row = result[0]
        # LINE1 had 3 points; output is list of [x,y,z] triples
        assert len(row) == 3          # 3 coordinate points
        assert len(row[0]) == 3       # each is [x, y, z]
        assert isclose(row[0][2], 99.0)   # z of first point
        assert isclose(row[1][2], 99.0)   # z of second point

    def test_force_3d_already_3d_noop(self):
        from akimbo_geo.accessor import _op_force_3d
        arr_3d = self._line_3d()
        original = arr_3d.layout
        result = _op_force_3d(original, z_val=0.0)
        assert result is original

    def test_force_2d_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.force_2d(self._line_3d()))
        assert len(result[0]) == 6

    def test_force_3d_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.force_3d(_line_arr(), z=5.0))
        # Each row is now a list of [x, y, z] triples
        assert len(result[0]) == 3      # 3 coordinate points in LINE1
        assert len(result[0][0]) == 3   # each is [x, y, z]
        assert isclose(result[0][0][2], 5.0)

    def test_round_trip_force_2d_after_3d(self):
        """force_2d(force_3d(arr)) should recover the original 2D x,y coords."""
        from akimbo_geo.accessor import _op_force_2d, _op_force_3d
        original = ak.to_list(_line_arr())
        promoted = _op_force_3d(_line_arr().layout, z_val=0.0)
        restored = _op_force_2d(ak.Array(promoted).layout)
        result = ak.Array(restored).tolist()
        # Result is INTERLEAVED_FLAT: [[x0,y0,x1,y1,...], ...]
        assert isclose(result[0][0], original[0][0])  # x0
        assert isclose(result[0][1], original[0][1])  # y0
        assert len(result[0]) == len(original[0])      # same number of floats


# ===========================================================================
# Segmentize
# ===========================================================================

class TestSegmentize:

    def test_no_segmentization_needed(self):
        """Edges shorter than max_len → output same as input."""
        from akimbo_geo.accessor import _op_segmentize
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 0.5, 0.0]], type=pa.list_(pa.float64())))
        result = ak.Array(_op_segmentize(arr.layout, max_segment_length=1.0)).tolist()
        # edge length = 0.5 < 1.0, so no new points
        assert len(result[0]) == 4   # still 2 points = 4 floats

    def test_one_intermediate_point(self):
        """Edge length 2.0 with max_len 1.0 → 1 intermediate point added."""
        from akimbo_geo.accessor import _op_segmentize
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 2.0, 0.0]], type=pa.list_(pa.float64())))
        result = ak.Array(_op_segmentize(arr.layout, max_segment_length=1.0)).tolist()
        # 2 segments → 3 points = 6 floats
        assert len(result[0]) == 6
        assert isclose(result[0][2], 1.0) and isclose(result[0][3], 0.0)  # midpoint

    def test_preserves_total_length(self):
        """Segmentizing a line should not change its total length."""
        from akimbo_geo.accessor import _op_segmentize, _op_length
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64())))
        seg = _op_segmentize(arr.layout, max_segment_length=1.0)
        length_seg = float(ak.Array(_op_length(seg)).tolist()[0])
        assert isclose(length_seg, 5.0, rel_tol=1e-6)

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 4.0, 0.0]], type=pa.list_(pa.float64())))
        result = ak.to_list(GeoAccessor.segmentize(arr, max_segment_length=2.0))
        # 4.0/2.0 = 2 segments → 3 points = 6 floats
        assert len(result[0]) == 6


# ===========================================================================
# orient_polygons
# ===========================================================================

class TestOrientPolygons:

    def test_ccw_stays_ccw(self):
        """Already CCW polygon with exterior_cw=False → unchanged."""
        from akimbo_geo.accessor import _op_orient_polygons, _op_is_ccw
        poly = _square_poly_arr()
        assert ak.Array(_op_is_ccw(poly.layout)).tolist()[0] is True
        oriented = _op_orient_polygons(poly.layout, exterior_cw=False)
        assert ak.Array(_op_is_ccw(oriented)).tolist()[0] is True

    def test_cw_flipped_to_ccw(self):
        """CW polygon with exterior_cw=False → flipped to CCW."""
        from akimbo_geo.accessor import _op_orient_polygons, _op_is_ccw
        cw_ring = [0.0,0.0, 0.0,1.0, 1.0,1.0, 1.0,0.0, 0.0,0.0]
        poly = ak.from_arrow(
            pa.array([[cw_ring]], type=pa.list_(pa.list_(pa.float64())))
        )
        assert ak.Array(_op_is_ccw(poly.layout)).tolist()[0] is False
        oriented = _op_orient_polygons(poly.layout, exterior_cw=False)
        assert ak.Array(_op_is_ccw(oriented)).tolist()[0] is True

    def test_ccw_flipped_to_cw(self):
        """CCW polygon with exterior_cw=True → flipped to CW."""
        from akimbo_geo.accessor import _op_orient_polygons, _op_is_ccw
        poly = _square_poly_arr()
        oriented = _op_orient_polygons(poly.layout, exterior_cw=True)
        assert ak.Array(_op_is_ccw(oriented)).tolist()[0] is False

    def test_area_magnitude_preserved(self):
        """Flipping orientation should not change |area|."""
        from akimbo_geo.accessor import _op_orient_polygons, _op_area
        poly = _square_poly_arr()
        flipped = _op_orient_polygons(poly.layout, exterior_cw=True)
        area = abs(ak.Array(_op_area(flipped)).tolist()[0])
        assert isclose(area, 1.0)

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        cw_ring = [0.0,0.0, 0.0,1.0, 1.0,1.0, 1.0,0.0, 0.0,0.0]
        poly = ak.from_arrow(
            pa.array([[cw_ring]], type=pa.list_(pa.list_(pa.float64())))
        )
        result_layout = GeoAccessor.orient_polygons(poly)
        from akimbo_geo.accessor import _op_is_ccw
        assert ak.Array(_op_is_ccw(ak.Array(result_layout).layout)).tolist()[0] is True

    def test_hole_orientation_enforced(self):
        """After orient_polygons: exterior CCW, hole CW."""
        from akimbo_geo.accessor import _op_orient_polygons
        from akimbo_geo.algorithms import orient_ring
        # Build polygon where hole is also CCW initially
        exterior = [0.0,0.0, 2.0,0.0, 2.0,2.0, 0.0,2.0, 0.0,0.0]   # CCW
        hole_ccw  = [0.5,0.5, 1.0,0.5, 1.0,1.0, 0.5,1.0, 0.5,0.5]  # CCW (wrong for hole)
        poly = ak.from_arrow(
            pa.array([[exterior, hole_ccw]], type=pa.list_(pa.list_(pa.float64())))
        )
        oriented = _op_orient_polygons(poly.layout, exterior_cw=False)
        vals = ak.to_list(ak.Array(oriented))
        # Extract hole ring and check it is CW (area < 0)
        hole_flat = np.array(vals[0][1])
        area = orient_ring(hole_flat, 0, len(hole_flat))
        assert area < 0.0  # hole should be CW
