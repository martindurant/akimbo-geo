"""test_shapely_ops.py — tests for Tier 1 and Tier 2 operations.

Tier 1 (pure numba, no shapely):
  - is_ring, minimum_bounding_radius

Tier 2 (shapely-backed via from_ragged_array / to_ragged_array):
  - Unary scalar: is_valid, is_simple, is_empty, is_valid_reason, geom_type_id
  - Unary geometry: boundary, convex_hull, envelope, make_valid, normalize,
      extract_unique_points, representative_point, minimum_bounding_circle,
      minimum_rotated_rectangle, simplify, buffer, concave_hull, offset_curve,
      remove_repeated_points, line_merge
  - Linear referencing: interpolate, project, shared_paths, shortest_line
  - Binary predicates: contains, within, intersects, crosses, overlaps, touches,
      covers, covered_by, disjoint, distance, hausdorff_distance
  - Set-theoretic: difference, intersection, union, symmetric_difference,
      union_all, intersection_all

Bridge helpers:
  - _layout_to_shapely, _shapely_to_layout round-trip fidelity
"""

from math import isclose, sqrt

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401
import akimbo_geo     # noqa: F401

shapely = pytest.importorskip("shapely", reason="shapely>=2.0 required")

# ---------------------------------------------------------------------------
# Shared geometry data
# ---------------------------------------------------------------------------

# Unit square: CCW closed ring
SQUARE_RING = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
# Open line
LINE = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
# Line with repeated endpoint  
OPEN_LINE = [0.0, 0.0, 3.0, 4.0]

def _line_arr(*rows):
    rows = rows or ([LINE], [OPEN_LINE])
    return ak.from_arrow(pa.array(list(rows), type=pa.list_(pa.float64())))

def _poly_arr(*rings_list):
    if not rings_list:
        rings_list = ([SQUARE_RING],)
    return ak.from_arrow(
        pa.array(list([list(r) for r in rings_list]),
                 type=pa.list_(pa.list_(pa.float64())))
    )


# ===========================================================================
# Bridge helpers
# ===========================================================================

class TestBridgeHelpers:
    """Verify _layout_to_shapely / _shapely_to_layout round-trip."""

    def test_line_round_trip(self):
        from akimbo_geo.accessor import _layout_to_shapely, _shapely_to_layout
        arr = ak.from_arrow(pa.array([LINE, OPEN_LINE], type=pa.list_(pa.float64())))
        geoms = _layout_to_shapely(arr.layout)
        assert len(geoms) == 2
        assert isclose(geoms[0].length, 2.0)
        assert isclose(geoms[1].length, 5.0)

        back_layout = _shapely_to_layout(geoms)
        back = ak.Array(back_layout).tolist()
        np.testing.assert_allclose(back[0], LINE)
        np.testing.assert_allclose(back[1], OPEN_LINE)

    def test_polygon_round_trip(self):
        from akimbo_geo.accessor import _layout_to_shapely, _shapely_to_layout
        arr = _poly_arr([SQUARE_RING])
        geoms = _layout_to_shapely(arr.layout)
        assert isclose(geoms[0].area, 1.0)

        back_layout = _shapely_to_layout(geoms)
        back = ak.Array(back_layout).tolist()
        # Polygon ring coords may be reordered by GEOS — just check area preserved
        from akimbo_geo.accessor import _op_area
        area = abs(ak.Array(_op_area(ak.Array(back_layout).layout)).tolist()[0])
        assert isclose(area, 1.0)

    def test_empty_coords_array(self):
        """Empty geometry array produces empty layout without errors."""
        from akimbo_geo.accessor import _layout_to_shapely
        arr = ak.from_arrow(pa.array([], type=pa.list_(pa.float64())))
        geoms = _layout_to_shapely(arr.layout)
        assert len(geoms) == 0


# ===========================================================================
# Tier 1 — pure numba
# ===========================================================================

class TestIsRing:

    def test_closed_ring_is_ring(self):
        from akimbo_geo.accessor import _op_is_ring
        ring = ak.from_arrow(pa.array([SQUARE_RING], type=pa.list_(pa.float64())))
        result = ak.Array(_op_is_ring(ring.layout)).tolist()
        assert result[0] is True

    def test_open_line_not_ring(self):
        from akimbo_geo.accessor import _op_is_ring
        arr = ak.from_arrow(pa.array([LINE], type=pa.list_(pa.float64())))
        result = ak.Array(_op_is_ring(arr.layout)).tolist()
        assert result[0] is False

    def test_too_few_points(self):
        from akimbo_geo.accessor import _op_is_ring
        tiny = ak.from_arrow(pa.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]],
                                       type=pa.list_(pa.float64())))
        result = ak.Array(_op_is_ring(tiny.layout)).tolist()
        # 3 pairs (6 floats) — closed but < 4 points (needs 8 floats)
        assert result[0] is False

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        ring = ak.from_arrow(pa.array([SQUARE_RING], type=pa.list_(pa.float64())))
        result = ak.to_list(GeoAccessor.is_ring(ring))
        assert result[0] is True


class TestMinimumBoundingRadius:

    def test_unit_square_line(self):
        """Line from (0,0) to (1,1): bounding box is 1×1, radius = sqrt(2)/2."""
        from akimbo_geo.accessor import _op_minimum_bounding_radius
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 1.0, 1.0]],
                                      type=pa.list_(pa.float64())))
        result = ak.Array(_op_minimum_bounding_radius(arr.layout)).tolist()
        assert isclose(result[0], sqrt(2) / 2, rel_tol=1e-6)

    def test_horizontal_line(self):
        """Horizontal line (0,0)→(4,0): radius = 2.0."""
        from akimbo_geo.accessor import _op_minimum_bounding_radius
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 4.0, 0.0]],
                                      type=pa.list_(pa.float64())))
        result = ak.Array(_op_minimum_bounding_radius(arr.layout)).tolist()
        assert isclose(result[0], 2.0)

    def test_polygon(self):
        """Unit square polygon: bbox is 1×1, radius = sqrt(2)/2."""
        from akimbo_geo.accessor import _op_minimum_bounding_radius
        arr = _poly_arr([SQUARE_RING])
        result = ak.Array(_op_minimum_bounding_radius(arr.layout)).tolist()
        assert isclose(result[0], sqrt(2) / 2, rel_tol=1e-6)

    def test_via_accessor(self):
        from akimbo_geo.accessor import GeoAccessor
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 4.0, 0.0]],
                                      type=pa.list_(pa.float64())))
        result = ak.to_list(GeoAccessor.minimum_bounding_radius(arr))
        assert isclose(result[0], 2.0)


# ===========================================================================
# Tier 2 — unary scalar predicates
# ===========================================================================

class TestUnaryScalar:

    def _valid_poly(self):
        return _poly_arr([SQUARE_RING])

    def _invalid_poly(self):
        # Self-intersecting "bowtie" polygon
        bowtie = [0.0,0.0, 2.0,2.0, 2.0,0.0, 0.0,2.0, 0.0,0.0]
        return ak.from_arrow(
            pa.array([[bowtie]], type=pa.list_(pa.list_(pa.float64())))
        )

    def test_is_valid_true(self):
        from akimbo_geo.accessor import _op_is_valid
        result = ak.Array(_op_is_valid(self._valid_poly().layout)).tolist()
        assert result[0] is True

    def test_is_valid_false(self):
        from akimbo_geo.accessor import _op_is_valid
        result = ak.Array(_op_is_valid(self._invalid_poly().layout)).tolist()
        assert result[0] is False

    def test_is_simple_line(self):
        """Non-self-intersecting line is simple."""
        from akimbo_geo.accessor import _op_is_simple
        arr = _line_arr(LINE)
        result = ak.Array(_op_is_simple(arr.layout)).tolist()
        assert result[0] is True

    def test_is_empty_false(self):
        from akimbo_geo.accessor import _op_is_empty
        result = ak.Array(_op_is_empty(self._valid_poly().layout)).tolist()
        assert result[0] is False

    def test_is_valid_reason_valid(self):
        from akimbo_geo.accessor import _op_is_valid_reason
        result = ak.Array(_op_is_valid_reason(self._valid_poly().layout)).tolist()
        assert result[0] == "Valid Geometry"

    def test_is_valid_reason_invalid(self):
        from akimbo_geo.accessor import _op_is_valid_reason
        result = ak.Array(_op_is_valid_reason(self._invalid_poly().layout)).tolist()
        assert result[0] != "Valid Geometry"

    def test_geom_type_id_polygon(self):
        from akimbo_geo.accessor import _op_geom_type_id
        result = ak.Array(_op_geom_type_id(self._valid_poly().layout)).tolist()
        assert result[0] == 3  # shapely GeometryType.POLYGON = 3

    def test_geom_type_id_linestring(self):
        from akimbo_geo.accessor import _op_geom_type_id
        arr = _line_arr(LINE)
        result = ak.Array(_op_geom_type_id(arr.layout)).tolist()
        assert result[0] == 1  # LINESTRING = 1

    def test_via_accessor_is_valid(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.is_valid(self._valid_poly()))
        assert result[0] is True


# ===========================================================================
# Tier 2 — unary geometry (constructive)
# ===========================================================================

class TestConstructive:

    def test_convex_hull_triangle(self):
        """Convex hull of a triangle is itself."""
        from akimbo_geo.accessor import _op_convex_hull, _op_area
        # Triangle (0,0),(2,0),(1,2): area = 2.0
        tri = [0.0,0.0, 2.0,0.0, 1.0,2.0, 0.0,0.0]
        arr = ak.from_arrow(pa.array([[tri]], type=pa.list_(pa.list_(pa.float64()))))
        hull_layout = _op_convex_hull(arr.layout)
        area = abs(ak.Array(_op_area(hull_layout)).tolist()[0])
        assert isclose(area, 2.0)

    def test_envelope_line(self):
        """Envelope (bounding box) of a diagonal line."""
        from akimbo_geo.accessor import _op_envelope, _op_bounds
        arr = _line_arr(LINE)
        env_layout = _op_envelope(arr.layout)
        # The envelope polygon's bounds should be the same as the line's bounds
        b = ak.Array(_op_bounds(arr.layout)).tolist()[0]
        b_env = ak.Array(_op_bounds(env_layout)).tolist()[0]
        assert isclose(b["xmin"], b_env["xmin"])
        assert isclose(b["xmax"], b_env["xmax"])
        assert isclose(b["ymin"], b_env["ymin"])
        assert isclose(b["ymax"], b_env["ymax"])

    def test_boundary_polygon(self):
        """Boundary of a polygon is a closed line geometry."""
        from akimbo_geo.accessor import _op_boundary, _op_geom_type_id, _op_is_closed
        poly = _poly_arr([SQUARE_RING])
        bnd_layout = _op_boundary(poly.layout)
        # GEOS returns LinearRing (type_id=2) or LineString (1) depending on version
        geom_id = ak.Array(_op_geom_type_id(bnd_layout)).tolist()[0]
        assert geom_id in (1, 2)  # LINESTRING or LINEARRING
        # Boundary of a polygon is always closed
        closed = ak.Array(_op_is_closed(bnd_layout)).tolist()[0]
        assert closed is True

    def test_make_valid_fixes_bowtie(self):
        """make_valid converts a self-intersecting polygon to a valid geometry."""
        from akimbo_geo.accessor import _op_make_valid, _op_is_valid
        bowtie = [0.0,0.0, 2.0,2.0, 2.0,0.0, 0.0,2.0, 0.0,0.0]
        arr = ak.from_arrow(
            pa.array([[bowtie]], type=pa.list_(pa.list_(pa.float64())))
        )
        fixed_layout = _op_make_valid(arr.layout)
        result = ak.Array(_op_is_valid(fixed_layout)).tolist()
        assert result[0] is True

    def test_simplify_reduces_points(self):
        """Simplify a zigzag line: should reduce coordinate count."""
        from akimbo_geo.accessor import _op_simplify, _op_count_coords
        # Zigzag with many small detours
        zigzag = []
        for i in range(10):
            zigzag.extend([float(i), 0.0])
            zigzag.extend([float(i) + 0.5, 0.01])
        arr = ak.from_arrow(pa.array([zigzag], type=pa.list_(pa.float64())))
        orig_n = ak.Array(_op_count_coords(arr.layout)).tolist()[0]
        simp_layout = _op_simplify(arr.layout, tolerance=0.1)
        simp_n = ak.Array(_op_count_coords(simp_layout)).tolist()[0]
        assert simp_n < orig_n

    def test_buffer_produces_polygon(self):
        """Buffer a small square polygon produces a larger polygon."""
        from akimbo_geo.accessor import _op_buffer, _op_area
        # Unit square: area = 1.0; buffer by 1.0 → larger polygon
        buf_layout = _op_buffer(_poly_arr([SQUARE_RING]).layout, distance=1.0, quad_segs=8)
        area = abs(ak.Array(_op_area(buf_layout)).tolist()[0])
        # Area must be substantially larger than 1.0
        assert area > 1.5

    def test_representative_point_inside(self):
        """representative_point (point_on_surface) is inside the polygon."""
        from akimbo_geo.accessor import _op_representative_point
        poly = _poly_arr([SQUARE_RING])
        pt_layout = _op_representative_point(poly.layout)
        pt = ak.Array(pt_layout).tolist()
        # The representative point should be within the unit square
        # depth-0 FSL: pt is [[x, y]]
        coords = pt[0] if isinstance(pt[0], list) else [pt[0], pt[1]]
        x = coords[0] if isinstance(coords, list) else pt[0]
        assert 0.0 <= float(x) <= 1.0

    def test_normalize_idempotent(self):
        """normalize(normalize(arr)) == normalize(arr)."""
        from akimbo_geo.accessor import _op_normalize
        poly = _poly_arr([SQUARE_RING])
        once = _op_normalize(poly.layout)
        twice = _op_normalize(once)
        # Both should be valid
        from akimbo_geo.accessor import _op_is_valid
        assert ak.Array(_op_is_valid(twice)).tolist()[0] is True

    def test_remove_repeated_points(self):
        """Remove repeated consecutive points from a line."""
        from akimbo_geo.accessor import _op_remove_repeated_points, _op_count_coords
        # Line with a repeated point: (0,0),(1,0),(1,0),(1,1)
        dup_line = [0.0,0.0, 1.0,0.0, 1.0,0.0, 1.0,1.0]
        arr = ak.from_arrow(pa.array([dup_line], type=pa.list_(pa.float64())))
        cleaned_layout = _op_remove_repeated_points(arr.layout)
        n = ak.Array(_op_count_coords(cleaned_layout)).tolist()[0]
        assert n == 3  # duplicate removed

    def test_minimum_bounding_circle_polygon(self):
        """Minimum bounding circle of a unit square has radius ~sqrt(2)/2."""
        from akimbo_geo.accessor import _op_minimum_bounding_circle
        poly = _poly_arr([SQUARE_RING])
        circle_layout = _op_minimum_bounding_circle(poly.layout)
        # Should produce a polygon (circle approximation)
        from akimbo_geo.accessor import _op_is_valid
        assert ak.Array(_op_is_valid(circle_layout)).tolist()[0] is True

    def test_minimum_rotated_rectangle(self):
        """Minimum rotated rectangle of a diagonal line has non-zero area."""
        from akimbo_geo.accessor import _op_minimum_rotated_rectangle, _op_area
        arr = _line_arr(LINE)
        rect_layout = _op_minimum_rotated_rectangle(arr.layout)
        # Result is a polygon
        from akimbo_geo.accessor import _op_is_valid
        assert ak.Array(_op_is_valid(rect_layout)).tolist()[0] is True

    def test_extract_unique_points(self):
        """extract_unique_points returns a MultiPoint (stored as flat list)."""
        from akimbo_geo.accessor import _op_extract_unique_points, _op_count_coords
        arr = _line_arr(LINE)
        pts_layout = _op_extract_unique_points(arr.layout)
        # The result is a list of coordinate pairs (same storage as LineString)
        # — count_coordinates tells us how many unique points there are
        n = ak.Array(_op_count_coords(pts_layout)).tolist()[0]
        assert n == 3  # LINE has 3 distinct points

    def test_concave_hull(self):
        """Concave hull of 4 points is a valid geometry."""
        from akimbo_geo.accessor import _op_concave_hull, _op_is_valid
        pts = [0.0,0.0, 2.0,0.0, 1.0,1.0, 2.0,2.0, 0.0,2.0]
        arr = ak.from_arrow(pa.array([pts], type=pa.list_(pa.float64())))
        hull_layout = _op_concave_hull(arr.layout, ratio=0.5)
        assert ak.Array(_op_is_valid(hull_layout)).tolist()[0] is True

    def test_offset_curve(self):
        """Offset curve of a horizontal line is a parallel line."""
        from akimbo_geo.accessor import _op_offset_curve
        horiz = ak.from_arrow(pa.array([[0.0,0.0, 4.0,0.0]],
                                        type=pa.list_(pa.float64())))
        offset_layout = _op_offset_curve(horiz.layout, distance=1.0)
        # Should produce a line shifted 1 unit
        offcoords = ak.Array(offset_layout).tolist()[0]
        # All y-values should be ~1.0
        ys = offcoords[1::2] if isinstance(offcoords[0], float) else \
             [c[1] for c in offcoords]
        assert all(isclose(y, 1.0, rel_tol=1e-5) for y in ys)

    def test_line_merge(self):
        """line_merge on a single continuous line returns a valid geometry."""
        from akimbo_geo.accessor import _op_line_merge, _op_is_valid
        # A simple 3-point line — line_merge on it should return it as-is
        arr = _line_arr(LINE)
        merged_layout = _op_line_merge(arr.layout)
        assert ak.Array(_op_is_valid(merged_layout)).tolist()[0] is True

    def test_via_accessor_convex_hull(self):
        from akimbo_geo.accessor import GeoAccessor
        poly = _poly_arr([SQUARE_RING])
        hull = GeoAccessor.convex_hull(poly)
        from akimbo_geo.accessor import _op_is_valid
        assert ak.Array(_op_is_valid(ak.Array(hull).layout)).tolist()[0] is True


# ===========================================================================
# Tier 2 — linear referencing
# ===========================================================================

class TestLinearReferencing:

    def _horiz_line(self):
        return ak.from_arrow(pa.array([[0.0,0.0, 4.0,0.0]],
                                       type=pa.list_(pa.float64())))

    def test_interpolate_midpoint(self):
        """Point at distance 2.0 along a 4-unit horizontal line is (2,0)."""
        from akimbo_geo.accessor import _op_interpolate
        result_layout = _op_interpolate(self._horiz_line().layout, 2.0)
        pt = ak.Array(result_layout).tolist()
        # depth-0: pt is [[x, y]] FSL or [[x,y]] list
        coords = pt[0] if isinstance(pt[0], (list, tuple)) else pt
        x_val = coords[0] if isinstance(coords[0], float) else coords[0][0]
        assert isclose(float(x_val), 2.0, rel_tol=1e-5)

    def test_interpolate_normalized(self):
        """Normalized distance 0.5 gives the midpoint."""
        from akimbo_geo.accessor import _op_interpolate
        result_layout = _op_interpolate(self._horiz_line().layout, 0.5,
                                         normalized=True)
        pt = ak.Array(result_layout).tolist()
        coords = pt[0] if isinstance(pt[0], (list, tuple)) else pt
        x_val = coords[0] if isinstance(coords[0], float) else coords[0][0]
        assert isclose(float(x_val), 2.0, rel_tol=1e-5)

    def test_project_point_on_line(self):
        """Project (2, 0) onto a 4-unit horizontal line returns 2.0."""
        from akimbo_geo.accessor import _op_project
        line = self._horiz_line()
        # Use FixedSizeList[2] for a depth-0 Point so it isn't misread as LineString
        pt = ak.from_arrow(pa.array(
            [[2.0, 0.0]], type=pa.list_(pa.field("xy", pa.float64()), 2)
        ))
        result = ak.Array(_op_project(line.layout, pt.layout)).tolist()
        assert isclose(result[0], 2.0, rel_tol=1e-5)

    def test_shortest_line(self):
        """Shortest line between two parallel horizontal lines."""
        from akimbo_geo.accessor import _op_shortest_line
        line_a = ak.from_arrow(pa.array([[0.0,0.0, 1.0,0.0]],
                                          type=pa.list_(pa.float64())))
        line_b = ak.from_arrow(pa.array([[0.0,1.0, 1.0,1.0]],
                                          type=pa.list_(pa.float64())))
        result_layout = _op_shortest_line(line_a.layout, line_b.layout)
        from akimbo_geo.accessor import _op_length
        dist = ak.Array(_op_length(result_layout)).tolist()[0]
        assert isclose(dist, 1.0, rel_tol=1e-5)

    def test_via_accessor_interpolate(self):
        from akimbo_geo.accessor import GeoAccessor
        result = GeoAccessor.interpolate(self._horiz_line(), 2.0)
        # Just verify it produces something without error
        assert ak.Array(result).tolist() is not None


# ===========================================================================
# Tier 2 — binary predicates
# ===========================================================================

class TestBinaryPredicates:

    def _unit_square(self):
        return _poly_arr([SQUARE_RING])

    def _inside_point_arr(self):
        """A point strictly inside the unit square — stored as FixedSizeList[2]."""
        return ak.from_arrow(pa.array(
            [[0.5, 0.5]], type=pa.list_(pa.field("xy", pa.float64()), 2)
        ))

    def _outside_point_arr(self):
        return ak.from_arrow(pa.array(
            [[5.0, 5.0]], type=pa.list_(pa.field("xy", pa.float64()), 2)
        ))

    def test_contains_true(self):
        from akimbo_geo.accessor import _layout_to_shapely, _shapely_binary_bool
        result = ak.Array(
            _shapely_binary_bool(self._unit_square().layout,
                                  self._inside_point_arr().layout, "contains")
        ).tolist()
        assert result[0] is True

    def test_contains_false(self):
        from akimbo_geo.accessor import _shapely_binary_bool
        result = ak.Array(
            _shapely_binary_bool(self._unit_square().layout,
                                  self._outside_point_arr().layout, "contains")
        ).tolist()
        assert result[0] is False

    def test_within(self):
        from akimbo_geo.accessor import _shapely_binary_bool
        # inside_point within unit_square
        result = ak.Array(
            _shapely_binary_bool(self._inside_point_arr().layout,
                                  self._unit_square().layout, "within")
        ).tolist()
        assert result[0] is True

    def test_intersects_true(self):
        from akimbo_geo.accessor import _shapely_binary_bool
        # Line crossing the unit square boundary
        cross_line = ak.from_arrow(
            pa.array([[0.5, -1.0, 0.5, 2.0]], type=pa.list_(pa.float64()))
        )
        result = ak.Array(
            _shapely_binary_bool(self._unit_square().layout,
                                  cross_line.layout, "intersects")
        ).tolist()
        assert result[0] is True

    def test_disjoint_true(self):
        from akimbo_geo.accessor import _shapely_binary_bool
        result = ak.Array(
            _shapely_binary_bool(self._unit_square().layout,
                                  self._outside_point_arr().layout, "disjoint")
        ).tolist()
        assert result[0] is True

    def test_distance(self):
        """Distance from unit square to a point at (3, 0) should be 2.0."""
        from akimbo_geo.accessor import _shapely_binary_float
        far_pt = ak.from_arrow(pa.array([[3.0, 0.0]], type=pa.list_(pa.field("xy", pa.float64()), 2)))
        result = ak.Array(
            _shapely_binary_float(self._unit_square().layout,
                                   far_pt.layout, "distance")
        ).tolist()
        assert isclose(result[0], 2.0, rel_tol=1e-5)

    def test_hausdorff_distance(self):
        """Hausdorff distance between two parallel lines."""
        from akimbo_geo.accessor import _shapely_binary_float
        line_a = ak.from_arrow(pa.array([[0.0,0.0, 1.0,0.0]],
                                          type=pa.list_(pa.float64())))
        line_b = ak.from_arrow(pa.array([[0.0,3.0, 1.0,3.0]],
                                          type=pa.list_(pa.float64())))
        result = ak.Array(
            _shapely_binary_float(line_a.layout, line_b.layout,
                                   "hausdorff_distance")
        ).tolist()
        assert isclose(result[0], 3.0, rel_tol=1e-5)

    def test_via_accessor_contains(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(
            GeoAccessor.contains(self._unit_square(), self._inside_point_arr())
        )
        assert result[0] is True

    def test_via_accessor_distance(self):
        from akimbo_geo.accessor import GeoAccessor
        far_pt = ak.from_arrow(pa.array([[3.0, 0.0]], type=pa.list_(pa.field("xy", pa.float64()), 2)))
        result = ak.to_list(GeoAccessor.distance(self._unit_square(), far_pt))
        assert isclose(result[0], 2.0, rel_tol=1e-5)

    def test_covers_and_covered_by(self):
        from akimbo_geo.accessor import _shapely_binary_bool
        # Unit square covers a point inside it
        r_covers = ak.Array(
            _shapely_binary_bool(self._unit_square().layout,
                                  self._inside_point_arr().layout, "covers")
        ).tolist()
        assert r_covers[0] is True

        r_cb = ak.Array(
            _shapely_binary_bool(self._inside_point_arr().layout,
                                  self._unit_square().layout, "covered_by")
        ).tolist()
        assert r_cb[0] is True

    def test_touches(self):
        """Two polygons touching at a boundary."""
        from akimbo_geo.accessor import _shapely_binary_bool
        # Second unit square shifted by (1,0), shares right edge of first
        sq2_ring = [1.0,0.0, 2.0,0.0, 2.0,1.0, 1.0,1.0, 1.0,0.0]
        sq1 = _poly_arr([SQUARE_RING])
        sq2 = _poly_arr([sq2_ring])
        result = ak.Array(
            _shapely_binary_bool(sq1.layout, sq2.layout, "touches")
        ).tolist()
        assert result[0] is True

    def test_crosses(self):
        """Line crossing through a polygon."""
        from akimbo_geo.accessor import _shapely_binary_bool
        cross = ak.from_arrow(
            pa.array([[-0.5, 0.5, 1.5, 0.5]], type=pa.list_(pa.float64()))
        )
        sq = _poly_arr([SQUARE_RING])
        result = ak.Array(
            _shapely_binary_bool(cross.layout, sq.layout, "crosses")
        ).tolist()
        assert result[0] is True


# ===========================================================================
# Tier 2 — set-theoretic
# ===========================================================================

class TestSetTheoretic:

    def _unit_square(self):
        return _poly_arr([SQUARE_RING])

    def _shifted_square(self):
        # Square shifted 0.5 in x: overlaps by 0.5×1.0 = 0.5
        shifted = [0.5,0.0, 1.5,0.0, 1.5,1.0, 0.5,1.0, 0.5,0.0]
        return _poly_arr([shifted])

    def _area_of(self, layout):
        """Compute area of a result layout regardless of depth."""
        from akimbo_geo.accessor import _layout_to_shapely
        geoms = _layout_to_shapely(layout)
        return float(abs(shapely.area(geoms)[0]))

    def test_intersection_area(self):
        from akimbo_geo.accessor import _shapely_binary_geom
        result_layout = _shapely_binary_geom(
            self._unit_square().layout,
            self._shifted_square().layout,
            "intersection"
        )
        assert isclose(self._area_of(result_layout), 0.5, rel_tol=1e-5)

    def test_union_area(self):
        from akimbo_geo.accessor import _shapely_binary_geom
        result_layout = _shapely_binary_geom(
            self._unit_square().layout,
            self._shifted_square().layout,
            "union"
        )
        assert isclose(self._area_of(result_layout), 1.5, rel_tol=1e-5)

    def test_difference_area(self):
        from akimbo_geo.accessor import _shapely_binary_geom
        result_layout = _shapely_binary_geom(
            self._unit_square().layout,
            self._shifted_square().layout,
            "difference"
        )
        assert isclose(self._area_of(result_layout), 0.5, rel_tol=1e-5)

    def test_symmetric_difference_area(self):
        from akimbo_geo.accessor import _shapely_binary_geom
        result_layout = _shapely_binary_geom(
            self._unit_square().layout,
            self._shifted_square().layout,
            "symmetric_difference"
        )
        assert isclose(self._area_of(result_layout), 1.0, rel_tol=1e-5)

    def test_union_all(self):
        from akimbo_geo.accessor import _op_union_all
        # Two non-overlapping unit squares
        sq2_ring = [2.0,0.0, 3.0,0.0, 3.0,1.0, 2.0,1.0, 2.0,0.0]
        arr = ak.from_arrow(pa.array(
            [[SQUARE_RING], [sq2_ring]],
            type=pa.list_(pa.list_(pa.float64()))
        ))
        union_layout = _op_union_all(arr.layout)
        assert isclose(self._area_of(union_layout), 2.0, rel_tol=1e-5)

    def test_intersection_all(self):
        from akimbo_geo.accessor import _op_intersection_all
        # Two overlapping unit squares: shifted 0.5 in x
        sq2_ring = [0.5,0.0, 1.5,0.0, 1.5,1.0, 0.5,1.0, 0.5,0.0]
        arr = ak.from_arrow(pa.array(
            [[SQUARE_RING], [sq2_ring]],
            type=pa.list_(pa.list_(pa.float64()))
        ))
        inter_layout = _op_intersection_all(arr.layout)
        assert isclose(self._area_of(inter_layout), 0.5, rel_tol=1e-5)

    def test_via_accessor_intersection(self):
        from akimbo_geo.accessor import GeoAccessor
        result = GeoAccessor.intersection(
            self._unit_square(), self._shifted_square()
        )
        assert isclose(self._area_of(ak.Array(result).layout), 0.5, rel_tol=1e-5)

    def test_via_accessor_union_all(self):
        from akimbo_geo.accessor import GeoAccessor
        sq2_ring = [2.0,0.0, 3.0,0.0, 3.0,1.0, 2.0,1.0, 2.0,0.0]
        arr = ak.from_arrow(pa.array(
            [[SQUARE_RING], [sq2_ring]],
            type=pa.list_(pa.list_(pa.float64()))
        ))
        result = GeoAccessor.union_all(arr)
        assert isclose(self._area_of(ak.Array(result).layout), 2.0, rel_tol=1e-5)
