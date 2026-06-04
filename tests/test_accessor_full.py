"""test_accessor_full.py — full coverage of accessor.py op functions.

Tests all op functions not covered by existing tests:
- _op_length2 / _op_length3 (polygon/multipolygon perimeter)
- _op_area3 (multipolygon area)
- _op_centroid2 (polygon area-weighted centroid)
- _op_bounds with depth-2 and depth-3 inputs
- _op_intersects_bounds
- GeoAccessor.total_bounds
- GeoAccessor.intersects_bounds
- WKB/WKT delegation methods (from_wkb, to_wkb, from_wkt, to_wkt)
- length2 / length3 / area3 / centroid_polygon staticmethods
"""

from math import isclose

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401
import akimbo_geo     # noqa: F401


# ---------------------------------------------------------------------------
# Shared test geometry data
# ---------------------------------------------------------------------------

# Unit square polygon (flat/spatialpandas): (0,0),(1,0),(1,1),(0,1),(0,0)
SQUARE_FLAT   = [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]
SQUARE_PA     = pa.array([SQUARE_FLAT], type=pa.list_(pa.list_(pa.float64())))

# Two unit squares as MultiPolygon
MPOLY_PA = pa.array(
    [[
        [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
        [[2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0]],
    ]],
    type=pa.list_(pa.list_(pa.list_(pa.float64())))
)


def _arr(pa_array):
    return ak.from_arrow(pa_array)


# ===========================================================================
# _op_length2 — Polygon/MultiLine perimeter (depth-2)
# ===========================================================================

class TestOpLength2:

    def test_polygon_perimeter(self):
        from akimbo_geo.accessor import _op_length2
        result = ak.Array(_op_length2(_arr(SQUARE_PA).layout)).tolist()
        # unit square perimeter = 4.0
        assert isclose(result[0], 4.0)

    def test_polygon_perimeter_two(self):
        from akimbo_geo.accessor import _op_length2
        # 2×2 square perimeter = 8.0
        sq2 = pa.array(
            [[[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0]]],
            type=pa.list_(pa.list_(pa.float64()))
        )
        result = ak.Array(_op_length2(_arr(sq2).layout)).tolist()
        assert isclose(result[0], 8.0)

    def test_via_staticmethod(self):
        from akimbo_geo.accessor import GeoAccessor
        arr = _arr(SQUARE_PA)
        result = ak.to_list(GeoAccessor.length2(arr))
        assert isclose(result[0], 4.0)


# ===========================================================================
# _op_length3 — MultiPolygon perimeter (depth-3)
# ===========================================================================

class TestOpLength3:

    def test_multipolygon_perimeter(self):
        from akimbo_geo.accessor import _op_length3
        result = ak.Array(_op_length3(_arr(MPOLY_PA).layout)).tolist()
        # Two unit square perimeters = 8.0
        assert isclose(result[0], 8.0)

    def test_via_staticmethod(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.length3(_arr(MPOLY_PA)))
        assert isclose(result[0], 8.0)


# ===========================================================================
# _op_area3 — MultiPolygon area (depth-3)
# ===========================================================================

class TestOpArea3:

    def test_multipolygon_area(self):
        from akimbo_geo.accessor import _op_area3
        result = ak.Array(_op_area3(_arr(MPOLY_PA).layout)).tolist()
        assert isclose(abs(result[0]), 2.0)

    def test_via_staticmethod(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.area3(_arr(MPOLY_PA)))
        assert isclose(abs(result[0]), 2.0)


# ===========================================================================
# _op_bounds with depth-2 and depth-3 inputs
# ===========================================================================

class TestOpBoundsDeep:

    def test_bounds_depth2(self):
        from akimbo_geo.accessor import _op_bounds
        result = ak.Array(_op_bounds(_arr(SQUARE_PA).layout)).tolist()
        r = result[0]
        assert r["xmin"] == 0.0 and r["ymin"] == 0.0
        assert r["xmax"] == 1.0 and r["ymax"] == 1.0

    def test_bounds_depth3(self):
        from akimbo_geo.accessor import _op_bounds
        result = ak.Array(_op_bounds(_arr(MPOLY_PA).layout)).tolist()
        r = result[0]
        assert r["xmin"] == 0.0 and r["ymin"] == 0.0
        assert r["xmax"] == 3.0 and r["ymax"] == 1.0

    def test_bounds_via_staticmethod_depth2(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.bounds(_arr(SQUARE_PA)))
        assert result[0]["xmax"] == 1.0

    def test_bounds_via_staticmethod_depth3(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.bounds(_arr(MPOLY_PA)))
        assert result[0]["xmax"] == 3.0


# ===========================================================================
# _op_centroid2 — area-weighted polygon centroid (depth-2)
# ===========================================================================

class TestOpCentroid2:

    def test_centroid_unit_square(self):
        from akimbo_geo.accessor import _op_centroid2
        result = ak.Array(_op_centroid2(_arr(SQUARE_PA).layout)).tolist()
        assert isclose(result[0]["x"], 0.5, rel_tol=1e-6)
        assert isclose(result[0]["y"], 0.5, rel_tol=1e-6)

    def test_centroid_asymmetric_polygon(self):
        """Right triangle: (0,0),(2,0),(0,2). Centroid should be (2/3, 2/3)."""
        from akimbo_geo.accessor import _op_centroid2
        pa_arr = pa.array(
            [[[0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0]]],
            type=pa.list_(pa.list_(pa.float64()))
        )
        result = ak.Array(_op_centroid2(_arr(pa_arr).layout)).tolist()
        assert isclose(result[0]["x"], 2/3, rel_tol=1e-5)
        assert isclose(result[0]["y"], 2/3, rel_tol=1e-5)

    def test_via_staticmethod(self):
        from akimbo_geo.accessor import GeoAccessor
        result = ak.to_list(GeoAccessor.centroid_polygon(_arr(SQUARE_PA)))
        assert isclose(result[0]["x"], 0.5, rel_tol=1e-6)


# ===========================================================================
# _op_intersects_bounds / GeoAccessor.intersects_bounds
# ===========================================================================

class TestOpIntersectsBounds:

    def _line_arr(self):
        # Three lines: inside, outside, crossing
        pa_arr = pa.array(
            [
                [0.5, 0.5, 0.8, 0.8],   # entirely inside [0,0,1,1]
                [5.0, 5.0, 6.0, 6.0],   # entirely outside
                [-1.0, 0.5, 2.0, 0.5],  # crosses
            ],
            type=pa.list_(pa.float64())
        )
        return ak.from_arrow(pa_arr)

    def test_op_intersects_bounds_direct(self):
        from akimbo_geo.accessor import _op_intersects_bounds
        arr = self._line_arr()
        result = ak.Array(_op_intersects_bounds(arr.layout, 0.0, 0.0, 1.0, 1.0)).tolist()
        assert result[0] is True
        assert result[1] is False
        assert result[2] is True

    def test_via_staticmethod(self):
        from akimbo_geo.accessor import GeoAccessor
        arr = self._line_arr()
        result = ak.to_list(GeoAccessor.intersects_bounds(arr, 0.0, 0.0, 1.0, 1.0))
        assert result[0] is True
        assert result[1] is False
        assert result[2] is True


# ===========================================================================
# GeoAccessor.total_bounds
# ===========================================================================

class TestTotalBounds:

    def test_line_array(self):
        from akimbo_geo.accessor import GeoAccessor
        pa_arr = pa.array(
            [[0.0, 0.0, 3.0, 4.0], [-1.0, 2.0, 5.0, -3.0]],
            type=pa.list_(pa.float64())
        )
        arr = ak.from_arrow(pa_arr)
        xmin, ymin, xmax, ymax = GeoAccessor.total_bounds(arr)
        assert xmin == -1.0
        assert ymin == -3.0
        assert xmax == 5.0
        assert ymax == 4.0

    def test_polygon_array(self):
        from akimbo_geo.accessor import GeoAccessor
        arr = _arr(SQUARE_PA)
        xmin, ymin, xmax, ymax = GeoAccessor.total_bounds(arr)
        assert xmin == 0.0 and ymin == 0.0
        assert xmax == 1.0 and ymax == 1.0


# ===========================================================================
# WKB / WKT delegation methods
# ===========================================================================

shapely = pytest.importorskip("shapely", reason="shapely>=2.0 required")


class TestWKBWKTDelegation:
    """Tests that GeoAccessor.from_wkb / to_wkb / from_wkt / to_wkt delegate
    correctly to convert.py and produce round-trippable results."""

    def _wkb_arr(self):
        import shapely.geometry as sg
        geoms = [sg.LineString([(0, 0), (3, 4)]), sg.LineString([(1, 1), (2, 2)])]
        wkb = [shapely.to_wkb(g) for g in geoms]
        return ak.from_arrow(pa.array(wkb, type=pa.large_binary()))

    def _coord_arr(self):
        return ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0], [1.0, 1.0, 2.0, 2.0]],
                     type=pa.list_(pa.float64()))
        )

    def test_from_wkb(self):
        from akimbo_geo.accessor import GeoAccessor
        result = GeoAccessor.from_wkb(self._wkb_arr())
        coords = ak.to_list(result)
        assert isclose(coords[0][2], 3.0)  # second point x

    def test_to_wkb(self):
        from akimbo_geo.accessor import GeoAccessor
        result = GeoAccessor.to_wkb(self._coord_arr())
        wkb_bytes = ak.to_list(result)[0]
        g = shapely.from_wkb(bytes(wkb_bytes))
        assert isclose(g.length, 5.0)

    def test_from_wkt(self):
        from akimbo_geo.accessor import GeoAccessor
        wkt_arr = ak.from_arrow(
            pa.array(["LINESTRING (0 0, 3 4)", "LINESTRING (1 1, 2 2)"],
                     type=pa.large_string())
        )
        result = GeoAccessor.from_wkt(wkt_arr)
        coords = ak.to_list(result)
        assert isclose(coords[0][2], 3.0)

    def test_to_wkt(self):
        from akimbo_geo.accessor import GeoAccessor
        result = GeoAccessor.to_wkt(self._coord_arr())
        wkt_str = ak.to_list(result)[0]
        assert "LINESTRING" in wkt_str.upper()

    def test_wkb_round_trip(self):
        from akimbo_geo.accessor import GeoAccessor
        coord_arr = self._coord_arr()
        wkb = GeoAccessor.to_wkb(coord_arr)
        back = GeoAccessor.from_wkb(wkb)
        original = ak.to_list(coord_arr)
        recovered = ak.to_list(back)
        for orig, rec in zip(original, recovered):
            np.testing.assert_allclose(orig, rec, rtol=1e-10)

    def test_wkt_round_trip(self):
        from akimbo_geo.accessor import GeoAccessor
        coord_arr = self._coord_arr()
        wkt = GeoAccessor.to_wkt(coord_arr)
        back = GeoAccessor.from_wkt(wkt)
        original = ak.to_list(coord_arr)
        recovered = ak.to_list(back)
        for orig, rec in zip(original, recovered):
            np.testing.assert_allclose(orig, rec, rtol=1e-10)
