"""test_accessor.py — tests for the GeoAccessor sub-accessor.

These tests use only core dependencies (akimbo, numpy, pyarrow, numba)
and do not require shapely, spatialpandas, or geopandas.
"""

from math import isclose, sqrt

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401
import akimbo_geo     # noqa: F401 — registers .ak.geo


# ===========================================================================
# Helpers
# ===========================================================================

def _to_ak(pa_array):
    return ak.from_arrow(pa_array)


def _to_series(pa_array):
    """Wrap a pa.Array as a pandas ArrowDtype Series so .ak works."""
    return pd.Series(pa_array, dtype=pd.ArrowDtype(pa_array.type))


# ===========================================================================
# match.py unit tests
# ===========================================================================

class TestMatch:
    def test_match_line(self):
        from akimbo_geo.match import match_line
        arr = ak.from_arrow(pa.array([[1.0, 2.0, 3.0, 4.0]], type=pa.list_(pa.float64())))
        layout = arr.layout.content  # the inner ListArray layout
        assert match_line(arr.layout)

    def test_match_polygon(self):
        from akimbo_geo.match import match_polygon
        data = [[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]]
        arr = ak.from_arrow(pa.array(data, type=pa.list_(pa.list_(pa.float64()))))
        assert match_polygon(arr.layout)

    def test_match_multipolygon(self):
        from akimbo_geo.match import match_multipolygon
        data = [[[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]]]
        arr = ak.from_arrow(
            pa.array(data, type=pa.list_(pa.list_(pa.list_(pa.float64()))))
        )
        assert match_multipolygon(arr.layout)

    def test_match_any_geom(self):
        from akimbo_geo.match import match_any_geom
        for data, ptype in [
            ([[1.0, 2.0]], pa.list_(pa.float64())),
            ([[[1.0, 2.0]]], pa.list_(pa.list_(pa.float64()))),
            ([[[[1.0, 2.0]]]], pa.list_(pa.list_(pa.list_(pa.float64())))),
        ]:
            arr = ak.from_arrow(pa.array(data, type=ptype))
            assert match_any_geom(arr.layout), f"failed for {ptype}"

    def test_no_match_plain_float(self):
        from akimbo_geo.match import match_any_geom
        arr = ak.from_arrow(pa.array([1.0, 2.0, 3.0], type=pa.float64()))
        assert not match_any_geom(arr.layout)

    def test_no_match_string(self):
        from akimbo_geo.match import match_any_geom
        arr = ak.from_arrow(pa.array(["hello", "world"]))
        assert not match_any_geom(arr.layout)


# ===========================================================================
# extract_offsets_and_values
# ===========================================================================

class TestExtract:
    def test_1level(self):
        from akimbo_geo.match import extract_offsets_and_values
        data = [[0.0, 0.0, 1.0, 0.0], [2.0, 2.0, 3.0, 3.0]]
        arr = ak.from_arrow(pa.array(data, type=pa.list_(pa.float64())))
        values, offsets = extract_offsets_and_values(arr.layout)
        assert len(offsets) == 1
        assert list(offsets[0]) == [0, 4, 8]
        np.testing.assert_allclose(values, [0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 3.0, 3.0])

    def test_2level(self):
        from akimbo_geo.match import extract_offsets_and_values
        data = [[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]]
        arr = ak.from_arrow(pa.array(data, type=pa.list_(pa.list_(pa.float64()))))
        values, offsets = extract_offsets_and_values(arr.layout)
        assert len(offsets) == 2
        assert len(values) == 8


# ===========================================================================
# algorithms.py unit tests
# ===========================================================================

class TestAlgorithms:
    def test_compute_line_length_simple(self):
        from akimbo_geo.algorithms import compute_line_length
        # (0,0)→(3,4): length = 5
        values = np.array([0.0, 0.0, 3.0, 4.0])
        offsets = np.array([0, 4], dtype=np.int32)
        result = compute_line_length(values, offsets)
        assert isclose(result, 5.0)

    def test_compute_line_length_multi_segment(self):
        from akimbo_geo.algorithms import compute_line_length
        # (0,0)→(1,0)→(1,1): length = 2
        values = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        offsets = np.array([0, 6], dtype=np.int32)
        result = compute_line_length(values, offsets)
        assert isclose(result, 2.0)

    def test_compute_area_unit_square(self):
        from akimbo_geo.algorithms import compute_area
        # CCW unit square
        values = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        offsets = np.array([0, 10], dtype=np.int32)
        result = compute_area(values, offsets)
        assert isclose(abs(result), 1.0), f"got {result}"

    def test_compute_area_degenerate(self):
        from akimbo_geo.algorithms import compute_area
        # Only 2 points — degenerate, area = 0
        values = np.array([0.0, 0.0, 1.0, 1.0])
        offsets = np.array([0, 4], dtype=np.int32)
        assert compute_area(values, offsets) == 0.0

    def test_compute_bounds(self):
        from akimbo_geo.algorithms import compute_bounds
        values = np.array([1.0, 2.0, 5.0, 0.0, 3.0, 4.0])
        xmin, ymin, xmax, ymax = compute_bounds(values, 0, 6)
        assert xmin == 1.0 and ymin == 0.0 and xmax == 5.0 and ymax == 4.0

    def test_compute_centroid_line(self):
        from akimbo_geo.algorithms import compute_centroid_line
        # Two points: (0,0) and (2,2) → centroid = (1,1)
        values = np.array([0.0, 0.0, 2.0, 2.0])
        cx, cy = compute_centroid_line(values, 0, 4)
        assert isclose(cx, 1.0) and isclose(cy, 1.0)

    def test_total_bounds(self):
        from akimbo_geo.algorithms import total_bounds
        values = np.array([1.0, 5.0, 3.0, 2.0, 0.0, 4.0])
        xmin, ymin, xmax, ymax = total_bounds(values)
        assert xmin == 0.0 and ymin == 2.0 and xmax == 3.0 and ymax == 5.0

    def test_length_map1(self):
        from akimbo_geo.algorithms import length_map1
        # Three lines
        values = np.array([
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0,   # line 0: length 2
            0.0, 0.0, 3.0, 4.0,               # line 1: length 5
            0.0, 0.0, 1.0, 0.0,               # line 2: length 1
        ])
        offsets = np.array([0, 6, 10, 14], dtype=np.int32)
        result = np.full(3, np.nan)
        missing = np.zeros(3, dtype=np.bool_)
        length_map1(values, offsets, result, missing)
        assert isclose(result[0], 2.0)
        assert isclose(result[1], 5.0)
        assert isclose(result[2], 1.0)

    def test_length_map1_respects_missing(self):
        from akimbo_geo.algorithms import length_map1
        values = np.array([0.0, 0.0, 1.0, 0.0])
        offsets = np.array([0, 4], dtype=np.int32)
        result = np.full(1, np.nan)
        missing = np.ones(1, dtype=np.bool_)  # marked missing
        length_map1(values, offsets, result, missing)
        assert np.isnan(result[0])

    def test_area_map2_unit_square(self):
        from akimbo_geo.algorithms import area_map2
        # One polygon: unit square
        values = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        offsets0 = np.array([0, 1], dtype=np.int32)   # 1 polygon, 1 ring
        offsets1 = np.array([0, 10], dtype=np.int32)  # ring spans all values
        result = np.full(1, np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        area_map2(values, offsets0, offsets1, result, missing)
        assert isclose(abs(result[0]), 1.0)

    def test_bounds_map1(self):
        from akimbo_geo.algorithms import bounds_map1
        values = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 3.0])
        offsets = np.array([0, 6], dtype=np.int32)
        result = np.full((1, 4), np.nan)
        missing = np.zeros(1, dtype=np.bool_)
        bounds_map1(values, offsets, result, missing)
        assert result[0, 0] == 0.0  # xmin
        assert result[0, 1] == 0.0  # ymin
        assert result[0, 2] == 2.0  # xmax
        assert result[0, 3] == 3.0  # ymax


# ===========================================================================
# GeoAccessor via ak.Array directly
# ===========================================================================

class TestGeoAccessorDirect:
    """Tests calling op functions directly on ak.Array (no pandas needed)."""

    def test_length_lines(self, line_array):
        result = ak.to_list(
            GeoAccessor_length_call(line_array)
        )
        assert isclose(result[0], 2.0)
        assert isclose(result[1], 5.0)
        assert isclose(result[2], 4.0)

    def test_area_polygon(self, polygon_array):
        from akimbo_geo.accessor import _op_area
        from akimbo_geo.match import extract_offsets_and_values
        import awkward as ak
        result_layout = _op_area(polygon_array.layout)
        result = ak.Array(result_layout).tolist()
        assert isclose(abs(result[0]), 1.0), f"unit square area={result[0]}"
        # 2x2 square − 0.5×0.5 hole = 3.75
        assert isclose(abs(result[1]), 3.75, rel_tol=1e-6), f"got {result[1]}"

    def test_bounds_line(self, line_array):
        from akimbo_geo.accessor import _op_bounds
        result_layout = _op_bounds(line_array.layout)
        result = ak.Array(result_layout).tolist()
        # Line 0: (0,0)→(1,0)→(1,1)  bounds = (0,0,1,1)
        assert result[0]["xmin"] == 0.0
        assert result[0]["ymin"] == 0.0
        assert result[0]["xmax"] == 1.0
        assert result[0]["ymax"] == 1.0

    def test_centroid_line(self, line_array):
        from akimbo_geo.accessor import _op_centroid1
        result_layout = _op_centroid1(line_array.layout)
        result = ak.Array(result_layout).tolist()
        # Line 1: (0,0)→(3,4)  mean = (1.5, 2.0)
        assert isclose(result[1]["x"], 1.5)
        assert isclose(result[1]["y"], 2.0)

    def test_total_bounds(self, line_array):
        from akimbo_geo.accessor import GeoAccessor
        xmin, ymin, xmax, ymax = GeoAccessor.total_bounds(line_array)
        assert xmin == 0.0 and ymin == 0.0
        assert xmax == 3.0 and ymax == 4.0

    def test_intersects_bounds(self, line_array):
        from akimbo_geo.accessor import GeoAccessor
        # bbox (0,0,2,2) should intersect lines 0 and 2 but not line 1 (3,4 outside)
        result = ak.to_list(GeoAccessor.intersects_bounds(line_array, 0.0, 0.0, 2.0, 2.0))
        assert result[0] is True   # line 0 entirely inside bbox
        assert result[2] is True   # line 2 entirely inside bbox


def GeoAccessor_length_call(arr):
    """Helper: call the length dec'd function on a line array."""
    from akimbo_geo.accessor import _op_length
    return ak.Array(_op_length(arr.layout))


# ===========================================================================
# GeoAccessor via pandas .ak.geo
# ===========================================================================

class TestGeoAccessorPandas:
    def test_accessor_registered(self):
        """Verify .ak.geo is accessible on a pandas Series."""
        s = pd.Series(
            pa.array([[1.0, 2.0, 3.0, 4.0]], type=pa.list_(pa.float64())),
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
        )
        assert hasattr(s.ak, "geo")

    def test_length_via_pandas(self, line_series):
        result = line_series.ak.geo.length(line_series.ak.array)
        values = ak.to_list(result)
        assert isclose(values[0], 2.0)
        assert isclose(values[1], 5.0)
        assert isclose(values[2], 4.0)

    def test_area_via_pandas(self, polygon_series):
        result = polygon_series.ak.geo.area(polygon_series.ak.array)
        values = ak.to_list(result)
        assert isclose(abs(values[0]), 1.0)
        assert isclose(abs(values[1]), 3.75, rel_tol=1e-6)


# ===========================================================================
# Nested structure — list of lines per row
# ===========================================================================

class TestNestedStructures:
    """Verify dec() correctly descends into list-of-geometries structures."""

    def test_list_of_lines(self):
        """Each row is a variable-length list of line geometries."""
        from akimbo_geo.accessor import _op_length
        from akimbo_geo.match import match_line
        from akimbo.apply_tree import dec

        # Row 0: two lines; Row 1: one line
        data = [
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 3.0, 4.0]],
            [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]],
        ]
        pa_arr = pa.array(data, type=pa.list_(pa.list_(pa.float64())))
        arr = ak.from_arrow(pa_arr)

        length_fn = dec(_op_length, match=match_line, inmode="ak")
        result = ak.to_list(length_fn(arr))

        # Row 0: [1.0, 5.0]; Row 1: [2.0]
        assert isclose(result[0][0], 1.0)
        assert isclose(result[0][1], 5.0)
        assert isclose(result[1][0], 2.0)

    def test_record_with_geometry_field(self):
        """Records where one field contains line geometries."""
        from akimbo_geo.accessor import _op_length
        from akimbo_geo.match import match_line
        from akimbo.apply_tree import dec

        arr = ak.Array({
            "id": [1, 2],
            "geom": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 3.0, 4.0]],
        })
        length_fn = dec(_op_length, match=match_line, inmode="ak")
        result = length_fn(arr, where="geom")
        lengths = ak.to_list(result["geom"])
        assert isclose(lengths[0], 1.0)
        assert isclose(lengths[1], 5.0)
