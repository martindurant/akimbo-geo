"""test_edge_cases.py — targeted tests for the remaining uncovered lines.

These test the specific None/edge branches inside _shapely_to_arrow_list inner
functions, the RegularArray size guard in match.py, and the accessor __dir__.
"""

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

import akimbo_geo  # noqa: F401

shapely = pytest.importorskip("shapely", reason="shapely>=2.0 required")
import shapely.geometry as sg  # noqa: E402


# ===========================================================================
# accessor.py line 319 — __dir__
# ===========================================================================

def test_geo_accessor_dir():
    """GeoAccessor.__dir__ returns the expected method list."""
    from akimbo_geo.accessor import GeoAccessor, _METHODS
    d = GeoAccessor().__dir__()
    assert d == _METHODS


# ===========================================================================
# convert.py — None values inside multi-geometry _*_coords helpers
# ===========================================================================

class TestConvertNoneInMulti:
    """None elements inside MultiPoint / Polygon-with-hole / MultiLine arrays."""

    def test_multipoint_with_none_element(self):
        """Array of [MultiPoint, None]."""
        from akimbo_geo.convert import _shapely_to_arrow_list
        mp = sg.MultiPoint([(0.0, 0.0), (1.0, 1.0)])
        arr = np.array([mp, None], dtype=object)
        result = _shapely_to_arrow_list(arr)
        rows = result.to_pylist()
        assert rows[0] is not None
        assert rows[1] is None

    def test_polygon_with_none_element(self):
        """Array of [Polygon, None]."""
        from akimbo_geo.convert import _shapely_to_arrow_list
        poly = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        arr = np.array([poly, None], dtype=object)
        result = _shapely_to_arrow_list(arr)
        rows = result.to_pylist()
        assert rows[0] is not None
        assert rows[1] is None

    def test_multilinestring_with_none_element(self):
        """Array of [MultiLineString, None]."""
        from akimbo_geo.convert import _shapely_to_arrow_list
        mls = sg.MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 4)]])
        arr = np.array([mls, None], dtype=object)
        result = _shapely_to_arrow_list(arr)
        rows = result.to_pylist()
        assert rows[0] is not None
        assert rows[1] is None

    def test_multipolygon_with_none_element(self):
        """Array of [MultiPolygon, None]."""
        from akimbo_geo.convert import _shapely_to_arrow_list
        poly1 = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        mp = sg.MultiPolygon([poly1])
        arr = np.array([mp, None], dtype=object)
        result = _shapely_to_arrow_list(arr)
        rows = result.to_pylist()
        assert rows[0] is not None
        assert rows[1] is None

    def test_to_spatialpandas_explicit_class(self):
        """to_spatialpandas with geom_class passed explicitly (skips depth inference)."""
        spg = pytest.importorskip("spatialpandas.geometry")
        from akimbo_geo.convert import to_spatialpandas
        arr = ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64()))
        )
        result = to_spatialpandas(arr, geom_class=spg.LineArray)
        assert isinstance(result, spg.LineArray)


# ===========================================================================
# match.py — RegularArray size guard and deep recursion None
# ===========================================================================

class TestMatchEdgeCases:

    def test_regular_array_wrong_size_no_match(self):
        """RegularArray whose content is not numeric → not a geometry leaf."""
        from akimbo_geo.match import _geo_layout_of
        # A regular array of booleans — not numeric, should not match
        arr = ak.to_regular(ak.Array([[True, False], [False, True]]))
        geo = _geo_layout_of(arr.layout)
        assert geo is None

    def test_list_wrapping_non_geom_inner(self):
        """list<string> → should not match (inner is not a geometry leaf)."""
        from akimbo_geo.match import _geo_layout_of
        arr = ak.from_arrow(pa.array([["hello", "world"]]))
        geo = _geo_layout_of(arr.layout)
        assert geo is None

    def test_deeply_nested_non_geom(self):
        """list<list<string>> → no match at any depth."""
        from akimbo_geo.match import _geo_layout_of, match_any_geom
        arr = ak.from_arrow(pa.array([[["hello"]]]))
        assert _geo_layout_of(arr.layout) is None
        assert not match_any_geom(arr.layout)

    def test_heuristic_regular_array_at_series_level(self):
        """RegularArray(size=2) heuristic path in _geo_layout_of."""
        from akimbo_geo.match import _geo_layout_of, CoordKind
        # Build a regular (fixed-size) array from Python
        arr = ak.to_regular(ak.Array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
        geo = _geo_layout_of(arr.layout)
        assert geo is not None
        assert geo.coord_kind == CoordKind.INTERLEAVED_FSL
        assert geo.list_depth == 0
