"""test_convert.py — tests for WKB/WKT conversion and framework bridges.

Shapely is required for these tests.  The test module is guarded with
``pytest.importorskip`` so that the suite still passes in environments
without shapely, spatialpandas, or geopandas.
"""

from math import isclose

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

shapely = pytest.importorskip("shapely", reason="shapely>=2.0 required")
import shapely.geometry as sg  # noqa: E402 — after importorskip


# ===========================================================================
# WKB round-trip
# ===========================================================================

class TestWKB:
    def _make_wkb_series(self, geometries):
        """Encode shapely geometries to a pa.array of WKB large_binary."""
        wkb_list = [shapely.to_wkb(g) for g in geometries]
        return ak.from_arrow(pa.array(wkb_list, type=pa.large_binary()))

    def test_from_wkb_linestring(self):
        from akimbo_geo.convert import from_wkb
        line = sg.LineString([(0, 0), (3, 4)])
        arr = self._make_wkb_series([line])
        result = from_wkb(arr)
        coords = ak.to_list(result)[0]
        assert isclose(coords[0], 0.0)
        assert isclose(coords[1], 0.0)
        assert isclose(coords[2], 3.0)
        assert isclose(coords[3], 4.0)

    def test_from_wkb_polygon(self):
        from akimbo_geo.convert import from_wkb
        poly = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        arr = self._make_wkb_series([poly])
        result = from_wkb(arr)
        # Should be list<list<float>> — 1 polygon with 1 ring
        coords = ak.to_list(result)
        assert len(coords) == 1            # 1 polygon
        assert isinstance(coords[0], list) # rings
        assert isinstance(coords[0][0], list)  # ring coords

    def test_to_wkb_linestring(self):
        from akimbo_geo.convert import to_wkb
        data = [[0.0, 0.0, 3.0, 4.0]]
        arr = ak.from_arrow(pa.array(data, type=pa.list_(pa.float64())))
        result = to_wkb(arr)
        wkb_bytes = ak.to_list(result)[0]
        reconstructed = shapely.from_wkb(bytes(wkb_bytes))
        assert isclose(reconstructed.length, 5.0)

    def test_round_trip_linestring(self):
        from akimbo_geo.convert import from_wkb, to_wkb
        line = sg.LineString([(0, 0), (1, 0), (1, 1)])
        wkb_arr = self._make_wkb_series([line])
        coord_arr = from_wkb(wkb_arr)
        wkb_back = to_wkb(coord_arr)
        wkb_bytes = ak.to_list(wkb_back)[0]
        reconstructed = shapely.from_wkb(bytes(wkb_bytes))
        assert isclose(reconstructed.length, line.length)

    def test_round_trip_polygon(self):
        from akimbo_geo.convert import from_wkb, to_wkb
        poly = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        wkb_arr = self._make_wkb_series([poly])
        coord_arr = from_wkb(wkb_arr)
        wkb_back = to_wkb(coord_arr)
        wkb_bytes = ak.to_list(wkb_back)[0]
        reconstructed = shapely.from_wkb(bytes(wkb_bytes))
        assert isclose(abs(reconstructed.area), abs(poly.area))

    def test_multiple_geometries(self):
        from akimbo_geo.convert import from_wkb
        geoms = [
            sg.LineString([(0, 0), (3, 4)]),
            sg.LineString([(1, 1), (2, 2)]),
        ]
        arr = self._make_wkb_series(geoms)
        result = from_wkb(arr)
        decoded = ak.to_list(result)
        assert len(decoded) == 2
        assert isclose(decoded[0][2], 3.0)  # second point x
        assert isclose(decoded[1][0], 1.0)  # first point x


# ===========================================================================
# WKT round-trip
# ===========================================================================

class TestWKT:
    def _make_wkt_series(self, geometries):
        wkt_list = [shapely.to_wkt(g) for g in geometries]
        return ak.from_arrow(pa.array(wkt_list, type=pa.large_string()))

    def test_from_wkt_linestring(self):
        from akimbo_geo.convert import from_wkt
        line = sg.LineString([(0, 0), (3, 4)])
        arr = self._make_wkt_series([line])
        result = from_wkt(arr)
        coords = ak.to_list(result)[0]
        assert isclose(coords[2], 3.0)
        assert isclose(coords[3], 4.0)

    def test_to_wkt_linestring(self):
        from akimbo_geo.convert import to_wkt
        data = [[0.0, 0.0, 3.0, 4.0]]
        arr = ak.from_arrow(pa.array(data, type=pa.list_(pa.float64())))
        result = to_wkt(arr)
        wkt_str = ak.to_list(result)[0]
        assert "LINESTRING" in wkt_str.upper()

    def test_round_trip_wkt(self):
        from akimbo_geo.convert import from_wkt, to_wkt
        line = sg.LineString([(0, 0), (1, 0), (1, 1)])
        wkt_arr = self._make_wkt_series([line])
        coord_arr = from_wkt(wkt_arr)
        wkt_back = to_wkt(coord_arr)
        wkt_str = ak.to_list(wkt_back)[0]
        reconstructed = shapely.from_wkt(wkt_str)
        assert isclose(reconstructed.length, line.length)


# ===========================================================================
# spatialpandas bridge
# ===========================================================================

class TestSpatialPandas:
    spg = pytest.importorskip("spatialpandas.geometry")

    def test_from_spatialpandas_line(self):
        from akimbo_geo.convert import from_spatialpandas
        spg = pytest.importorskip("spatialpandas.geometry")
        line_arr = spg.LineArray([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]])
        result = from_spatialpandas(line_arr)
        coords = ak.to_list(result)[0]
        assert isclose(coords[0], 0.0)
        assert isclose(coords[2], 1.0)

    def test_from_spatialpandas_polygon(self):
        from akimbo_geo.convert import from_spatialpandas
        spg = pytest.importorskip("spatialpandas.geometry")
        poly_arr = spg.PolygonArray(
            [[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]]
        )
        result = from_spatialpandas(poly_arr)
        # Should be list<list<float>> — depth 2
        coords = ak.to_list(result)
        assert len(coords) == 1
        assert isinstance(coords[0][0], list)

    def test_from_spatialpandas_zero_copy(self):
        """from_spatialpandas wraps the same Arrow buffer — no data copy."""
        from akimbo_geo.convert import from_spatialpandas
        spg = pytest.importorskip("spatialpandas.geometry")
        line_arr = spg.LineArray([[0.0, 0.0, 3.0, 4.0]])
        result = from_spatialpandas(line_arr)
        # Both share the same underlying buffer address
        import pyarrow as pa
        pa_back = ak.to_arrow(result)
        assert pa_back.buffers()[-1].address == line_arr.data.buffers()[-1].address

    def test_to_spatialpandas_line(self):
        from akimbo_geo.convert import to_spatialpandas
        spg = pytest.importorskip("spatialpandas.geometry")
        arr = ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64()))
        )
        result = to_spatialpandas(arr)
        assert isinstance(result, spg.LineArray)
        assert isclose(result[0].length, 5.0)

    def test_to_spatialpandas_polygon(self):
        from akimbo_geo.convert import to_spatialpandas
        spg = pytest.importorskip("spatialpandas.geometry")
        arr = ak.from_arrow(
            pa.array(
                [[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]],
                type=pa.list_(pa.list_(pa.float64())),
            )
        )
        result = to_spatialpandas(arr)
        assert isinstance(result, spg.PolygonArray)
        assert isclose(abs(result[0].area), 1.0)

    def test_round_trip_spatialpandas(self):
        """from_spatialpandas → to_spatialpandas gives equivalent array."""
        from akimbo_geo.convert import from_spatialpandas, to_spatialpandas
        spg = pytest.importorskip("spatialpandas.geometry")
        original = spg.LineArray([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]])
        ak_arr = from_spatialpandas(original)
        back = to_spatialpandas(ak_arr, geom_class=spg.LineArray)
        assert isclose(back[0].length, original[0].length)


# ===========================================================================
# geopandas bridge
# ===========================================================================

class TestGeoPandas:
    gpd = pytest.importorskip("geopandas")

    def test_from_geopandas_linestring(self):
        from akimbo_geo.convert import from_geopandas
        gpd = pytest.importorskip("geopandas")
        gs = gpd.GeoSeries([sg.LineString([(0, 0), (3, 4)])])
        result = from_geopandas(gs)
        coords = ak.to_list(result)[0]
        assert isclose(coords[2], 3.0)
        assert isclose(coords[3], 4.0)

    def test_from_geopandas_polygon(self):
        from akimbo_geo.convert import from_geopandas
        gpd = pytest.importorskip("geopandas")
        gs = gpd.GeoSeries([sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
        result = from_geopandas(gs)
        coords = ak.to_list(result)
        assert len(coords) == 1
        assert isinstance(coords[0][0], list)  # nested rings

    def test_to_geopandas_linestring(self):
        from akimbo_geo.convert import to_geopandas
        pytest.importorskip("geopandas")
        arr = ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64()))
        )
        gs = to_geopandas(arr)
        assert isclose(gs.iloc[0].length, 5.0)

    def test_round_trip_geopandas(self):
        from akimbo_geo.convert import from_geopandas, to_geopandas
        gpd = pytest.importorskip("geopandas")
        gs = gpd.GeoSeries([
            sg.LineString([(0, 0), (1, 0), (1, 1)]),
            sg.LineString([(0, 0), (3, 4)]),
        ])
        ak_arr = from_geopandas(gs)
        gs_back = to_geopandas(ak_arr)
        for orig, back in zip(gs, gs_back):
            assert isclose(orig.length, back.length)
