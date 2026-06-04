"""test_optional_deps.py — verify that shapely-dependent operations raise a
clear ImportError when shapely is unavailable, and that pure-numba operations
work without shapely.

These tests use ``monkeypatch`` to hide shapely from the import system,
simulating an environment where it is not installed.
"""

import sys
import importlib

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401
import akimbo_geo     # noqa: F401


# ---------------------------------------------------------------------------
# Helper: temporarily remove shapely from sys.modules
# ---------------------------------------------------------------------------

@pytest.fixture()
def no_shapely(monkeypatch):
    """Make shapely appear uninstalled for the duration of a test."""
    # Remove shapely and all sub-modules from the import cache
    shapely_keys = [k for k in sys.modules if k == "shapely" or k.startswith("shapely.")]
    saved = {k: sys.modules.pop(k) for k in shapely_keys}

    # Block future imports
    class _BlockedFinder:
        def find_spec(self, name, path, target=None):
            if name == "shapely" or name.startswith("shapely."):
                raise ModuleNotFoundError(f"No module named '{name}' (blocked by test)")
            return None

    blocker = _BlockedFinder()
    sys.meta_path.insert(0, blocker)

    # Also invalidate cached require_shapely result in _compat
    from akimbo_geo import _compat
    monkeypatch.delattr(_compat, "shapely", raising=False)

    yield

    # Restore
    sys.meta_path.remove(blocker)
    sys.modules.update(saved)
    # Re-import shapely to restore the module state
    importlib.import_module("shapely")


# ===========================================================================
# require_shapely raises clearly when shapely is absent
# ===========================================================================

class TestRequireShapely:

    def test_require_shapely_raises(self, no_shapely):
        from akimbo_geo._compat import require_shapely
        with pytest.raises(ImportError, match="shapely >= 2.0 is required"):
            require_shapely()

    def test_error_message_mentions_install_command(self, no_shapely):
        from akimbo_geo._compat import require_shapely
        with pytest.raises(ImportError, match="pip install"):
            require_shapely()

    def test_error_message_mentions_pure_numba(self, no_shapely):
        """Error message reassures user that numba ops still work."""
        from akimbo_geo._compat import require_shapely
        with pytest.raises(ImportError, match="Pure-numba"):
            require_shapely()


# ===========================================================================
# Pure-numba operations work without shapely
# ===========================================================================

class TestNumbaOpsWithoutShapely:
    """These operations must not import shapely at all."""

    def _line_arr(self):
        return ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64()))
        )

    def _poly_arr(self):
        ring = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        return ak.from_arrow(
            pa.array([[ring]], type=pa.list_(pa.list_(pa.float64())))
        )

    def test_length(self, no_shapely):
        from akimbo_geo.accessor import _op_length
        result = ak.Array(_op_length(self._line_arr().layout)).tolist()
        assert abs(result[0] - 5.0) < 1e-10

    def test_area(self, no_shapely):
        from akimbo_geo.accessor import _op_area
        result = ak.Array(_op_area(self._poly_arr().layout)).tolist()
        assert abs(abs(result[0]) - 1.0) < 1e-10

    def test_bounds(self, no_shapely):
        from akimbo_geo.accessor import _op_bounds
        result = ak.Array(_op_bounds(self._line_arr().layout)).tolist()
        assert result[0]["xmin"] == 0.0
        assert result[0]["xmax"] == 3.0

    def test_centroid(self, no_shapely):
        from akimbo_geo.accessor import _op_centroid1
        result = ak.Array(_op_centroid1(self._line_arr().layout)).tolist()
        assert abs(result[0]["x"] - 1.5) < 1e-10

    def test_translate(self, no_shapely):
        from akimbo_geo.accessor import _op_translate
        result = ak.Array(
            _op_translate(self._line_arr().layout, 1.0, 2.0)
        ).tolist()
        assert abs(result[0][0] - 1.0) < 1e-10
        assert abs(result[0][1] - 2.0) < 1e-10

    def test_is_closed(self, no_shapely):
        from akimbo_geo.accessor import _op_is_closed
        result = ak.Array(_op_is_closed(self._line_arr().layout)).tolist()
        assert result[0] is False

    def test_is_ring(self, no_shapely):
        from akimbo_geo.accessor import _op_is_ring
        ring = ak.from_arrow(
            pa.array(
                [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
                type=pa.list_(pa.float64())
            )
        )
        result = ak.Array(_op_is_ring(ring.layout)).tolist()
        assert result[0] is True

    def test_count_coordinates(self, no_shapely):
        from akimbo_geo.accessor import _op_count_coords
        result = ak.Array(_op_count_coords(self._line_arr().layout)).tolist()
        assert result[0] == 2

    def test_segmentize(self, no_shapely):
        from akimbo_geo.accessor import _op_segmentize
        result = ak.Array(
            _op_segmentize(self._line_arr().layout, 2.0)
        ).tolist()
        # (0,0)→(3,4) length=5, max_seg=2 → 2 segments → 3 points = 6 floats
        assert len(result[0]) == 6

    def test_minimum_bounding_radius(self, no_shapely):
        from akimbo_geo.accessor import _op_minimum_bounding_radius
        result = ak.Array(
            _op_minimum_bounding_radius(self._line_arr().layout)
        ).tolist()
        assert result[0] > 0.0

    def test_affine_transform(self, no_shapely):
        from akimbo_geo.accessor import _op_affine_transform
        # identity: x'=x, y'=y
        result = ak.Array(
            _op_affine_transform(self._line_arr().layout, [1, 0, 0, 1, 0, 0])
        ).tolist()
        assert abs(result[0][2] - 3.0) < 1e-10

    def test_reverse(self, no_shapely):
        from akimbo_geo.accessor import _op_reverse
        result = ak.Array(_op_reverse(self._line_arr().layout)).tolist()
        assert abs(result[0][0] - 3.0) < 1e-10

    def test_orient_polygons(self, no_shapely):
        from akimbo_geo.accessor import _op_orient_polygons
        result_layout = _op_orient_polygons(self._poly_arr().layout, exterior_cw=False)
        assert result_layout is not None  # didn't crash

    def test_has_z_false(self, no_shapely):
        from akimbo_geo.accessor import _op_has_z
        result = ak.Array(_op_has_z(self._line_arr().layout)).tolist()
        assert result[0] is False

    def test_total_bounds(self, no_shapely):
        from akimbo_geo.accessor import GeoAccessor
        xmin, ymin, xmax, ymax = GeoAccessor.total_bounds(self._line_arr())
        assert xmin == 0.0 and xmax == 3.0


# ===========================================================================
# Shapely-dependent operations raise ImportError without shapely
# ===========================================================================

class TestShapelyOpsRaiseWithoutShapely:

    def _line_arr(self):
        return ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64()))
        )

    def _poly_arr(self):
        ring = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        return ak.from_arrow(
            pa.array([[ring]], type=pa.list_(pa.list_(pa.float64())))
        )

    def test_is_valid_raises(self, no_shapely):
        from akimbo_geo.accessor import _op_is_valid
        with pytest.raises(ImportError, match="shapely"):
            _op_is_valid(self._poly_arr().layout)

    def test_convex_hull_raises(self, no_shapely):
        from akimbo_geo.accessor import _op_convex_hull
        with pytest.raises(ImportError, match="shapely"):
            _op_convex_hull(self._line_arr().layout)

    def test_buffer_raises(self, no_shapely):
        from akimbo_geo.accessor import _op_buffer
        with pytest.raises(ImportError, match="shapely"):
            _op_buffer(self._poly_arr().layout, distance=1.0)

    def test_simplify_raises(self, no_shapely):
        from akimbo_geo.accessor import _op_simplify
        with pytest.raises(ImportError, match="shapely"):
            _op_simplify(self._line_arr().layout, tolerance=0.1)

    def test_to_wkb_raises(self, no_shapely):
        from akimbo_geo.convert import to_wkb
        with pytest.raises(ImportError, match="shapely"):
            to_wkb(self._line_arr())

    def test_from_wkb_raises(self, no_shapely):
        from akimbo_geo.convert import from_wkb
        wkb = ak.from_arrow(pa.array([b"\x01"], type=pa.large_binary()))
        with pytest.raises(ImportError, match="shapely"):
            from_wkb(wkb)

    def test_from_wkt_raises(self, no_shapely):
        from akimbo_geo.convert import from_wkt
        wkt = ak.from_arrow(pa.array(["LINESTRING (0 0, 1 1)"],
                                       type=pa.large_string()))
        with pytest.raises(ImportError, match="shapely"):
            from_wkt(wkt)

    def test_distance_raises(self, no_shapely):
        from akimbo_geo.accessor import _shapely_binary_float
        other = ak.from_arrow(
            pa.array([[5.0, 5.0]], type=pa.list_(pa.field("xy", pa.float64()), 2))
        )
        with pytest.raises(ImportError, match="shapely"):
            _shapely_binary_float(
                self._poly_arr().layout, other.layout, "distance"
            )

    def test_intersection_raises(self, no_shapely):
        from akimbo_geo.accessor import _shapely_binary_geom
        with pytest.raises(ImportError, match="shapely"):
            _shapely_binary_geom(
                self._poly_arr().layout, self._poly_arr().layout, "intersection"
            )
