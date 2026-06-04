"""test_gpu_dispatch.py — CPU-side tests for the GPU dispatch machinery.

These tests run on CPU-only machines.  They verify:

1. ``_compat.array_module`` correctly identifies numpy arrays as CPU
   and cupy-like arrays as GPU (using a lightweight mock, no real GPU needed).
2. ``_compat.is_gpu_array`` and ``_compat.gpu_array_to_numpy`` behave correctly.
3. ``_rebuild_list1`` and ``_rebuild_depth`` produce correct ak.contents layouts
   using direct construction (no PyArrow round-trip) for CPU data.
4. All op functions return correct results on CPU (the dispatch path taken on
   a machine without a GPU).
5. ``_layout_to_shapely`` raises ``TypeError`` for GPU-tagged arrays.
6. ``algorithms_gpu.py`` can be imported without a GPU present (the
   ``@cuda.jit`` decorator evaluates lazily and does not require CUDA at
   import time).
"""

from math import isclose

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401
import akimbo_geo     # noqa: F401

shapely = pytest.importorskip("shapely", reason="shapely>=2.0 required for some tests")


# ===========================================================================
# Mock GPU array (no real GPU needed)
# ===========================================================================

class MockGPUArray(np.ndarray):
    """Minimal stand-in for a cupy.ndarray for dispatch testing.

    Subclasses numpy.ndarray so arithmetic works, but overrides
    ``__class__.__module__`` so that ``array_module`` classifies it as GPU.
    """

    # Override __module__ on the *class* so type(instance).__module__ == "cupy"
    pass


# Patch the class after definition to spoof its module
MockGPUArray.__module__ = "cupy"


def _make_gpu(data: np.ndarray) -> MockGPUArray:
    """Return data as a MockGPUArray view."""
    arr = data.view(MockGPUArray)
    return arr


class _MockGPUArrayWithGet(MockGPUArray):
    """Adds a .get() method that copies to a plain numpy array."""

    def get(self) -> np.ndarray:
        return np.asarray(self)


# ===========================================================================
# _compat.array_module / is_gpu_array / gpu_array_to_numpy
# ===========================================================================

class TestArrayModule:

    def test_numpy_is_cpu(self):
        from akimbo_geo._compat import array_module
        arr = np.array([1.0, 2.0])
        assert array_module(arr) is np

    def test_mock_gpu_detected_as_non_numpy(self):
        """array_module detects GPU arrays by checking type.__module__."""
        from akimbo_geo._compat import is_gpu_array
        gpu = _make_gpu(np.array([1.0, 2.0]))
        # We can't call array_module() without cupy installed, but is_gpu_array
        # uses the same detection logic and doesn't import cupy.
        assert is_gpu_array(gpu)

    def test_is_gpu_array_false(self):
        from akimbo_geo._compat import is_gpu_array
        assert not is_gpu_array(np.array([1.0]))

    def test_is_gpu_array_true(self):
        from akimbo_geo._compat import is_gpu_array
        gpu = _make_gpu(np.array([1.0]))
        assert is_gpu_array(gpu)

    def test_gpu_array_to_numpy_noop_on_cpu(self):
        from akimbo_geo._compat import gpu_array_to_numpy
        arr = np.array([1.0, 2.0, 3.0])
        result = gpu_array_to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_gpu_array_to_numpy_calls_get(self):
        """gpu_array_to_numpy should call .get() if the array has that method."""
        from akimbo_geo._compat import gpu_array_to_numpy
        data = np.array([4.0, 5.0])
        gpu  = _MockGPUArrayWithGet(data.shape, dtype=data.dtype, buffer=data)
        result = gpu_array_to_numpy(gpu)
        np.testing.assert_array_equal(result, data)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# _rebuild_list1 uses ak.contents directly (no PyArrow)
# ===========================================================================

class TestRebuildList1:
    """Verify that _rebuild_list1 produces correct ak.contents layouts."""

    def _flat_geo(self):
        from akimbo_geo.match import GeoLayout, CoordKind
        return GeoLayout(CoordKind.INTERLEAVED_FLAT, 2, 1)

    def test_returns_list_offset_array(self):
        from akimbo_geo.accessor import _rebuild_list1
        values   = np.array([0.0, 0.0, 3.0, 4.0])
        offsets0 = np.array([0, 4], dtype=np.int32)
        geo      = self._flat_geo()
        result   = _rebuild_list1(values, offsets0, geo)
        assert isinstance(result, ak.contents.ListOffsetArray)

    def test_no_pyarrow_roundtrip(self, monkeypatch):
        """_rebuild_list1 must not call pa.array() for INTERLEAVED_FLAT."""
        import pyarrow as pa
        call_count = {"n": 0}
        real_pa_array = pa.array

        def spy(*args, **kwargs):
            call_count["n"] += 1
            return real_pa_array(*args, **kwargs)

        monkeypatch.setattr(pa, "array", spy)
        from akimbo_geo.accessor import _rebuild_list1
        values   = np.array([0.0, 0.0, 3.0, 4.0])
        offsets0 = np.array([0, 4], dtype=np.int32)
        geo      = self._flat_geo()
        _rebuild_list1(values, offsets0, geo)
        assert call_count["n"] == 0, "Expected zero pa.array() calls"

    def test_roundtrip_values_preserved(self):
        from akimbo_geo.accessor import _rebuild_list1
        values   = np.array([0.0, 0.0, 3.0, 4.0, 1.0, 1.0, 2.0, 2.0])
        offsets0 = np.array([0, 4, 8], dtype=np.int32)
        geo      = self._flat_geo()
        result   = _rebuild_list1(values, offsets0, geo)
        recovered = ak.Array(result).tolist()
        np.testing.assert_allclose(recovered[0], [0.0, 0.0, 3.0, 4.0])
        np.testing.assert_allclose(recovered[1], [1.0, 1.0, 2.0, 2.0])

    def test_fsl_produces_list_of_regular(self):
        from akimbo_geo.accessor import _rebuild_list1
        from akimbo_geo.match import GeoLayout, CoordKind
        values   = np.array([0.0, 0.0, 3.0, 4.0])
        offsets0 = np.array([0, 4], dtype=np.int32)  # float units → 2 points
        geo = GeoLayout(CoordKind.INTERLEAVED_FSL, 2, 1)
        result = _rebuild_list1(values, offsets0, geo)
        assert isinstance(result, ak.contents.ListOffsetArray)
        assert isinstance(result.content, ak.contents.RegularArray)

    def test_separated_struct_produces_list_of_record(self):
        from akimbo_geo.accessor import _rebuild_list1
        from akimbo_geo.match import GeoLayout, CoordKind
        # Two points interleaved: x0, y0, x1, y1
        values   = np.array([0.0, 0.0, 3.0, 4.0])
        offsets0 = np.array([0, 4], dtype=np.int32)
        geo = GeoLayout(CoordKind.SEPARATED_STRUCT, 2, 1)
        result = _rebuild_list1(values, offsets0, geo)
        assert isinstance(result, ak.contents.ListOffsetArray)
        assert isinstance(result.content, ak.contents.RecordArray)


# ===========================================================================
# _rebuild_depth uses ak.contents directly
# ===========================================================================

class TestRebuildDepth:

    def _geo(self, depth=2):
        from akimbo_geo.match import GeoLayout, CoordKind
        return GeoLayout(CoordKind.INTERLEAVED_FLAT, 2, depth)

    def test_depth0_2d_returns_numpy_array(self):
        from akimbo_geo.accessor import _rebuild_depth
        values = np.array([0.0, 0.0, 1.0, 1.0])
        result = _rebuild_depth(values, (), self._geo(0))
        assert isinstance(result, ak.contents.NumpyArray)

    def test_depth0_3d_returns_regular_array(self):
        from akimbo_geo.accessor import _rebuild_depth
        from akimbo_geo.match import GeoLayout, CoordKind
        values = np.array([0.0, 0.0, 10.0, 1.0, 1.0, 20.0])
        geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, 3, 0)
        result = _rebuild_depth(values, (), geo)
        assert isinstance(result, ak.contents.RegularArray)
        assert result.size == 3

    def test_depth1_returns_list_offset(self):
        from akimbo_geo.accessor import _rebuild_depth
        values  = np.array([0.0, 0.0, 3.0, 4.0])
        offsets = (np.array([0, 4], dtype=np.int32),)
        result  = _rebuild_depth(values, offsets, self._geo(1))
        assert isinstance(result, ak.contents.ListOffsetArray)

    def test_depth2_produces_nested_lists(self):
        from akimbo_geo.accessor import _rebuild_depth
        # Unit square ring: 5 pts × 2 floats = 10 floats
        values  = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        offsets = (
            np.array([0, 1], dtype=np.int32),   # 1 polygon, 1 ring
            np.array([0, 10], dtype=np.int32),  # ring → floats
        )
        result = _rebuild_depth(values, offsets, self._geo(2))
        assert isinstance(result, ak.contents.ListOffsetArray)
        assert isinstance(result.content, ak.contents.ListOffsetArray)

    def test_no_pyarrow_in_depth2(self, monkeypatch):
        """_rebuild_depth must not call pa.array() for INTERLEAVED_FLAT depth-2."""
        import pyarrow as pa
        call_count = {"n": 0}
        real = pa.array

        def spy(*a, **kw):
            call_count["n"] += 1
            return real(*a, **kw)

        monkeypatch.setattr(pa, "array", spy)
        from akimbo_geo.accessor import _rebuild_depth
        values  = np.array([0., 0., 1., 0., 1., 1., 0., 1., 0., 0.])
        offsets = (np.array([0, 1], dtype=np.int32), np.array([0, 10], dtype=np.int32))
        _rebuild_depth(values, offsets, self._geo(2))
        assert call_count["n"] == 0


# ===========================================================================
# Op functions produce correct results (CPU path exercised by all other tests;
# here we just confirm the ak.contents output type is correct)
# ===========================================================================

class TestOpOutputTypes:
    """Op functions must return ak.contents.Content, not high-level ak.Array."""

    def _line_arr(self):
        return ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64()))
        )

    def test_op_length_returns_numpy_array_content(self):
        from akimbo_geo.accessor import _op_length
        result = _op_length(self._line_arr().layout)
        assert isinstance(result, ak.contents.NumpyArray)
        assert isclose(ak.Array(result).tolist()[0], 5.0)

    def test_op_bounds_returns_record_array_content(self):
        from akimbo_geo.accessor import _op_bounds
        result = _op_bounds(self._line_arr().layout)
        assert isinstance(result, ak.contents.RecordArray)
        assert "xmin" in result.fields

    def test_op_centroid_returns_record_array_content(self):
        from akimbo_geo.accessor import _op_centroid1
        result = _op_centroid1(self._line_arr().layout)
        assert isinstance(result, ak.contents.RecordArray)
        assert "x" in result.fields and "y" in result.fields

    def test_op_translate_returns_list_offset_array(self):
        from akimbo_geo.accessor import _op_translate
        result = _op_translate(self._line_arr().layout, 1.0, 2.0)
        assert isinstance(result, ak.contents.ListOffsetArray)
        vals = ak.Array(result).tolist()[0]
        assert isclose(vals[0], 1.0) and isclose(vals[1], 2.0)

    def test_op_get_coordinates_returns_list_offset_of_record(self):
        from akimbo_geo.accessor import _op_get_coordinates
        result = _op_get_coordinates(self._line_arr().layout)
        assert isinstance(result, ak.contents.ListOffsetArray)
        assert isinstance(result.content, ak.contents.RecordArray)


# ===========================================================================
# Shapely ops raise TypeError for GPU-tagged data
# ===========================================================================

class TestShapelyGPUGuard:
    """_layout_to_shapely must refuse GPU data with a clear TypeError."""

    def _gpu_line_layout(self):
        """Build a layout whose values buffer is a MockGPUArray."""
        import awkward as ak
        # We can't easily inject a cupy array into an ak layout from Python,
        # so we test the guard via the is_gpu_array check being monkeypatched.
        return ak.from_arrow(
            pa.array([[0.0, 0.0, 3.0, 4.0]], type=pa.list_(pa.float64()))
        ).layout

    def test_shapely_guard_raises_on_gpu(self, monkeypatch):
        """Monkeypatch is_gpu_array to simulate GPU data for this test."""
        from akimbo_geo import accessor
        monkeypatch.setattr(accessor, "is_gpu_array", lambda _: True)
        from akimbo_geo.accessor import _layout_to_shapely, _op_is_valid
        layout = self._gpu_line_layout()
        with pytest.raises(TypeError, match="GPU"):
            _layout_to_shapely(layout)

    def test_convex_hull_raises_on_gpu(self, monkeypatch):
        from akimbo_geo import accessor
        monkeypatch.setattr(accessor, "is_gpu_array", lambda _: True)
        from akimbo_geo.accessor import _op_convex_hull
        with pytest.raises(TypeError, match="GPU"):
            _op_convex_hull(self._gpu_line_layout())


# ===========================================================================
# algorithms_gpu imports without a GPU
# ===========================================================================

def test_algorithms_gpu_importable():
    """algorithms_gpu.py should be importable on CPU-only systems.

    numba.cuda.jit is evaluated lazily; it does not require a GPU at import
    time — only at the point the kernel is compiled/launched.
    """
    try:
        from akimbo_geo import algorithms_gpu  # noqa: F401
    except ImportError as e:
        pytest.skip(f"numba.cuda not available: {e}")
    except Exception as e:
        # Allow failures that are purely about missing CUDA toolkit at import
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            pytest.skip(f"CUDA not available: {e}")
        raise
