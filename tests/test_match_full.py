"""test_match_full.py — full coverage of match.py.

Tests:
- flat_offsets_into_values (multi-level offset flattening)
- _extract_coord_values with INTERLEAVED_FLAT path explicitly
- extract_interleaved / extract_offsets_and_values error path (non-geometry layout)
- _scale_offsets for all CoordKind values
- match_point
- GeoLayout depth counting for all depths (0-3)
- _unwrap on non-UnmaskedArray (pass-through)
- _is_float_numpy on non-leaf and on integer dtype
"""

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak


# ===========================================================================
# flat_offsets_into_values
# ===========================================================================

class TestFlatOffsetsIntoValues:

    def test_single_level(self):
        from akimbo_geo.match import flat_offsets_into_values
        off0 = np.array([0, 4, 8], dtype=np.int32)
        result = flat_offsets_into_values((off0,))
        np.testing.assert_array_equal(result, off0)

    def test_two_levels(self):
        from akimbo_geo.match import flat_offsets_into_values
        # off0 → off1 → values
        # off0 = [0, 2, 3]  (outer: polygon 0 has 2 rings, polygon 1 has 1 ring)
        # off1 = [0, 5, 10, 15]  (ring → coord offsets)
        off0 = np.array([0, 2, 3], dtype=np.int32)
        off1 = np.array([0, 5, 10, 15], dtype=np.int32)
        result = flat_offsets_into_values((off0, off1))
        # polygon 0: off1[0]=0, off1[2]=10; polygon 1: off1[2]=10, off1[3]=15
        np.testing.assert_array_equal(result, [0, 10, 15])

    def test_three_levels(self):
        from akimbo_geo.match import flat_offsets_into_values
        off0 = np.array([0, 2], dtype=np.int32)
        off1 = np.array([0, 1, 2], dtype=np.int32)
        off2 = np.array([0, 10, 20], dtype=np.int32)
        result = flat_offsets_into_values((off0, off1, off2))
        # off1[off0] = off1[[0,2]] = [0, 2]; off2[[0,2]] = [0, 20]
        np.testing.assert_array_equal(result, [0, 20])


# ===========================================================================
# _extract_coord_values — INTERLEAVED_FLAT path
# ===========================================================================

class TestExtractCoordValuesFlatPath:

    def test_flat_list_extract(self):
        """INTERLEAVED_FLAT: the flat NumpyArray is passed through as-is."""
        from akimbo_geo.match import _extract_coord_values, CoordKind
        flat = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]],
                                     type=pa.list_(pa.float64())))
        # Get the NumpyArray leaf
        layout = arr.layout.content.content   # UnmaskedArray → NumpyArray
        result = _extract_coord_values(layout, CoordKind.INTERLEAVED_FLAT, 2)
        np.testing.assert_allclose(result, flat)

    def test_unknown_coord_kind_raises(self):
        """An unrecognised CoordKind raises ValueError."""
        from akimbo_geo.match import _extract_coord_values, CoordKind
        import enum
        # Construct a fake CoordKind value
        FakeKind = enum.Enum("FakeKind", {"UNKNOWN": 99})
        arr = ak.from_arrow(pa.array([[0.0, 1.0]], type=pa.list_(pa.float64())))
        layout = arr.layout.content.content
        with pytest.raises((ValueError, KeyError)):
            _extract_coord_values(layout, FakeKind.UNKNOWN, 2)


# ===========================================================================
# extract_interleaved — error path
# ===========================================================================

class TestExtractInterleavedError:

    def test_non_geometry_raises(self):
        """Passing a plain float array raises ValueError."""
        from akimbo_geo.match import extract_interleaved
        arr = ak.from_arrow(pa.array([1.0, 2.0, 3.0], type=pa.float64()))
        with pytest.raises(ValueError, match="not a recognised geometry"):
            extract_interleaved(arr.layout)


# ===========================================================================
# _scale_offsets
# ===========================================================================

class TestScaleOffsets:

    def test_flat_no_scaling(self):
        """INTERLEAVED_FLAT: offsets returned unchanged."""
        from akimbo_geo.accessor import _scale_offsets
        from akimbo_geo.match import CoordKind, GeoLayout
        off = (np.array([0, 4, 8], dtype=np.int32),)
        geo = GeoLayout(CoordKind.INTERLEAVED_FLAT, 2, 1)
        result = _scale_offsets(off, geo)
        np.testing.assert_array_equal(result[0], off[0])

    def test_fsl_scaling(self):
        """INTERLEAVED_FSL: offsets multiplied by n_dims."""
        from akimbo_geo.accessor import _scale_offsets
        from akimbo_geo.match import CoordKind, GeoLayout
        off = (np.array([0, 3, 5], dtype=np.int32),)
        geo = GeoLayout(CoordKind.INTERLEAVED_FSL, 2, 1)
        result = _scale_offsets(off, geo)
        np.testing.assert_array_equal(result[0], [0, 6, 10])

    def test_struct_scaling(self):
        """SEPARATED_STRUCT: offsets multiplied by n_dims (3 for XYZ)."""
        from akimbo_geo.accessor import _scale_offsets
        from akimbo_geo.match import CoordKind, GeoLayout
        off = (np.array([0, 2], dtype=np.int32),)
        geo = GeoLayout(CoordKind.SEPARATED_STRUCT, 3, 1)
        result = _scale_offsets(off, geo)
        np.testing.assert_array_equal(result[0], [0, 6])


# ===========================================================================
# match_point
# ===========================================================================

class TestMatchPoint:

    def test_fsl_point(self):
        from akimbo_geo.match import match_point
        pt = ak.from_arrow(pa.array(
            [[0.0, 0.0], [1.0, 2.0]],
            type=pa.list_(pa.field("xy", pa.float64()), 2)
        ))
        assert match_point(pt.layout)

    def test_struct_point(self):
        from akimbo_geo.match import match_point
        coord_type = pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
        pts = pa.StructArray.from_arrays(
            [pa.array([0.0, 1.0]), pa.array([0.0, 2.0])], names=["x", "y"]
        )
        arr = ak.from_arrow(pts)
        assert match_point(arr.layout)

    def test_line_not_point(self):
        from akimbo_geo.match import match_point
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 1.0, 0.0]], type=pa.list_(pa.float64())))
        # list<float> is depth-1 (Line), not depth-0 (Point)
        assert not match_point(arr.layout)


# ===========================================================================
# _unwrap pass-through
# ===========================================================================

class TestUnwrap:

    def test_non_unmasked_passthrough(self):
        """_unwrap on a layout that is not UnmaskedArray returns it unchanged."""
        from akimbo_geo.match import _unwrap
        arr = ak.from_arrow(pa.array([1.0, 2.0, 3.0], type=pa.float64()))
        layout = arr.layout
        result = _unwrap(layout)
        assert result is layout

    def test_unmasked_strips(self):
        """_unwrap on an UnmaskedArray returns its content."""
        from akimbo_geo.match import _unwrap
        arr = ak.from_arrow(pa.array([[0.0, 1.0]], type=pa.list_(pa.float64())))
        # layout.content is an UnmaskedArray wrapping a NumpyArray
        unmasked = arr.layout.content
        assert isinstance(unmasked, ak.contents.UnmaskedArray)
        result = _unwrap(unmasked)
        assert not isinstance(result, ak.contents.UnmaskedArray)


# ===========================================================================
# _is_float_numpy edge cases
# ===========================================================================

class TestIsFloatNumpy:

    def test_integer_leaf_matches(self):
        """Integer NumpyArray also matches (kind 'i')."""
        from akimbo_geo.match import _is_float_numpy
        arr = ak.from_arrow(pa.array([0, 1, 2], type=pa.int32()))
        assert _is_float_numpy(arr.layout)

    def test_string_does_not_match(self):
        from akimbo_geo.match import _is_float_numpy
        arr = ak.from_arrow(pa.array(["hello"]))
        assert not _is_float_numpy(arr.layout)

    def test_list_does_not_match(self):
        """A list array is not a leaf."""
        from akimbo_geo.match import _is_float_numpy
        arr = ak.from_arrow(pa.array([[1.0, 2.0]], type=pa.list_(pa.float64())))
        assert not _is_float_numpy(arr.layout)


# ===========================================================================
# GeoLayout depth counting for all depths
# ===========================================================================

class TestGeoLayoutDepths:

    def test_depth_0_fsl(self):
        from akimbo_geo.match import _geo_layout_of
        pt = ak.from_arrow(pa.array(
            [[0.0, 0.0], [1.0, 2.0]],
            type=pa.list_(pa.field("xy", pa.float64()), 2)
        ))
        geo = _geo_layout_of(pt.layout)
        assert geo is not None and geo.list_depth == 0

    def test_depth_1_flat(self):
        from akimbo_geo.match import _geo_layout_of
        arr = ak.from_arrow(pa.array([[0.0, 0.0, 1.0, 0.0]], type=pa.list_(pa.float64())))
        geo = _geo_layout_of(arr.layout)
        assert geo is not None and geo.list_depth == 1

    def test_depth_2_flat(self):
        from akimbo_geo.match import _geo_layout_of
        arr = ak.from_arrow(pa.array(
            [[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]],
            type=pa.list_(pa.list_(pa.float64()))
        ))
        geo = _geo_layout_of(arr.layout)
        assert geo is not None and geo.list_depth == 2

    def test_depth_3_flat(self):
        from akimbo_geo.match import _geo_layout_of
        arr = ak.from_arrow(pa.array(
            [[[[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]]],
            type=pa.list_(pa.list_(pa.list_(pa.float64())))
        ))
        geo = _geo_layout_of(arr.layout)
        assert geo is not None and geo.list_depth == 3
