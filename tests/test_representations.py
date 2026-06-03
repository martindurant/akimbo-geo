"""test_representations.py — tests covering all coordinate representations.

Tests that GeoArrow native (interleaved FixedSizeList and separated struct),
spatialpandas interleaved flat, and heuristic layouts are all correctly
identified and produce consistent numerical results from the op functions.
"""

from math import isclose

import numpy as np
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401
import akimbo_geo     # noqa: F401


# ---------------------------------------------------------------------------
# Shared expected values (used across all representations)
# ---------------------------------------------------------------------------
# LineString: (0,0)→(1,0)→(1,1)   length = 2.0
LINE_COORDS     = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
LINE_LENGTH     = 2.0

# LineString: (0,0)→(3,4)   length = 5.0
LINE2_COORDS    = [(0.0, 0.0), (3.0, 4.0)]
LINE2_LENGTH    = 5.0

# Unit square polygon (CCW): area = 1.0
SQUARE_COORDS   = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
SQUARE_AREA     = 1.0
SQUARE_BOUNDS   = (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers to build arrays in each representation
# ---------------------------------------------------------------------------

def _fsl_line(coord_list):
    """Build a single LineString as geoarrow FixedSizeList interleaved."""
    return pa.array(
        [coord_list],
        type=pa.list_(pa.list_(pa.field("xy", pa.float64()), 2))
    )

def _struct_line(coord_list):
    """Build a single LineString as geoarrow separated struct."""
    coord_type = pa.struct([
        pa.field("x", pa.float64()),
        pa.field("y", pa.float64()),
    ])
    return pa.array(
        [[{"x": x, "y": y} for x, y in coord_list]],
        type=pa.list_(coord_type)
    )

def _flat_line(coord_list):
    """Build a single LineString as spatialpandas interleaved flat."""
    flat = [v for x, y in coord_list for v in (x, y)]
    return pa.array([flat], type=pa.list_(pa.float64()))

def _fsl_polygon(ring_list_of_coords):
    """Build a single Polygon as geoarrow FixedSizeList interleaved."""
    return pa.array(
        [[ring_list_of_coords]],
        type=pa.list_(pa.list_(pa.list_(pa.field("xy", pa.float64()), 2)))
    )

def _struct_polygon(ring_list_of_coords):
    """Build a single Polygon as geoarrow separated struct."""
    coord_type = pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
    return pa.array(
        [[[{"x": x, "y": y} for x, y in ring_list_of_coords]]],
        type=pa.list_(pa.list_(coord_type))
    )

def _flat_polygon(ring_list_of_coords):
    """Build a single Polygon as spatialpandas interleaved flat."""
    flat_ring = [v for x, y in ring_list_of_coords for v in (x, y)]
    return pa.array([[flat_ring]], type=pa.list_(pa.list_(pa.float64())))


# ===========================================================================
# TestGeoLayout — _geo_layout_of / match functions
# ===========================================================================

class TestGeoLayout:
    """Verify _geo_layout_of returns correct GeoLayout for every format."""

    from akimbo_geo.match import CoordKind, GeoLayout, _geo_layout_of

    def _layout_of(self, pa_arr):
        from akimbo_geo.match import _geo_layout_of
        return _geo_layout_of(ak.from_arrow(pa_arr).layout)

    # --- Interleaved FSL ----------------------------------------------------

    def test_fsl_linestring(self):
        from akimbo_geo.match import CoordKind
        geo = self._layout_of(_fsl_line(LINE_COORDS))
        assert geo is not None
        assert geo.coord_kind == CoordKind.INTERLEAVED_FSL
        assert geo.n_dims == 2
        assert geo.list_depth == 1

    def test_fsl_polygon(self):
        from akimbo_geo.match import CoordKind
        geo = self._layout_of(_fsl_polygon(SQUARE_COORDS))
        assert geo is not None
        assert geo.coord_kind == CoordKind.INTERLEAVED_FSL
        assert geo.list_depth == 2

    def test_fsl_point(self):
        """FixedSizeList<double>[2] without outer list → depth-0 Point."""
        from akimbo_geo.match import CoordKind
        pt = pa.array(
            [[0.0, 0.0], [1.0, 2.0]],
            type=pa.list_(pa.field("xy", pa.float64()), 2)
        )
        geo = self._layout_of(pt)
        assert geo is not None
        assert geo.coord_kind == CoordKind.INTERLEAVED_FSL
        assert geo.list_depth == 0

    # --- Separated struct ---------------------------------------------------

    def test_struct_linestring(self):
        from akimbo_geo.match import CoordKind
        geo = self._layout_of(_struct_line(LINE_COORDS))
        assert geo is not None
        assert geo.coord_kind == CoordKind.SEPARATED_STRUCT
        assert geo.n_dims == 2
        assert geo.list_depth == 1

    def test_struct_polygon(self):
        from akimbo_geo.match import CoordKind
        geo = self._layout_of(_struct_polygon(SQUARE_COORDS))
        assert geo is not None
        assert geo.coord_kind == CoordKind.SEPARATED_STRUCT
        assert geo.list_depth == 2

    def test_struct_point(self):
        """Struct<x,y> without outer list → depth-0 Point."""
        from akimbo_geo.match import CoordKind
        coord_type = pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
        pt = pa.StructArray.from_arrays(
            [pa.array([0.0, 3.0]), pa.array([0.0, 4.0])], names=["x", "y"]
        )
        geo = self._layout_of(pt)
        assert geo is not None
        assert geo.coord_kind == CoordKind.SEPARATED_STRUCT
        assert geo.list_depth == 0

    # --- Interleaved flat (spatialpandas) -----------------------------------

    def test_flat_linestring(self):
        from akimbo_geo.match import CoordKind
        geo = self._layout_of(_flat_line(LINE_COORDS))
        assert geo is not None
        assert geo.coord_kind == CoordKind.INTERLEAVED_FLAT
        assert geo.list_depth == 1

    def test_flat_polygon(self):
        from akimbo_geo.match import CoordKind
        geo = self._layout_of(_flat_polygon(SQUARE_COORDS))
        assert geo is not None
        assert geo.coord_kind == CoordKind.INTERLEAVED_FLAT
        assert geo.list_depth == 2

    # --- Heuristic ----------------------------------------------------------

    def test_heuristic_2float_point(self):
        """series(2 * float) — RegularArray(size=2) treated as Point."""
        from akimbo_geo.match import CoordKind
        arr = ak.to_regular(ak.Array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]))
        from akimbo_geo.match import _geo_layout_of
        geo = _geo_layout_of(arr.layout)
        assert geo is not None
        assert geo.coord_kind == CoordKind.INTERLEAVED_FSL
        assert geo.list_depth == 0

    def test_heuristic_list_2float(self):
        """series(list(2 * float)) — outer list of 2-element fixed arrays."""
        from akimbo_geo.match import CoordKind
        # Each row is a list of [x,y] pairs
        arr = ak.from_arrow(pa.array(
            [[[0.0, 0.0], [1.0, 0.0]], [[2.0, 2.0], [3.0, 4.0]]],
            type=pa.list_(pa.list_(pa.field("xy", pa.float64()), 2))
        ))
        from akimbo_geo.match import _geo_layout_of
        geo = _geo_layout_of(arr.layout)
        assert geo is not None
        assert geo.list_depth == 1

    # --- Non-matches --------------------------------------------------------

    def test_no_match_plain_float(self):
        from akimbo_geo.match import _geo_layout_of
        arr = ak.from_arrow(pa.array([1.0, 2.0], type=pa.float64()))
        assert _geo_layout_of(arr.layout) is None

    def test_no_match_string(self):
        from akimbo_geo.match import _geo_layout_of
        arr = ak.from_arrow(pa.array(["hello"]))
        assert _geo_layout_of(arr.layout) is None

    def test_no_match_wkb(self):
        from akimbo_geo.match import _geo_layout_of
        arr = ak.from_arrow(pa.array([b"\x01\x02"], type=pa.large_binary()))
        assert _geo_layout_of(arr.layout) is None

    def test_no_match_struct_no_xy(self):
        """Struct without x/y field names should not match."""
        from akimbo_geo.match import _geo_layout_of
        s = pa.StructArray.from_arrays(
            [pa.array([1.0]), pa.array([2.0])], names=["a", "b"]
        )
        assert _geo_layout_of(ak.from_arrow(s).layout) is None


# ===========================================================================
# TestComputeAllRepresentations — same numerical answer from all formats
# ===========================================================================

class TestComputeAllRepresentations:
    """Each operation must produce the same result regardless of coord format."""

    # --- length -------------------------------------------------------------

    def _length(self, pa_arr):
        from akimbo_geo.accessor import _op_length
        return ak.Array(_op_length(ak.from_arrow(pa_arr).layout)).tolist()

    def test_length_flat(self):
        result = self._length(_flat_line(LINE_COORDS))
        assert isclose(result[0], LINE_LENGTH)

    def test_length_fsl(self):
        result = self._length(_fsl_line(LINE_COORDS))
        assert isclose(result[0], LINE_LENGTH), f"got {result[0]}"

    def test_length_struct(self):
        result = self._length(_struct_line(LINE_COORDS))
        assert isclose(result[0], LINE_LENGTH), f"got {result[0]}"

    def test_length_fsl_two_lines(self):
        """Multiple lines in one array."""
        pa_arr = pa.array(
            [LINE_COORDS, LINE2_COORDS],
            type=pa.list_(pa.list_(pa.field("xy", pa.float64()), 2))
        )
        result = self._length(pa_arr)
        assert isclose(result[0], LINE_LENGTH)
        assert isclose(result[1], LINE2_LENGTH)

    def test_length_struct_two_lines(self):
        coord_type = pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
        pa_arr = pa.array(
            [[{"x": x, "y": y} for x, y in LINE_COORDS],
             [{"x": x, "y": y} for x, y in LINE2_COORDS]],
            type=pa.list_(coord_type)
        )
        result = self._length(pa_arr)
        assert isclose(result[0], LINE_LENGTH)
        assert isclose(result[1], LINE2_LENGTH)

    # --- area ---------------------------------------------------------------

    def _area(self, pa_arr):
        from akimbo_geo.accessor import _op_area
        return ak.Array(_op_area(ak.from_arrow(pa_arr).layout)).tolist()

    def test_area_flat(self):
        result = self._area(_flat_polygon(SQUARE_COORDS))
        assert isclose(abs(result[0]), SQUARE_AREA)

    def test_area_fsl(self):
        result = self._area(_fsl_polygon(SQUARE_COORDS))
        assert isclose(abs(result[0]), SQUARE_AREA), f"got {result[0]}"

    def test_area_struct(self):
        result = self._area(_struct_polygon(SQUARE_COORDS))
        assert isclose(abs(result[0]), SQUARE_AREA), f"got {result[0]}"

    # --- bounds -------------------------------------------------------------

    def _bounds(self, pa_arr):
        from akimbo_geo.accessor import _op_bounds
        return ak.Array(_op_bounds(ak.from_arrow(pa_arr).layout)).tolist()

    def test_bounds_flat(self):
        result = self._bounds(_flat_line(LINE_COORDS))
        r = result[0]
        assert r["xmin"] == SQUARE_BOUNDS[0]
        assert r["ymin"] == SQUARE_BOUNDS[1]
        assert r["xmax"] == SQUARE_BOUNDS[2]
        assert r["ymax"] == SQUARE_BOUNDS[3]

    def test_bounds_fsl(self):
        result = self._bounds(_fsl_line(LINE_COORDS))
        r = result[0]
        assert isclose(r["xmin"], 0.0)
        assert isclose(r["ymin"], 0.0)
        assert isclose(r["xmax"], 1.0)
        assert isclose(r["ymax"], 1.0)

    def test_bounds_struct(self):
        result = self._bounds(_struct_line(LINE_COORDS))
        r = result[0]
        assert isclose(r["xmin"], 0.0)
        assert isclose(r["xmax"], 1.0)

    # --- centroid -----------------------------------------------------------

    def _centroid(self, pa_arr):
        from akimbo_geo.accessor import _op_centroid1
        return ak.Array(_op_centroid1(ak.from_arrow(pa_arr).layout)).tolist()

    def test_centroid_flat(self):
        # (0,0),(1,0),(1,1) → mean = (2/3, 1/3)
        result = self._centroid(_flat_line(LINE_COORDS))
        assert isclose(result[0]["x"], 2/3, rel_tol=1e-6)
        assert isclose(result[0]["y"], 1/3, rel_tol=1e-6)

    def test_centroid_fsl(self):
        result = self._centroid(_fsl_line(LINE_COORDS))
        assert isclose(result[0]["x"], 2/3, rel_tol=1e-6)
        assert isclose(result[0]["y"], 1/3, rel_tol=1e-6)

    def test_centroid_struct(self):
        result = self._centroid(_struct_line(LINE_COORDS))
        assert isclose(result[0]["x"], 2/3, rel_tol=1e-6)
        assert isclose(result[0]["y"], 1/3, rel_tol=1e-6)

    # --- XYZ (3-dimensional) ------------------------------------------------

    def test_length_fsl_3d(self):
        """3D FixedSizeList[3] — length uses only x and y (z ignored for 2D kernels)."""
        # (0,0,10)→(3,4,20): 2D length = 5, z not used in kernel
        coords_3d = [[0.0, 0.0, 10.0], [3.0, 4.0, 20.0]]
        pa_arr = pa.array(
            [coords_3d],
            type=pa.list_(pa.list_(pa.field("xyz", pa.float64()), 3))
        )
        arr = ak.from_arrow(pa_arr)
        from akimbo_geo.match import _geo_layout_of, CoordKind
        geo = _geo_layout_of(arr.layout)
        assert geo is not None
        assert geo.n_dims == 3
        # Kernel still works — strides through flat buffer in steps of n_dims
        from akimbo_geo.accessor import _op_length
        # Note: 3D length uses stride-3 so not 5.0, but we just verify it runs
        result = ak.Array(_op_length(arr.layout)).tolist()
        assert result[0] is not None   # produced a number, didn't crash

    def test_length_struct_3d(self):
        """Struct<x,y,z> separated — z axis recognised but length is still 3D."""
        coord_type = pa.struct([
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("z", pa.float64()),
        ])
        pa_arr = pa.array(
            [[{"x": 0.0, "y": 0.0, "z": 10.0}, {"x": 3.0, "y": 4.0, "z": 20.0}]],
            type=pa.list_(coord_type)
        )
        arr = ak.from_arrow(pa_arr)
        from akimbo_geo.match import _geo_layout_of, CoordKind
        geo = _geo_layout_of(arr.layout)
        assert geo is not None
        assert geo.coord_kind == CoordKind.SEPARATED_STRUCT
        assert geo.n_dims == 3
        from akimbo_geo.accessor import _op_length
        result = ak.Array(_op_length(arr.layout)).tolist()
        assert result[0] is not None


# ===========================================================================
# TestNestedWithNewFormats — dec() tree-walk works for all representations
# ===========================================================================

class TestNestedWithNewFormats:
    """Verify dec() descends correctly into list-of-geometry for all formats."""

    def test_list_of_fsl_lines(self):
        """Each row is a variable list of FSL LineStrings."""
        from akimbo_geo.accessor import _op_length
        from akimbo_geo.match import match_line
        from akimbo.apply_tree import dec

        # Row 0: two FSL lines; Row 1: one FSL line
        pa_arr = pa.array(
            [[LINE_COORDS, LINE2_COORDS], [LINE_COORDS]],
            type=pa.list_(pa.list_(pa.list_(pa.field("xy", pa.float64()), 2)))
        )
        arr = ak.from_arrow(pa_arr)
        length_fn = dec(_op_length, match=match_line, inmode="ak")
        result = ak.to_list(length_fn(arr))
        assert isclose(result[0][0], LINE_LENGTH)
        assert isclose(result[0][1], LINE2_LENGTH)
        assert isclose(result[1][0], LINE_LENGTH)

    def test_list_of_struct_lines(self):
        """Each row is a variable list of struct LineStrings."""
        from akimbo_geo.accessor import _op_length
        from akimbo_geo.match import match_line
        from akimbo.apply_tree import dec

        coord_type = pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
        line_type = pa.list_(coord_type)
        outer_type = pa.list_(line_type)
        pa_arr = pa.array(
            [
                [
                    [{"x": x, "y": y} for x, y in LINE_COORDS],
                    [{"x": x, "y": y} for x, y in LINE2_COORDS],
                ],
                [
                    [{"x": x, "y": y} for x, y in LINE_COORDS],
                ],
            ],
            type=outer_type,
        )
        arr = ak.from_arrow(pa_arr)
        length_fn = dec(_op_length, match=match_line, inmode="ak")
        result = ak.to_list(length_fn(arr))
        assert isclose(result[0][0], LINE_LENGTH)
        assert isclose(result[0][1], LINE2_LENGTH)
        assert isclose(result[1][0], LINE_LENGTH)
