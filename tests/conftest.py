"""conftest.py — shared fixtures for akimbo-geo tests.

All fixtures construct test data as plain nested Python lists so that no
optional dependency (spatialpandas, geopandas, shapely) is required for
the core accessor tests.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import awkward as ak

import akimbo.pandas  # noqa: F401 — registers .ak accessor


# ---------------------------------------------------------------------------
# Raw coordinate fixtures (geometry type → ak.Array)
# ---------------------------------------------------------------------------

@pytest.fixture
def line_array():
    """Three line geometries as list<float64> (1 nesting level).

    Line 0: (0,0)→(1,0)→(1,1)  length = 2.0
    Line 1: (0,0)→(3,4)        length = 5.0
    Line 2: (0,0)→(1,0)→(1,1)→(0,1)→(0,0) — closed, length = 4.0
    """
    data = [
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    ]
    return ak.from_arrow(pa.array(data, type=pa.list_(pa.float64())))


@pytest.fixture
def polygon_array():
    """Two polygon geometries as list<list<float64>> (2 nesting levels).

    Polygon 0: unit square, no holes
        exterior: (0,0),(1,0),(1,1),(0,1),(0,0)  area = 1.0
    Polygon 1: 2×2 square with a 0.5×0.5 hole
        exterior area = 4.0, hole area = 0.25  → net area = 3.75
    """
    data = [
        # Polygon 0 — unit square (CCW exterior, no holes)
        [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
        # Polygon 1 — 2×2 square (CCW exterior) with one hole (CW)
        [
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
        ],
    ]
    return ak.from_arrow(pa.array(data, type=pa.list_(pa.list_(pa.float64()))))


@pytest.fixture
def multipolygon_array():
    """One multipolygon geometry as list<list<list<float64>>> (3 nesting levels).

    MultiPolygon 0: two unit squares side-by-side
        total area = 2.0
    """
    data = [
        [
            [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
            [[2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 0.0]],
        ],
    ]
    return ak.from_arrow(
        pa.array(data, type=pa.list_(pa.list_(pa.list_(pa.float64()))))
    )


@pytest.fixture
def line_series(line_array):
    """Pandas Series wrapping line geometries."""
    return pd.Series(
        ak.to_arrow(line_array),
        dtype=pd.ArrowDtype(pa.list_(pa.float64())),
    )


@pytest.fixture
def polygon_series(polygon_array):
    """Pandas Series wrapping polygon geometries."""
    return pd.Series(
        ak.to_arrow(polygon_array),
        dtype=pd.ArrowDtype(pa.list_(pa.list_(pa.float64()))),
    )
