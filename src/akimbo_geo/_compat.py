"""_compat.py — optional-dependency guards and backend-dispatch helpers.

Centralises:
- ImportError messages for optional dependencies (shapely, spatialpandas, geopandas)
- Array-backend detection so that numba-CPU and CUDA-GPU kernels can be
  selected at runtime based on the memory location of the data.
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# Array-backend detection
# ===========================================================================

def array_module(arr):
    """Return the array module (``numpy`` or ``cupy``) for *arr*.

    This is the single point of dispatch between CPU and GPU code paths.
    All op functions call this on the ``values`` array returned by
    ``extract_offsets_and_values`` to decide which kernel set to use.

    Returns
    -------
    numpy or cupy module
        ``numpy`` for CPU arrays (default).
        ``cupy`` when the array lives in GPU device memory.
    """
    mod = type(arr).__module__
    if mod == "cupy" or mod.startswith("cupy."):
        try:
            import cupy
            return cupy
        except ImportError:
            # cupy is not installed but we detected a cupy-module-typed array.
            # This shouldn't happen in practice (you can't have a cupy array
            # without cupy installed), but guard it defensively.
            raise ImportError(
                "Detected a GPU array (module='cupy') but cupy is not "
                "installed.  Install it with: pip install 'akimbo-geo[gpu]'"
            )
    return np


def is_gpu_array(arr) -> bool:
    """Return True if *arr* lives in GPU device memory.

    Detection is based solely on the type's module name — does not import
    cupy and therefore works on CPU-only machines.
    """
    mod = type(arr).__module__
    return mod == "cupy" or mod.startswith("cupy.")


def gpu_array_to_numpy(arr) -> np.ndarray:
    """Copy a GPU array to CPU.  No-op if already on CPU."""
    if is_gpu_array(arr):
        return arr.get()      # cupy → numpy
    return np.asarray(arr)


# ===========================================================================
# Optional-dependency guards
# ===========================================================================

def require_shapely():
    """Return the ``shapely`` module, or raise a clear ImportError.

    shapely >= 2.0 is an *optional* dependency of akimbo-geo.  It is only
    required for:

    - Topology-based predicates: ``is_valid``, ``is_simple``, ``is_empty``, …
    - Constructive operations: ``buffer``, ``convex_hull``, ``simplify``, …
    - Binary predicates: ``contains``, ``intersects``, ``distance``, …
    - Set-theoretic operations: ``intersection``, ``union``, …
    - WKB / WKT serialisation: ``from_wkb``, ``to_wkb``, ``from_wkt``, ``to_wkt``
    - Framework bridges: ``from_geopandas``, ``to_geopandas``

    All pure-numba operations (``area``, ``length``, ``bounds``, ``centroid``,
    ``translate``, ``segmentize``, ``orient_polygons``, …) work without it.

    Install with::

        pip install "akimbo-geo[shapely]"
    """
    try:
        import shapely
        return shapely
    except ImportError as exc:
        raise ImportError(
            "shapely >= 2.0 is required for this operation but is not installed.\n"
            "Install it with:  pip install 'akimbo-geo[shapely]'\n"
            "Pure-numba operations (area, length, bounds, translate, …) "
            "do not need shapely."
        ) from exc


def require_spatialpandas():
    """Return the ``spatialpandas`` module, or raise a clear ImportError."""
    try:
        import spatialpandas
        return spatialpandas
    except ImportError as exc:
        raise ImportError(
            "spatialpandas is required for this operation but is not installed.\n"
            "Install it with:  pip install 'akimbo-geo[spatialpandas]'"
        ) from exc


def require_geopandas():
    """Return the ``geopandas`` module, or raise a clear ImportError."""
    try:
        import geopandas
        return geopandas
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for this operation but is not installed.\n"
            "Install it with:  pip install 'akimbo-geo[geopandas]'"
        ) from exc

