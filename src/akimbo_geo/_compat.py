"""_compat.py — optional-dependency guards.

Centralises the ImportError messages for optional dependencies so that
every function that needs shapely, spatialpandas, or geopandas raises a
consistent, actionable error message with the correct install command.
"""

from __future__ import annotations


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
