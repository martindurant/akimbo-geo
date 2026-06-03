"""akimbo-geo: geometry operations accessor for akimbo.

Importing this module registers the ``.geo`` sub-accessor on
``EagerAccessor`` and ``LazyAccessor``, making it available as
``series.ak.geo`` on any akimbo-enabled dataframe backend.
"""

from akimbo_geo.accessor import GeoAccessor  # noqa: F401 — side-effect import

__all__ = ["GeoAccessor"]
