# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MTBS Wildfire Perimeters (polars version)
#
# This is the polars equivalent of ``04_mtbs_wildfire_perimeters.py``.
# The analysis is identical; the DataFrame library is polars.
#
# **Dataset:** MTBS fire perimeters 1984–2026:
# - `data/mtbs_fires_flat.parquet` — one row per fire event, WKB geometry
# - `data/mtbs_fires_grouped.parquet` — one row per fire name,
#   `perimeters: List(Binary)` — a variable-length list of WKB per fire
#
# ## Polars-specific notes
#
# ``akimbo.polars`` registers the ``.ak`` namespace on ``pl.Series`` and
# ``pl.DataFrame`` via the same ``EagerAccessor`` base class used by pandas.
# The ``.ak.geo`` sub-accessor is available identically.
#
# **Three actual differences from the pandas version:**
#
# 1. **DataFrame mutation** — polars DataFrames are immutable;
#    use ``df.with_columns(...)`` instead of ``df["col"] = ...``.
#
# 2. **Struct field access** — ``.ak.geo.bounds()`` returns a ``pl.Struct``
#    Series; fields are accessed with ``.struct.field("name")`` instead of
#    unpacking into a DataFrame.  This is already idiomatic polars.
#
# All ``.ak`` and ``.ak.geo`` operations are otherwise identical between
# the two backends.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pyarrow as pa
import awkward as ak
import polars as pl

import akimbo.polars     # registers .ak on pl.Series / pl.DataFrame
import akimbo_geo        # registers .ak.geo

# %% [markdown]
# ## Part 1: Flat table — one geometry per row
#
# ### Load and decode
#
# The file stores geometry as WKB binary (``pl.Binary`` in polars).
# ``series.ak.geo.from_wkb()`` decodes the whole column in one vectorised
# call — no Python loops, no explicit conversion step.

# %%
raw = pl.read_parquet("data/mtbs_fires_flat.parquet")
print(f"{len(raw):,} fire events  {raw['year'].min()}–{raw['year'].max()}")
print("Schema before decode:", raw.schema)

# %%
# .ak.geo.from_wkb() is wired via dec() with match_wkb.
# It accepts the Binary column directly and returns a geometry Series.
df = raw.with_columns(
    raw["geometry"].ak.geo.from_wkb().alias("geometry")
)
print("geometry dtype after decode:", df["geometry"].dtype)
df.head(3)

# %% [markdown]
# ### Area and perimeter  (pure numba)

# %%
df = df.with_columns(
    df["geometry"].ak.geo.area3().abs().alias("area_deg2"),
    df["geometry"].ak.geo.length3().alias("perimeter_deg"),
)

nonzero = df.filter(pl.col("acres") > 0)
corr = np.corrcoef(
    np.log1p(nonzero["area_deg2"].to_numpy()),
    np.log1p(nonzero["acres"].to_numpy()),
)[0, 1]
print(f"log-log correlation (shoelace deg² vs reported acres): {corr:.4f}")

# %%
print("Largest fires by shoelace area:")
df.sort("area_deg2", descending=True).select(
    "fire_name", "fire_type", "year", "acres", "area_deg2"
).head(8)

# %% [markdown]
# ### Shape compactness  (polars expression API)

# %%
df = df.with_columns(
    pl.when(pl.col("perimeter_deg") > 0)
      .then(4 * np.pi * pl.col("area_deg2") / pl.col("perimeter_deg") ** 2)
      .otherwise(None)
      .alias("compactness")
)
print("Compactness by fire type:")
df.group_by("fire_type").agg(
    pl.col("compactness").median().alias("median"),
    pl.col("compactness").quantile(0.25).alias("q25"),
    pl.col("compactness").quantile(0.75).alias("q75"),
).sort("median", descending=True)

# %% [markdown]
# ### MultiPolygon patch count  (pure akimbo)

# %%
df = df.with_columns(
    df["geometry"].ak.num(axis=1).alias("n_patches")
)
print("Fire events with the most disjoint burned patches:")
df.sort("n_patches", descending=True).select(
    "fire_name", "fire_type", "year", "acres", "n_patches"
).head(8)

# %% [markdown]
# ### Bounding box  (pure numba)
#
# `.ak.geo.bounds()` returns a ``pl.Struct`` Series.
# Polars struct fields are accessed with ``.struct.field("name")`` —
# no need to unpack into separate columns unless desired.

# %%
bounds = df["geometry"].ak.geo.bounds()
print("bounds dtype:", bounds.dtype)

# Access struct fields directly in polars expressions
df = df.with_columns(
    bounds.struct.field("xmin").alias("xmin"),
    bounds.struct.field("ymin").alias("ymin"),
    bounds.struct.field("xmax").alias("xmax"),
    bounds.struct.field("ymax").alias("ymax"),
    (bounds.struct.field("xmax") - bounds.struct.field("xmin")).alias("width_deg"),
)
print("Geographic extents:")
print(df.select("xmin", "ymin", "xmax", "ymax").describe())

# %% [markdown]
# ### Annual burn statistics  (polars group_by)

# %%
annual = (
    df.filter(pl.col("year").is_between(1990, 2024))
      .group_by("year")
      .agg(
          pl.len().alias("n_fires"),
          pl.col("acres").sum().alias("total_acres"),
      )
      .sort("year")
)
# 5-year bands
print("Annual burn statistics (5-year bands):")
(
    annual
    .with_columns(((pl.col("year") // 5) * 5).alias("band"))
    .group_by("band")
    .agg(pl.col("n_fires").sum(), pl.col("total_acres").sum().round(0))
    .sort("band")
)

# %% [markdown]
# ### Validity check  (shapely / GEOS)

# %%
df = df.with_columns(df["geometry"].ak.geo.is_valid().alias("is_valid"))
print(f"Valid:   {df['is_valid'].sum():,}")
print(f"Invalid: {(~df['is_valid']).sum()}")

# %% [markdown]
# ## Part 2: Grouped table — `list[Polygon]` per fire name
#
# ### Load

# %%
gdf = pl.read_parquet("data/mtbs_fires_grouped.parquet")
print(f"{len(gdf):,} distinct fire names")
print("Schema:", gdf.schema)

# %% [markdown]
# ### Decode `List(Binary)` → `List(Polygon)`
#
# ``perimeters`` is ``List(Binary)`` — a list of WKB bytes per fire.
# We flatten to a 1-D binary array, decode the whole batch with
# ``from_wkb``, then re-nest using the original event counts.
#
# After re-nesting the geometry column has type
# ``List(List(List(List(Float64))))`` — depth-4 — and ``.ak.geo`` operates
# through the outer list automatically via ``dec()`` tree-walking.

# %%
import time

perims_ak = ak.from_arrow(gdf["perimeters"].to_arrow())
flat_wkb_pa = pa.array(ak.ravel(perims_ak).to_list(), type=pa.large_binary())
# Wrap as a polars Series so the .ak.geo accessor is available
flat_wkb_pl = pl.from_arrow(flat_wkb_pa)

t0 = time.perf_counter()
flat_geom_pl = flat_wkb_pl.ak.geo.from_wkb()   # vectorised shapely decode
t1 = time.perf_counter()
print(f"from_wkb on {len(flat_wkb_pl):,} perimeters: {(t1-t0)*1000:.0f} ms")
flat_geom_ak = ak.from_arrow(flat_geom_pl.to_arrow())

# Re-nest: group decoded geometries back by fire name
counts = ak.num(perims_ak, axis=1).to_list()
split_geom_ak = ak.unflatten(flat_geom_ak, counts, axis=0)
print(f"Re-nested type: {split_geom_ak.type}")

# Convert to a polars Series.  from_wkb guarantees a canonically-typed
# all-list Arrow array so polars can round-trip it without panicking.
geom_list_pl = pl.from_arrow(
    pa.array(ak.to_list(split_geom_ak))
)
gdf = gdf.with_columns(geom_list_pl.alias("geom_list"))
print("geom_list dtype:", gdf["geom_list"].dtype)

# %% [markdown]
# ### Per-event areas  (pure numba, depth-4 via dec() tree walk)
#
# ``.ak.geo.area3()`` is wired with ``match=match_multipolygon``.  The
# ``dec()`` tree-walker descends through the outer ``List`` at depth-4 and
# applies ``area3`` to each depth-3 ``MultiPolygon`` element inside each row.
# The result is ``List(Float64)`` — one area per event per fire name.
# **No workarounds needed**: the consistent list type from ``from_wkb`` makes
# ``Series.to_arrow()`` work correctly inside the ``.ak`` accessor.

# %%
event_areas = gdf["geom_list"].ak.geo.area3()
print("event_areas dtype:", event_areas.dtype)
print("  = List(Float64) — one area per event per fire name")
print()
# Sample a fire with multiple events
multi = gdf.filter(pl.col("n_events") >= 5).head(1)
idx = multi.row(0, named=True)
print(f"Fire: {idx['fire_name']}, {idx['n_events']} events")
print(f"Years: {idx['years']}")
print(f"Acres: {idx['acres']}")
fire_idx = gdf["fire_name"].to_list().index(idx["fire_name"])
print(f"Areas: {event_areas[fire_idx]}")

# %% [markdown]
# ### Aggregate bounding box per fire  (pure numba, depth-4 via dec())
#
# ``bounds`` uses ``match_any_geom`` and walks through the outer list,
# returning one aggregate bounding box per fire name (across all events).

# %%
agg_bounds = gdf["geom_list"].ak.geo.bounds()
print("bounds dtype:", agg_bounds.dtype)
print("  = Struct per fire name — aggregate bbox across all events")
gdf = gdf.with_columns(
    agg_bounds.struct.field("xmin").alias("hist_xmin"),
    agg_bounds.struct.field("ymin").alias("hist_ymin"),
    agg_bounds.struct.field("xmax").alias("hist_xmax"),
    agg_bounds.struct.field("ymax").alias("hist_ymax"),
)
gdf = gdf.with_columns(
    (pl.col("hist_xmax") - pl.col("hist_xmin")).alias("hist_width_deg"),
)
print("Fires with largest historical geographic footprint:")
gdf.sort("hist_width_deg", descending=True).select(
    "fire_name", "n_events", "year_min", "year_max", "hist_width_deg"
).head(8)

# %% [markdown]
# ### Cumulative burned area  (`ak.transform` + `ak.sum`/`ak.max`)
#
# ``event_areas`` is ``List(Float64)`` with negative values (CW ring orientation).
# ``.ak.transform(abs)`` applies element-wise abs across all list elements,
# returning a new ``List(Float64)`` Series.  ``.ak.sum(axis=1)`` and
# ``.ak.max(axis=1)`` then reduce each row — identical to the pandas version.

# %%
abs_event_areas = event_areas.ak.transform(abs)
gdf = gdf.with_columns(
    event_areas.alias("event_areas"),
    abs_event_areas.ak.sum(axis=1).alias("sum_area_deg2"),
    abs_event_areas.ak.max(axis=1).alias("max_area_deg2"),
)
print("Total cumulative area — top fires:")
gdf.sort("sum_area_deg2", descending=True).select(
    "fire_name", "n_events", "year_min", "year_max", "total_acres", "sum_area_deg2"
).head(10)

# %%
print("Top recurring fires with per-event area breakdown:")
gdf.filter(pl.col("n_events") >= 3).sort("total_acres", descending=True).select(
    "fire_name", "n_events", "year_min", "year_max", "years", "event_areas", "total_acres"
).head(5)

# %% [markdown]
# ## Summary: polars vs pandas

# %%
print("Actual differences from the pandas version (04_mtbs_wildfire_perimeters.py):")
print()
print("  1. DataFrame mutation:")
print("       polars:  df.with_columns(series.alias('col'))")
print("       pandas:  df['col'] = series")
print()
print("  2. Struct field access:")
print("       polars:  bounds.struct.field('xmin')")
print("       pandas:  pd.DataFrame(bounds.tolist())['xmin']")
print("     Both return a float Series; the polars form is more concise.")
print()
print("All .ak and .ak.geo operations are otherwise identical between backends.")
