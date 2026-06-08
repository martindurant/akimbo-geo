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
# # MTBS Wildfire Perimeters: struct records and list[Polygon] per fire
#
# **Dataset:** Monitoring Trends in Burn Severity (MTBS) — all mapped US wildfire
# and prescribed fire perimeters 1984–2026, 30,607 events.
#
# **Files:**
# - `data/mtbs_fires_flat.parquet` (~7 MB) — one row per fire event, WKB geometry
# - `data/mtbs_fires_grouped.parquet` (~7 MB) — one row per fire name,
#   `perimeters: list<binary>` — a variable-length list of WKB geometries per row
#
# **Source:** https://www.mtbs.gov/direct-download
#
# ## Why this demonstrates akimbo's value
#
# **Part 1 — struct records with a geometry field.**  The flat table is a
# standard DataFrame where one column is a geometry, stored as WKB.  The
# `.ak.geo` accessor treats that column like any other akimbo column — no
# special geometry DataFrame type, no crs-aware index, just a plain column
# that geometry operations can be applied to.  Other numeric columns remain
# standard pandas columns, mixing freely.
#
# **Part 2 — `list[Polygon]` per fire (ragged grouped structure).**  The same
# geographic area can burn in multiple separate years.  The grouped table has
# one row per fire *name* and a variable-length list of perimeters over time —
# a structure geopandas cannot represent at all.  akimbo-geo's `dec()` tree-
# walker applies geometry operations to each perimeter in the list and returns
# the nested result as a `list<float>` Series.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import awkward as ak
import akimbo.pandas
import akimbo_geo
import pyarrow as pa
import pyarrow.parquet as pq
from akimbo_geo.convert import read_parquet, from_wkb

# %% [markdown]
# ## Part 1: Flat table — struct records with geometry field
#
# Load one row per fire event.  The `geometry` column is WKB-encoded;
# `read_parquet` decodes it automatically into the native coordinate format.

# %%
df = read_parquet("data/mtbs_fires_flat.parquet")
print(f"{len(df):,} fire events, {df['year'].min()}–{df['year'].max()}")
print(f"geometry dtype: {df['geometry'].dtype}")
print(f"fire_type distribution: {df['fire_type'].value_counts().to_dict()}")
df[["event_id", "fire_name", "fire_type", "year", "acres", "geometry"]].head(4)

# %% [markdown]
# ### Area per fire event  (pure numba)
#
# Each fire is a MultiPolygon (disjoint burned patches); `.ak.geo.area3()`
# handles depth-3.  The result mixes with the scalar `acres` column directly.

# %%
df["area_deg2"]  = df["geometry"].ak.geo.area3().abs()
df["perimeter_km"] = df["geometry"].ak.geo.length3() * 111.0

nonzero = df["acres"] > 0
corr = np.corrcoef(
    np.log1p(df.loc[nonzero, "area_deg2"]),
    np.log1p(df.loc[nonzero, "acres"]),
)[0, 1]
print(f"log-log correlation (shoelace deg² vs reported acres): {corr:.4f}")

# %%
print("Largest fires by area (shoelace):")
df[["fire_name", "fire_type", "year", "acres", "area_deg2"]].nlargest(8, "area_deg2")

# %% [markdown]
# ### MultiPolygon disjointness  (pure akimbo)
#
# A fire event stored as a MultiPolygon has multiple disjoint burned patches.
# ``series.ak.num(axis=1)`` counts the outer-list length per row — the number
# of polygon patches per event.

# %%
df["n_patches"] = df["geometry"].ak.num(axis=1)

print("Patch count distribution:")
print(df["n_patches"].value_counts().sort_index().head(8))
print()
print("Fire events with the most disjoint burned patches:")
df[["fire_name", "fire_type", "year", "acres", "n_patches"]].nlargest(8, "n_patches")

# %% [markdown]
# ### Bounding box and shape compactness  (pure numba)
#
# This works exactly the same way as in the county/block notebooks —
# the geometry column is treated as a plain column, other columns remain
# unchanged.

# %%
bbox = df["geometry"].ak.geo.bounds()
df[["xmin", "ymin", "xmax", "ymax"]] = bbox.ak.unpack()

# Isoperimetric compactness: 4π·Area / Perimeter²
# Use degree² area and degree perimeter (consistent units)
df["perimeter_deg"] = df["geometry"].ak.geo.length3()
df["compactness"] = np.where(
    df["perimeter_deg"] > 0,
    4 * np.pi * df["area_deg2"] / df["perimeter_deg"] ** 2,
    np.nan,
)

print("Shape compactness by fire type:")
print(df.groupby("fire_type")["compactness"].describe(percentiles=[0.25, 0.5, 0.75]).round(3).to_string())

# %% [markdown]
# ### Trend over time: total annual burned area  (pure pandas on numba results)
#
# The `area_deg2` column is just a float column; pandas groupby/sum works
# normally on it alongside `acres`.

# %%
annual = (
    df.groupby("year")[["acres", "area_deg2"]]
    .agg(total_fires=("acres", "count"),
         total_acres=("acres", "sum"),
         total_area_deg2=("area_deg2", "sum"))
    .reset_index()
)
annual = annual[annual["year"].between(1990, 2024)]
print("Annual burned area trend (5-year bands):")
print(annual.groupby((annual["year"] // 5) * 5).agg(
    total_fires=("total_fires", "sum"),
    total_acres=("total_acres", "sum"),
).to_string())

# %% [markdown]
# ## Part 2: Grouped table — `list[Polygon]` per fire name
#
# Each row is one *fire name* with a variable-length list of WKB-encoded
# perimeters — one per year the fire was mapped.  This structure has no
# geopandas equivalent.

# %%
gdf = pq.read_table("data/mtbs_fires_grouped.parquet").to_pandas(
    types_mapper=pd.ArrowDtype
)
print(f"{len(gdf):,} distinct fire names")
print(f"perimeters dtype: {gdf['perimeters'].dtype}")
print(f"  = list<WKB bytes> — one WKB entry per mapped event per fire name")
print()
print("Events per fire name (n_events distribution):")
print(gdf["n_events"].describe().round(1))

# %%
print("\nFire names mapped in the most years:")
gdf[["fire_name", "n_events", "year_min", "year_max", "total_acres"]].nlargest(12, "n_events")

# %% [markdown]
# ### Decode `list[WKB]` into `list[Polygon]`  (shapely vectorised)
#
# Each cell in the `perimeters` column is a **Python list of WKB bytes**.
# We use `.ak.geo.from_wkb()` — which calls `shapely.from_wkb` + `to_ragged_array`
# at the C level — to decode the whole nested column at once.
#
# The result is a `list<Polygon-coordinates>` Series: each row is a list of
# polygon coordinate arrays, one per fire event for that name.

# %%
# The perimeters column contains list<large_binary> (list of WKB per fire)
# We need to decode each WKB in each list.
# ak.geo.from_wkb expects a flat array, so we flatten, decode, then re-nest.

import time

# Stack all WKB bytes into a flat array, decode, then split back by fire
perims_ak = ak.from_arrow(gdf["perimeters"].ak.arrow)
# Flatten: list<binary> per fire -> flat binary array with offsets
flat_wkb  = ak.ravel(perims_ak)         # flat array of WKB bytes
flat_wkb_arr = ak.from_arrow(
    pa.array(flat_wkb.tolist(), type=pa.large_binary())
)

print(f"Total WKB entries: {len(flat_wkb_arr):,}")
t0 = time.perf_counter()
flat_geom = from_wkb(flat_wkb_arr)       # vectorised shapely decode
t1 = time.perf_counter()
print(f"from_wkb on {len(flat_wkb_arr):,} fire perimeters: {(t1-t0)*1000:.0f} ms")
print(f"Decoded geometry type: {flat_geom.type}")

# Re-split into the original list-of-geometries structure using the original offsets
# counts[i] = number of events for fire i
counts = ak.num(perims_ak, axis=1).tolist()
split_geom = ak.unflatten(flat_geom, counts, axis=0)
print(f"\nRe-nested geometry type: {split_geom.type}")
print(f"This is: list[Polygon] per fire name")
print(f"  — {len(split_geom):,} rows, each a variable-length list of polygon coordinates")

# %%
# Store back as an ArrowDtype column
geom_pa_col = ak.to_arrow(split_geom, extensionarray=False)
gdf["geom_list"] = pd.array(geom_pa_col, dtype=pd.ArrowDtype(geom_pa_col.type))
print(f"geom_list dtype: {gdf['geom_list'].dtype}")
print(f"  = list[depth-3 MultiPolygon] per fire name")

# %% [markdown]
# ### Area of each individual perimeter event  (pure numba, depth-4 tree walk)
#
# `.ak.geo.area3()` is wired with `match=match_multipolygon` which matches
# depth-3 nodes.  Because `geom_list` is `list[depth-3]` = depth-4, `dec()`
# descends one level and applies `area3` to each depth-3 element inside each
# row's list.  The result is `list<float>` per fire name — one area per event.

# %%
event_areas_raw = gdf["geom_list"].ak.geo.area3()
# area3 returns list<float>; take abs element-wise using numpy on the nested structure
event_areas = pd.Series(
    [list(np.abs(v)) if isinstance(v, list) else v for v in event_areas_raw.tolist()],
    dtype=event_areas_raw.dtype, index=event_areas_raw.index
)
print("event_areas dtype:", event_areas.dtype)
print("  = list<float>[pyarrow] — one area per event per fire name")
print()
print("Sample (fire with multiple events):")
multi = gdf[gdf["n_events"] >= 5].iloc[0]
print(f"  Fire: {multi['fire_name']}, {multi['n_events']} events ({multi['year_min']}–{multi['year_max']})")
idx = gdf[gdf["n_events"] >= 5].index[0]
print(f"  Areas (deg²): {event_areas.iloc[idx]}")
acres_val = gdf.loc[idx, 'acres']
print(f"  Acres:        {acres_val if isinstance(acres_val, list) else list(acres_val)}")

# %%
# Max and sum of perimeter areas per fire (cumulative burn footprint).
# event_areas values are signed (negative for CW rings); apply abs first.
abs_event_areas = event_areas.ak.transform(abs)
gdf["max_event_area"] = abs_event_areas.ak.max(axis=1)
gdf["sum_event_area"] = abs_event_areas.ak.sum(axis=1)

print("Total cumulative area (sum of all events) — top fires:")
gdf[["fire_name", "n_events", "year_min", "year_max", "total_acres", "sum_event_area"]].nlargest(
    10, "sum_event_area"
)

# %% [markdown]
# ### Aggregate bounding box per fire name  (pure numba, dec() tree walk)
#
# `bounds` matches `match_any_geom` and walks through the outer list column,
# computing the aggregate bounding box across all events for each fire name
# (since the depth-4 list wraps all events, the result is one box per fire).

# %%
event_bounds = gdf["geom_list"].ak.geo.bounds()
print("event_bounds dtype:", event_bounds.dtype)
print("  = struct per fire name — aggregate bbox across all events in its history")
print(f"  len={len(event_bounds)} rows (one per fire name)")

# Unpack the aggregate bounding boxes
gdf[["hist_xmin", "hist_ymin", "hist_xmax", "hist_ymax"]] = event_bounds.ak.unpack()
gdf["hist_width_deg"]  = gdf["hist_xmax"] - gdf["hist_xmin"]
gdf["hist_height_deg"] = gdf["hist_ymax"] - gdf["hist_ymin"]

print("\nFires with the largest historical geographic footprint:")
gdf[["fire_name", "n_events", "year_min", "year_max",
     "hist_width_deg", "hist_height_deg"]].nlargest(8, "hist_width_deg")

# %% [markdown]
# ### Year-to-year area growth  (pure pandas on numba results)
#
# Because `event_areas` is a `list<float>` Series, we can convert it to a
# Python list and compute the cumulative sum of burn area per fire.
# All the heavy geometry work is done; the rest is plain pandas.

# %%
multi_burn = gdf[gdf["n_events"] >= 3].copy()

# Pair area and year for each event
multi_burn["event_area_list"] = event_areas[multi_burn.index].tolist()
multi_burn["year_list"]       = gdf.loc[multi_burn.index, "years"].tolist()

# Cumulative burn footprint over time for the top 5 most-burned fire names
print("Cumulative burn footprint (deg²) per event for top recurring fires:")
for _, row in multi_burn.nlargest(5, "total_acres").iterrows():
    yl = sorted(zip(row["year_list"], row["event_area_list"]), key=lambda x: x[0])
    cumsum = np.cumsum([a for _, a in yl])
    years_str = " → ".join(f"{y}({a:.4f})" for (y, a) in yl)
    print(f"  {row['fire_name']}: {years_str}")
    print(f"    cumulative: {list(np.round(cumsum, 4))}")

# %% [markdown]
# ## Summary

# %%
print("MTBS dataset: 30,607 fire events, 17,182 distinct fire names")
print()
print("Part 1 — flat table (one geometry per row, mixed with scalar columns):")
print("  .ak.geo.area3()      → acres per fire event")
print("  .ak.geo.length3()    → perimeter per fire event")
print("  .ak.geo.bounds()     → bounding box per fire event")
print("  ak.num(geom_ak)      → disjoint patch count per event")
print()
print("Part 2 — grouped table (list[Polygon] per fire name):")
print("  from_wkb()           → decode list<WKB> to list<coordinates>")
print("  ak.unflatten()       → re-nest flat geometry array by fire")
print("  .ak.geo.area3()      → list<float> of per-event areas (dec() tree walk)")
print("  .ak.geo.bounds()     → list<struct> of per-event bboxes (dec() tree walk)")
print()
print("Key difference from geopandas:")
print("  geopandas requires one geometry per row.")
print("  akimbo-geo handles list[Polygon] per row natively — the dec() tree-walker")
print("  descends through the outer list and applies the kernel to each element.")
