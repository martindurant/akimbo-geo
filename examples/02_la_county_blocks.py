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
# # LA County Census Blocks: population-density geometry analysis
#
# **Dataset:** US Census TIGER 2023 — Census Blocks for Los Angeles County
# (state FIPS 06, county FIPS 037).
#
# **File:** `data/la_county_blocks_2020.parquet` (~35 MB, WKB-encoded GeoParquet)
#
# **Source:** https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
#
# Each row is one census block with:
# - `geometry` — polygon boundary (EPSG:4269, geographic degrees)
# - `POP20` — 2020 decennial population
# - `HOUSING20` — housing unit count
# - `ALAND20_km2` — geodesic land area in km²
# - `UR20` — urban/rural classification (`U` / `R`)
# - `TRACTCE20` — census tract code
#
# ## Workflow
#
# `akimbo_geo.convert.read_parquet` reads the file and decodes WKB geometry
# into akimbo-geo's native format in one call.  Because 1 block out of 91,626
# is a MultiPolygon, shapely promotes the whole column to depth-3.
# All analysis uses `.ak.geo` on the `geometry` column directly.

# %% [markdown]
# ## Setup

# %%
import time

import numpy as np
import pandas as pd
import awkward as ak          # for ak.ravel/ak.sum on nested bool results
import akimbo.pandas          # registers .ak
import akimbo_geo             # registers .ak.geo
from akimbo_geo.convert import read_parquet

# %% [markdown]
# ## Load data

# %%
t0 = time.perf_counter()
df = read_parquet("data/la_county_blocks_2020.parquet")
t1 = time.perf_counter()

print(f"{len(df):,} census blocks loaded in {(t1-t0)*1000:.0f} ms")
print(f"geometry dtype:  {df['geometry'].dtype}")
print(f"2020 population: {df['POP20'].sum():,}")
print(f"Urban blocks:    {(df['UR20'] == 'U').sum():,}")
print(f"Rural blocks:    {(df['UR20'] == 'R').sum():,}")

df[["TRACTCE20", "POP20", "HOUSING20", "ALAND20_km2", "UR20", "geometry"]].head(3)

# %% [markdown]
# ## Bounding box  (pure numba)

# %%
t0 = time.perf_counter()
bbox = df["geometry"].ak.geo.bounds()
t1 = time.perf_counter()
print(f"bounds() on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")

df[["xmin", "ymin", "xmax", "ymax"]] = bbox.ak.unpack()
print(bbox.ak.unpack().describe().round(4))

# %% [markdown]
# ## Area  (pure numba, depth-3)
#
# The shoelace formula applied at MultiPolygon depth.
# Result is in degrees² — proportional to geodesic area.

# %%
t0 = time.perf_counter()
df["area_deg2"] = df["geometry"].ak.geo.area3().abs()
t1 = time.perf_counter()
print(f"area3() on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")

nonzero = df["ALAND20_km2"] > 0
corr = np.corrcoef(
    np.log1p(df.loc[nonzero, "area_deg2"]),
    np.log1p(df.loc[nonzero, "ALAND20_km2"]),
)[0, 1]
print(f"log-log correlation (shoelace deg2 vs ALAND km2): {corr:.5f}")

# %% [markdown]
# ## Population density

# %%
df["pop_density_km2"] = np.where(
    df["ALAND20_km2"] > 0,
    df["POP20"] / df["ALAND20_km2"],
    np.nan,
)

print("Population density (people/km2) by urban/rural class:")
print(
    df.groupby("UR20")["pop_density_km2"]
    .describe(percentiles=[0.5, 0.75, 0.95, 0.99])
    .round(1)
    .to_string()
)

# %%
print("\nTop 10 most densely populated blocks (min 10 residents):")
df.loc[df["POP20"] >= 10].nlargest(10, "pop_density_km2")[
    ["TRACTCE20", "POP20", "HOUSING20", "ALAND20_km2", "pop_density_km2", "UR20"]
]

# %% [markdown]
# ## Perimeter and compactness  (pure numba, depth-3)

# %%
t0 = time.perf_counter()
df["perimeter_deg"] = df["geometry"].ak.geo.length3()
t1 = time.perf_counter()
print(f"length3() on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")

df["compactness"] = np.where(
    df["perimeter_deg"] > 0,
    4 * np.pi * df["area_deg2"] / df["perimeter_deg"] ** 2,
    np.nan,
)

print("\nCompactness by urban/rural class:")
print(df.groupby("UR20")["compactness"]
      .describe(percentiles=[0.25, 0.5, 0.75])
      .round(4).to_string())

# %%
print("\nMost circular blocks (compactness near 1):")
df.nlargest(5, "compactness")[["TRACTCE20", "POP20", "ALAND20_km2", "compactness", "UR20"]]

# %% [markdown]
# ## Bounding-box centroid  (pure arithmetic)

# %%
df["cx"] = (df["xmin"] + df["xmax"]) / 2
df["cy"] = (df["ymin"] + df["ymax"]) / 2

df["INTPTLAT20"] = df["INTPTLAT20"].astype(float)
df["INTPTLON20"] = df["INTPTLON20"].astype(float)

lat_diff = (df["cy"] - df["INTPTLAT20"]).abs()
lon_diff = (df["cx"] - df["INTPTLON20"]).abs()
print(f"Median |bbox-centroid - Census interior-point|: "
      f"{lat_diff.median():.5f} deg lat, {lon_diff.median():.5f} deg lon")

# %% [markdown]
# ## Ring orientation  (pure numba, depth-2 via dec tree-walk)
#
# `is_ccw` and `orient_polygons` match at depth-2 (Polygon level).
# Because our depth-3 column wraps each polygon as a single-element
# MultiPolygon list, `dec` descends to the depth-2 nodes automatically.

# %%
t0 = time.perf_counter()
df["is_ccw"] = df["geometry"].ak.geo.is_ccw()
t1 = time.perf_counter()
print(f"is_ccw() on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")
# is_ccw returns list<bool> for depth-3; sum after flattening with .ak.ravel().ak.sum()
n_ccw = int(df["is_ccw"].ak.ravel().ak.sum())
print(f"CCW exteriors: {n_ccw:,} / {len(df):,}")

# %%
t0 = time.perf_counter()
geom_oriented = df["geometry"].ak.geo.orient_polygons(exterior_cw=False)
t1 = time.perf_counter()
print(f"orient_polygons() on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")

df["geom_ccw"] = geom_oriented
ccw_after = df["geom_ccw"].ak.geo.is_ccw()
n_ccw_after = int(ccw_after.ak.ravel().ak.sum())
print(f"CCW after orient_polygons: {n_ccw_after:,} / {len(df):,}")

# %% [markdown]
# ## Translate  (pure numba)

# %%
t0 = time.perf_counter()
geom_shifted = df["geometry"].ak.geo.translate(xoff=0.0, yoff=0.1)
t1 = time.perf_counter()
print(f"translate() on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")

bbox_shifted = geom_shifted.ak.geo.bounds()
cy_shifted   = (bbox_shifted.ak["ymin"] + bbox_shifted.ak["ymax"]) / 2
delta = cy_shifted - df["cy"]
print(f"Mean northward shift: {delta.mean():.6f} deg (expected 0.1)")

# %% [markdown]
# ## Downtown LA spatial filter  (from bounds)

# %%
dtla = {"xmin": -118.275, "ymin": 34.025, "xmax": -118.220, "ymax": 34.075}
in_dtla = (
    (df["xmax"] >= dtla["xmin"]) & (df["xmin"] <= dtla["xmax"])
    & (df["ymax"] >= dtla["ymin"]) & (df["ymin"] <= dtla["ymax"])
)
print(f"Blocks in Downtown LA bbox: {in_dtla.sum():,}")

dtla_gdf = df[in_dtla]
county_median = df["pop_density_km2"].median()
dtla_median   = dtla_gdf["pop_density_km2"].median()
print(f"DTLA median density:   {dtla_median:,.0f} people/km2")
print(f"County median density: {county_median:,.0f} people/km2")
print(f"DTLA/county ratio:     {dtla_median/county_median:.1f}x")

# %% [markdown]
# ## Simplification  (shapely / GEOS)

# %%
t0 = time.perf_counter()
geom_simplified = df["geom_ccw"].ak.geo.simplify(tolerance=0.0005)
t1 = time.perf_counter()
print(f"simplify(0.0005 deg) on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")

orig_coords = len(df["geom_ccw"].ak.ravel()) // 2
simp_coords = len(geom_simplified.ak.ravel()) // 2
print(f"Coordinate pairs: {orig_coords:,} -> {simp_coords:,}")
print(f"Reduction:        {(1 - simp_coords/orig_coords)*100:.1f}%")

# %% [markdown]
# ## Convex hull fill ratio  (shapely / GEOS)

# %%
t0 = time.perf_counter()
hull_geom = df["geometry"].ak.geo.convex_hull()
t1 = time.perf_counter()
print(f"convex_hull() on {len(df):,} polygons: {(t1-t0)*1000:.0f} ms")
# convex_hull of a MultiPolygon returns a Polygon (depth-2)
print("hull dtype:", hull_geom.dtype)

hull_area  = hull_geom.ak.geo.area().abs()
df["hull_fill"] = hull_area.values / np.maximum(df["area_deg2"].values, 1e-15)

print("\nHull fill ratio by urban/rural class:")
print(df.groupby("UR20")["hull_fill"]
      .describe(percentiles=[0.25, 0.5, 0.75])
      .round(4).to_string())

# %%
print("\nMost concave blocks (lowest hull fill, min 0.01 km2):")
df.loc[df["ALAND20_km2"] >= 0.01].nsmallest(8, "hull_fill")[
    ["TRACTCE20", "POP20", "ALAND20_km2", "hull_fill", "UR20"]
]

# %% [markdown]
# ## Timing summary

# %%
results = {}
for label, fn in [
    ("bounds",                lambda: df["geometry"].ak.geo.bounds()),
    ("area3",                 lambda: df["geometry"].ak.geo.area3()),
    ("length3",               lambda: df["geometry"].ak.geo.length3()),
    ("is_ccw",                lambda: df["geometry"].ak.geo.is_ccw()),
    ("orient_polygons",       lambda: df["geometry"].ak.geo.orient_polygons()),
    ("translate +0.1N",       lambda: df["geometry"].ak.geo.translate(0.0, 0.1)),
    ("minimum_bounding_radius", lambda: df["geometry"].ak.geo.minimum_bounding_radius()),
    ("convex_hull (shapely)", lambda: df["geometry"].ak.geo.convex_hull()),
    ("simplify 0.0005 (shapely)", lambda: df["geom_ccw"].ak.geo.simplify(0.0005)),
]:
    t0 = time.perf_counter()
    _ = fn()
    results[label] = (time.perf_counter() - t0) * 1000

timing = pd.DataFrame.from_dict(results, orient="index", columns=["ms"])
timing["us_per_block"] = timing["ms"] * 1000 / len(df)
timing = timing.round(1)
timing.index.name = "operation"

print(f"Timing on {len(df):,} LA County census blocks:")
print(timing.to_string())
