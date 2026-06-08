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
# # US Counties: geometry analysis with akimbo-geo
#
# **Dataset:** US Census TIGER 2023 county boundaries — 3,235 counties and
# county-equivalents across all 50 states, DC, Puerto Rico, and island territories.
#
# **File:** `data/us_counties_2023.parquet` (~120 MB, WKB-encoded GeoParquet)
#
# **Source:** https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
#
# Each row has:
# - `geometry` — polygon boundary (mix of Polygon and MultiPolygon)
# - `ALAND_km2` — geodesic land area in km²
# - `AWATER_km2` — water surface area in km²
# - `INTPTLAT` / `INTPTLON` — Census interior-point lat/lon
# - `NAME`, `NAMELSAD` — county name
# - `STATEFP`, `COUNTYFP`, `GEOID` — FIPS identifiers
#
# ## Workflow
#
# `akimbo_geo.convert.read_parquet` reads the file and decodes the WKB
# geometry column automatically — no geopandas, no awkward in the user's
# code.  Because the file contains a mix of `Polygon` and `MultiPolygon`
# geometries, shapely promotes the whole column to depth-3 (`MultiPolygon`).
# Operations that require depth-3 use the `area3`, `length3` variants.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import akimbo.pandas          # registers .ak accessor
import akimbo_geo             # registers .ak.geo sub-accessor
from akimbo_geo.convert import read_parquet

# %% [markdown]
# ## Load data
#
# `read_parquet` detects the geometry encoding from the GeoParquet metadata
# and decodes it to akimbo-geo's native coordinate format automatically.
# The result is a plain pandas DataFrame — no geopandas, no WKB objects.

# %%
df = read_parquet("data/us_counties_2023.parquet")
print(f"{len(df):,} counties")
print(f"geometry dtype: {df['geometry'].dtype}")
# depth-3 = list<list<list<float>>> = MultiPolygon (all Polygons promoted)
df[["NAME", "STATEFP", "ALAND_km2", "INTPTLAT", "INTPTLON"]].head(3)

# %% [markdown]
# ## Bounding box  (pure numba)
#
# `.ak.geo.bounds()` extracts the bounding rectangle of each geometry in a
# single parallel numba pass — works at any depth.

# %%
bbox = df["geometry"].ak.geo.bounds()
print("bounds() return dtype:", bbox.dtype)
df[["xmin", "ymin", "xmax", "ymax"]] = bbox.ak.unpack()

df["width_deg"]  = df["xmax"] - df["xmin"]
df["height_deg"] = df["ymax"] - df["ymin"]

print("\nCounties spanning the most longitude:")
df[["NAME", "NAMELSAD", "STATEFP", "width_deg"]].nlargest(8, "width_deg")

# %% [markdown]
# ## Bounding-box centroid  (pure arithmetic)

# %%
df["cx"] = (df["xmin"] + df["xmax"]) / 2
df["cy"] = (df["ymin"] + df["ymax"]) / 2

lat_diff = (df["cy"] - df["INTPTLAT"]).abs()
lon_diff = (df["cx"] - df["INTPTLON"]).abs()
print(f"Median bbox-centroid vs Census interior-point: {lat_diff.median():.4f} deg lat")

# %% [markdown]
# ## Area and perimeter  (pure numba, depth-3)

# %%
df["area_deg2"]     = df["geometry"].ak.geo.area3().abs()
df["perimeter_deg"] = df["geometry"].ak.geo.length3()

nonzero = df["ALAND_km2"] > 0
corr = np.corrcoef(
    np.log1p(df.loc[nonzero, "area_deg2"]),
    np.log1p(df.loc[nonzero, "ALAND_km2"]),
)[0, 1]
print(f"log-log correlation (shoelace deg2 vs ALAND km2): {corr:.4f}")

# %% [markdown]
# ## Shape compactness  (pure numba)

# %%
df["compactness"] = 4 * np.pi * df["area_deg2"] / df["perimeter_deg"] ** 2

print("Most compact counties:")
df[["NAME", "NAMELSAD", "STATEFP", "compactness"]].nlargest(8, "compactness")

# %%
print("Least compact counties:")
df[["NAME", "NAMELSAD", "STATEFP", "compactness"]].nsmallest(8, "compactness")

# %% [markdown]
# ## Polygon count per county  (pure akimbo)
#
# Most counties are single polygons wrapped in a MultiPolygon of size 1.
# Counties with `n_polygons > 1` are discontiguous (islands, exclaves, etc).

# %%
df["n_polygons"] = df["geometry"].ak.num(axis=1)

print("Polygon count distribution:")
print(df["n_polygons"].value_counts().sort_index().head(10))
print()
print("Counties with the most sub-polygons:")
df[["NAME", "NAMELSAD", "STATEFP", "n_polygons"]].nlargest(8, "n_polygons")

# %% [markdown]
# ## Minimum bounding radius  (pure numba)

# %%
df["mbr_deg"] = df["geometry"].ak.geo.minimum_bounding_radius()

print("Geographically largest counties by bounding radius:")
df[["NAME", "NAMELSAD", "STATEFP", "mbr_deg", "ALAND_km2"]].nlargest(8, "mbr_deg")

# %% [markdown]
# ## Translate all counties north  (pure numba)

# %%
geom_shifted = df["geometry"].ak.geo.translate(xoff=0.0, yoff=1.0)

shifted_bounds = geom_shifted.ak.geo.bounds()
cy_shifted = (shifted_bounds.ak["ymin"] + shifted_bounds.ak["ymax"]) / 2
delta = cy_shifted - df["cy"]
print(f"Mean northward shift: {delta.mean():.6f} deg  (expected 1.0)")
print(f"Std:                  {delta.std():.2e}")

# %% [markdown]
# ## Spatial filter: CONUS  (from bounds, no GEOS)

# %%
conus = {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
in_conus = (
    (df["xmax"] >= conus["xmin"]) & (df["xmin"] <= conus["xmax"])
    & (df["ymax"] >= conus["ymin"]) & (df["ymin"] <= conus["ymax"])
)
print(f"Counties whose bbox overlaps CONUS: {in_conus.sum():,} / {len(df):,}")

print("\nExcluded (outside CONUS):")
df.loc[~in_conus, ["NAME", "NAMELSAD", "STATEFP"]].sort_values("STATEFP").head(12)

# %% [markdown]
# ## Convex hull fill ratio  (shapely / GEOS)
#
# `convex_hull` of a MultiPolygon returns a single Polygon (depth-2);
# its area is computed with `.ak.geo.area()`.

# %%
hull_geom  = df["geometry"].ak.geo.convex_hull()
print("Convex hull dtype:", hull_geom.dtype)   # depth-2

hull_area  = hull_geom.ak.geo.area().abs()
df["hull_fill"] = hull_area.values / np.maximum(df["area_deg2"].values, 1e-15)

print("\nMost concave counties (lowest hull fill):")
df[["NAME", "NAMELSAD", "STATEFP", "hull_fill"]].nsmallest(8, "hull_fill")

# %% [markdown]
# ## Validity check  (shapely / GEOS)

# %%
df["is_valid"] = df["geometry"].ak.geo.is_valid()
print(f"Valid:   {df['is_valid'].sum():,}")
print(f"Invalid: {(~df['is_valid']).sum()}")

# %% [markdown]
# ## Summary

# %%
df[["NAME", "STATEFP", "ALAND_km2", "area_deg2", "perimeter_deg",
    "compactness", "n_polygons", "mbr_deg", "hull_fill", "is_valid"]].describe().round(3)
