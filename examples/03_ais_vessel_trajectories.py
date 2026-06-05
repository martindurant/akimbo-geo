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
# # AIS Vessel Trajectories: ragged list[Point] per vessel
#
# **Dataset:** NOAA MarineCadastre AIS (Automatic Identification System) vessel
# positions — 2023-01-01, aggregated from ~8.2 million position pings to one
# row per vessel.
#
# **File:** `data/ais_vessels_2023_01_01.parquet` (~40 MB)
#
# **Source:** https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/
#
# Each row is one vessel with:
# - `MMSI` — vessel identifier
# - `VesselName`, `VesselType`, `Length` — vessel metadata
# - `n_pings`, `sog_mean`, `sog_max` — derived ping statistics
# - `waypoints` — **`list<FixedSizeList[2]<double>>`** — the vessel's
#   complete ordered sequence of `[lon, lat]` position pings for the day
#
# ## Why this demonstrates akimbo's value
#
# This is a genuinely *ragged* column: different vessels have between 1 and
# 1,392 waypoints.  A moored vessel has ~10 pings; an active cargo ship
# crossing the Gulf has 1,000+.  Standard pandas/geopandas cannot represent
# this at all — a column where each row is a variable-length list of
# coordinate pairs.  With akimbo-geo every row has a different-length
# coordinate buffer, and all geometry operations (track length, bounding box,
# centroid, simplification) apply element-wise across the ragged structure in
# a single numba pass.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import awkward as ak
import akimbo.pandas
import akimbo_geo
import pyarrow.parquet as pq

# %% [markdown]
# ## Load data
#
# The `waypoints` column is already in native geoarrow format
# (`list<FixedSizeList[2]<double>>`).  `read_parquet` recognises this as
# depth-1 FSL geometry — no conversion needed.

# %%
df = pq.read_table("data/ais_vessels_2023_01_01.parquet").to_pandas(
    types_mapper=pd.ArrowDtype
)
print(f"{len(df):,} vessels, 2023-01-01")
print(f"waypoints dtype: {df['waypoints'].dtype}")
print()
print("Pings per vessel (ragged distribution):")
print(df["n_pings"].describe().astype(int))

# %%
df[["MMSI", "VesselName", "VesselType", "Length", "n_pings", "sog_mean", "sog_max"]].head(6)

# %% [markdown]
# ## Track length  (pure numba — depth-1 geometry)
#
# `.ak.geo.length()` computes the total Euclidean distance along each vessel's
# waypoint sequence.  Because the `waypoints` column is depth-1
# (`list<FSL[2]>`), this is a single `@ngpjit` pass over all 8.2 million
# coordinate pairs — one length value per vessel.
#
# Result is in degrees (geographic coordinates); multiply by ~111 km/deg for
# an approximate geodesic distance.

# %%
import time

t0 = time.perf_counter()
df["track_length_deg"] = df["waypoints"].ak.geo.length()
t1 = time.perf_counter()
print(f"length() on {len(df):,} vessels ({df['n_pings'].sum():,} total pings): {(t1-t0)*1000:.0f} ms")
print()

df["track_length_km"] = df["track_length_deg"] * 111.0

print("Track length statistics (km):")
print(df["track_length_km"].describe().round(1))

# %%
print("\nFastest-moving vessels (longest track in one day):")
df[["MMSI", "VesselName", "VesselType", "n_pings", "track_length_km", "sog_max"]].nlargest(
    10, "track_length_km"
)

# %% [markdown]
# ## Bounding box per vessel  (pure numba)
#
# The per-vessel bounding box shows the geographic extent of each track.
# Vessels with large boxes are actively transiting; vessels with tiny boxes
# are moored or operating in a small area.

# %%
bbox = df["waypoints"].ak.geo.bounds()
df[["xmin", "ymin", "xmax", "ymax"]] = pd.DataFrame(bbox.tolist())

df["bbox_width_deg"]  = df["xmax"] - df["xmin"]
df["bbox_height_deg"] = df["ymax"] - df["ymin"]
df["bbox_area_deg2"]  = df["bbox_width_deg"] * df["bbox_height_deg"]

print("Bounding box area distribution (deg²):")
print(df["bbox_area_deg2"].describe().round(4))

# %%
print("\nVessels with the largest geographic footprint:")
df[["MMSI", "VesselName", "VesselType", "n_pings", "bbox_area_deg2"]].nlargest(8, "bbox_area_deg2")

# %% [markdown]
# ## Centroid of track  (pure numba)
#
# `.ak.geo.centroid()` computes the mean of all waypoints — the geographic
# centre of mass of the track.  For active vessels this approximates the
# midpoint of their journey; for moored vessels it approximates their berth.

# %%
centroids = pd.DataFrame(df["waypoints"].ak.geo.centroid().tolist()).rename(
    columns={"x": "centroid_lon", "y": "centroid_lat"}
)
df[["centroid_lon", "centroid_lat"]] = centroids

# How far is the centroid from the bbox midpoint?
bbox_mid_lon = (df["xmin"] + df["xmax"]) / 2
bbox_mid_lat = (df["ymin"] + df["ymax"]) / 2
displacement = np.sqrt(
    (centroids["centroid_lon"] - bbox_mid_lon) ** 2
    + (centroids["centroid_lat"] - bbox_mid_lat) ** 2
)
print(f"Median centroid vs bbox-midpoint displacement: {displacement.median():.5f} deg")

# %% [markdown]
# ## Coordinate count vs. `n_pings`  (pure numba — demonstrates depth-1)
#
# `.ak.geo.count_coordinates()` counts the number of coordinate *pairs*
# in each geometry.  For depth-1 (list of points), this is exactly the number
# of waypoints — the same as `n_pings`.

# %%
n_coords = df["waypoints"].ak.geo.count_coordinates()
# Compare as plain numpy arrays to avoid ArrowDtype issues
match = np.array(n_coords.tolist()) == np.array(df["n_pings"].tolist())
print(f"count_coordinates() == n_pings: {bool(match.all())} ({int(match.sum()):,} / {len(df):,})")

# %% [markdown]
# ## Spatial filter: vessels in the Gulf of Mexico  (pure numba — bounds)
#
# Extract vessels whose track intersects the Gulf of Mexico bounding box.
# The `intersects_bounds` kernel checks each track's individual segments
# against the query box.

# %%
# Gulf of Mexico approximate bounding box
gulf = {"x0": -97.0, "y0": 18.0, "x1": -80.0, "y1": 30.5}

in_gulf = df["waypoints"].ak.geo.intersects_bounds(
    gulf["x0"], gulf["y0"], gulf["x1"], gulf["y1"]
)
df["in_gulf"] = in_gulf
gulf_vessels = df[df["in_gulf"]]
print(f"Vessels with tracks intersecting the Gulf of Mexico: {len(gulf_vessels):,}")

print("\nGulf vessel type distribution (top 10):")
print(gulf_vessels["VesselType"].value_counts().head(10))

# %%
print("\nLongest tracks in the Gulf:")
gulf_vessels[["MMSI", "VesselName", "VesselType", "n_pings", "track_length_km"]].nlargest(
    8, "track_length_km"
)

# %% [markdown]
# ## Simplify tracks  (shapely / GEOS)
#
# Douglas–Peucker simplification reduces the number of waypoints while
# preserving the overall shape of the track.  Useful for display at small
# scales or when computing approximate metrics with fewer points.

# %%
# Simplification requires >= 2 waypoints; filter to active vessels
active = df[df["n_pings"] >= 2].copy()
print(f"Active vessels (>=2 pings): {len(active):,} / {len(df):,}")

t0 = time.perf_counter()
simplified = active["waypoints"].ak.geo.simplify(tolerance=0.05)
t1 = time.perf_counter()
print(f"simplify(0.05 deg) on {len(active):,} vessels: {(t1-t0)*1000:.0f} ms")

orig_pts = int(active["n_pings"].sum())
simp_pts = int(np.sum(np.array(simplified.ak.geo.count_coordinates().tolist())))
print(f"Waypoints: {orig_pts:,} → {simp_pts:,} ({(1-simp_pts/orig_pts)*100:.1f}% reduction)")

# Simplified track lengths — should be close to originals for small tolerance
simp_len_km = simplified.ak.geo.length() * 111.0
orig_len = np.array(active["track_length_km"].tolist(), dtype=float).clip(0.001)
simp_arr = np.array(simp_len_km.tolist(), dtype=float)
len_diff_pct = np.abs((simp_arr - orig_len) / orig_len).clip(0, 1) * 100
print(f"Median track length change after simplification: {np.median(len_diff_pct):.2f}%")

# %% [markdown]
# ## Multi-level nesting: hourly sub-tracks per vessel
#
# The same raw data can be grouped at a finer level: vessel × hour gives a
# `list[list[Point]]` (depth-2) structure — each row is one vessel–hour with
# a variable number of pings.  This demonstrates that akimbo-geo's operations
# generalise automatically to deeper nesting via `dec()` tree-walking.

# %%
import zipfile, io

print("Loading raw pings for multi-level demonstration...", end=" ", flush=True)
with zipfile.ZipFile("data/AIS_2023_01_01.zip") as z:
    with z.open("AIS_2023_01_01.csv") as f:
        raw = pd.read_csv(
            f, usecols=["MMSI", "BaseDateTime", "LAT", "LON"],
            dtype={"MMSI": str, "LAT": float, "LON": float},
            parse_dates=["BaseDateTime"],
        )
print(f"{len(raw):,} pings")

raw = raw.sort_values(["MMSI", "BaseDateTime"])
raw["hour"] = raw["BaseDateTime"].dt.hour

# Group by vessel + hour → list[Point] per vessel–hour pair
vessel_hour = (
    raw.groupby(["MMSI", "hour"])
    .apply(lambda g: [[float(lon), float(lat)] for lon, lat in zip(g.LON, g.LAT)])
    .reset_index(name="pings")
)
print(f"{len(vessel_hour):,} vessel–hour segments")
print(f"Pings per vessel–hour: min={vessel_hour.pings.apply(len).min()}, "
      f"median={vessel_hour.pings.apply(len).median():.0f}, "
      f"max={vessel_hour.pings.apply(len).max()}")

# %%
# Pack into a per-vessel list[list[Point]] = depth-2
# Each row = one vessel, each inner list = one hour's pings
import pyarrow as pa

vessel_segments = (
    vessel_hour.groupby("MMSI")["pings"]
    .apply(list)
    .reset_index(name="hourly_tracks")
)
# hourly_tracks is list[list[list[float]]] — depth-2 (list of lines, each line is a list of [lon,lat])
fsl_type = pa.list_(pa.list_(pa.list_(pa.field("xy", pa.float64()), 2)))
hourly_col = pd.Series(
    pa.array(vessel_segments["hourly_tracks"].tolist(), type=fsl_type),
    dtype=pd.ArrowDtype(fsl_type),
    index=vessel_segments.index,
)
vessel_segments["hourly_tracks"] = hourly_col
vessel_segments = vessel_segments.set_index("MMSI")

print(f"\n{len(vessel_segments):,} vessels with hourly track breakdowns")
print(f"hourly_tracks dtype: {vessel_segments['hourly_tracks'].dtype}")
print(f"  = list[list[FSL[2]]] = one list of hourly segments per vessel")

# %%
# .ak.geo.length() descends into the ragged structure automatically:
# depth-2 list[Line] → length per Line → result is list[float] per vessel
hourly_lengths = vessel_segments["hourly_tracks"].ak.geo.length()
print("hourly_lengths dtype:", hourly_lengths.dtype)
print("This is list<float>[pyarrow] — one track length per hour per vessel")
print()
print("Sample (MMSI=first vessel):")
first_vessel_lengths = hourly_lengths.iloc[0]
if isinstance(first_vessel_lengths, list):
    print(f"  {first_vessel_lengths[:5]} ...")
else:
    print(f"  {first_vessel_lengths.tolist()[:5]} ...")

# Total track length = sum of hourly lengths (a vector sum per vessel)
total_per_vessel = hourly_lengths.ak.sum(axis=1) * 111.0  # sum within each row
print()
print("Sum of hourly lengths vs. original track length (should be close):")
merged = vessel_segments.join(
    df.set_index("MMSI")[["track_length_km"]], how="inner"
)
merged["hourly_sum_km"] = total_per_vessel.values[:len(merged)]
diff = (merged["hourly_sum_km"] - merged["track_length_km"]).abs()
print(f"  Median absolute difference: {diff.median():.4f} km")

# %% [markdown]
# ## Summary

# %%
print(f"Dataset: {len(df):,} vessels, {df['n_pings'].sum():,} total position pings")
print(f"Ping range per vessel: {df['n_pings'].min()} – {df['n_pings'].max()}")
print()
print("Core insight:")
print("  Every .ak.geo operation iterates exactly once over the flat coordinate")
print("  buffers regardless of how many pings each vessel has — the ragged")
print("  structure is handled by the offset arrays, not Python loops.")
print()
print("Operations used:")
for op in ["length (track distance)", "bounds (geographic footprint)",
           "centroid (mean position)", "count_coordinates (n_pings check)",
           "intersects_bounds (Gulf filter)", "simplify (waypoint reduction)",
           "length on depth-2 (hourly sub-tracks via dec() tree-walk)"]:
    print(f"  .ak.geo.{op}")
