"""download_data.py — download and prepare all example datasets.

Run this script once before executing the example notebooks:

    python download_data.py

Each ``download_*`` function fetches one source file and (where needed)
derives the parquet file(s) that the notebooks actually read.  The source
zip/shapefile archives are retained alongside the derived parquet files so
that the derivation can be inspected and re-run if needed.

All files are written into the ``data/`` subdirectory next to this script.

Dataset/notebook mapping
------------------------
``download_us_counties()``
    Produces ``data/us_counties_2023.parquet``
    Used by: ``01_us_counties.py``

``download_la_county_blocks()``
    Produces ``data/la_county_blocks_2020.parquet``
    Used by: ``02_la_county_blocks.py``

``download_ais()``
    Produces ``data/AIS_2023_01_01.zip``  (retained for the raw-CSV cell)
             ``data/ais_vessels_2023_01_01.parquet``
    Used by: ``03_ais_vessel_trajectories.py``

``download_mtbs()``
    Produces ``data/mtbs_perimeter_data.zip``  (retained)
             ``data/mtbs_fires_flat.parquet``
             ``data/mtbs_fires_grouped.parquet``
    Used by: ``04_mtbs_wildfire_perimeters.py``
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def _download(url: str, dest: Path, desc: str) -> None:
    """Download *url* to *dest*, showing a simple progress indicator."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists ({dest.stat().st_size // 1_048_576} MB)")
        return
    print(f"  Downloading {desc} …")
    tmp = dest.with_suffix(".part")
    try:
        def _reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r    {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        print()
        tmp.rename(dest)
        print(f"  Saved {dest.name} ({dest.stat().st_size // 1_048_576} MB)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _import_or_die(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        sys.exit(
            f"Required package '{pkg}' is not installed.  "
            f"Install with: pip install {pkg}"
        )


# ---------------------------------------------------------------------------
# Dataset 1: US Counties 2023
# ---------------------------------------------------------------------------

def download_us_counties() -> None:
    """Download and convert US Census TIGER 2023 county boundaries.

    Source:
        https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
        TIGER/Line Shapefiles 2023, County and County Equivalent

    Produces:
        data/tl_2023_us_county.zip    (~80 MB, retained)
        data/us_counties_2023.parquet (~120 MB, GeoParquet WKB)

    Used by: 01_us_counties.py
    """
    _ensure_data_dir()
    _import_or_die("geopandas")

    import geopandas as gpd

    zip_path = DATA_DIR / "tl_2023_us_county.zip"
    parquet_path = DATA_DIR / "us_counties_2023.parquet"

    _download(
        "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip",
        zip_path,
        "US Counties TIGER 2023 (~80 MB)",
    )

    if parquet_path.exists():
        print(f"  [skip] {parquet_path.name} already exists")
        return

    print("  Converting to GeoParquet …")
    gdf = gpd.read_file(zip_path)
    gdf["ALAND_km2"]  = gdf["ALAND"]  / 1e6
    gdf["AWATER_km2"] = gdf["AWATER"] / 1e6
    gdf["INTPTLAT"]   = gdf["INTPTLAT"].astype(float)
    gdf["INTPTLON"]   = gdf["INTPTLON"].astype(float)
    gdf.to_parquet(parquet_path, index=False)
    print(f"  Saved {parquet_path.name} ({parquet_path.stat().st_size // 1_048_576} MB)")


# ---------------------------------------------------------------------------
# Dataset 2: California Census Blocks 2020 (LA County subset)
# ---------------------------------------------------------------------------

def download_la_county_blocks() -> None:
    """Download California census blocks and extract Los Angeles County.

    Source:
        https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
        TIGER/Line Shapefiles 2023, Tabulation Block 2020 — California (state 06)

    Produces:
        data/tl_2023_06_tabblock20.zip      (~351 MB, retained)
        data/la_county_blocks_2020.parquet  (~35 MB, LA County only)

    LA County is FIPS county 037 within state 06.

    Used by: 02_la_county_blocks.py
    """
    _ensure_data_dir()
    _import_or_die("geopandas")

    import geopandas as gpd

    zip_path = DATA_DIR / "tl_2023_06_tabblock20.zip"
    parquet_path = DATA_DIR / "la_county_blocks_2020.parquet"

    _download(
        "https://www2.census.gov/geo/tiger/TIGER2023/TABBLOCK20/tl_2023_06_tabblock20.zip",
        zip_path,
        "California Census Blocks TIGER 2023 (~351 MB)",
    )

    if parquet_path.exists():
        print(f"  [skip] {parquet_path.name} already exists")
        return

    print("  Filtering to LA County (FIPS 037) …")
    gdf = gpd.read_file(zip_path)
    la = gdf[gdf["COUNTYFP20"] == "037"].copy()
    la["ALAND20_km2"]  = la["ALAND20"]  / 1e6
    la["AWATER20_km2"] = la["AWATER20"] / 1e6
    la.to_parquet(parquet_path, index=False)
    print(f"  Saved {parquet_path.name} ({parquet_path.stat().st_size // 1_048_576} MB), "
          f"{len(la):,} blocks")


# ---------------------------------------------------------------------------
# Dataset 3: NOAA AIS vessel positions (2023-01-01)
# ---------------------------------------------------------------------------

def download_ais() -> None:
    """Download one day of NOAA AIS vessel positions and build trajectory table.

    Source:
        https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/
        AIS (Automatic Identification System) daily CSV, 2023-01-01

    Produces:
        data/AIS_2023_01_01.zip             (~305 MB, retained for raw-CSV cell
                                             in the notebook)
        data/ais_vessels_2023_01_01.parquet (~40 MB)

    The parquet file has one row per vessel (MMSI) with:
        - scalar metadata: VesselName, VesselType, Length, n_pings,
          sog_mean, sog_max
        - waypoints: list<FixedSizeList[2]<double>> — ordered [lon, lat] pairs
          representing the vessel's complete track for the day.
    The waypoints column is already in native GeoArrow interleaved format;
    read_parquet() loads it with no conversion.

    Used by: 03_ais_vessel_trajectories.py
    """
    _ensure_data_dir()

    zip_path     = DATA_DIR / "AIS_2023_01_01.zip"
    parquet_path = DATA_DIR / "ais_vessels_2023_01_01.parquet"

    _download(
        "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_01.zip",
        zip_path,
        "NOAA AIS 2023-01-01 (~305 MB)",
    )

    if parquet_path.exists():
        print(f"  [skip] {parquet_path.name} already exists")
        return

    print("  Building vessel trajectory parquet …")
    import io
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    with zipfile.ZipFile(zip_path) as z:
        with z.open("AIS_2023_01_01.csv") as f:
            df = pd.read_csv(
                f,
                dtype={"MMSI": str, "SOG": float, "COG": float,
                       "LAT": float, "LON": float},
                usecols=["MMSI", "BaseDateTime", "LAT", "LON", "SOG",
                         "COG", "Heading", "VesselName", "VesselType", "Length"],
                parse_dates=["BaseDateTime"],
            )

    print(f"    Loaded {len(df):,} pings for {df['MMSI'].nunique():,} vessels")
    df = df.sort_values(["MMSI", "BaseDateTime"])

    # Per-vessel scalar metadata
    meta = (
        df.groupby("MMSI", sort=False)
          .agg(
              VesselName=("VesselName", "first"),
              VesselType=("VesselType", "first"),
              Length    =("Length",     "first"),
              n_pings   =("LAT",        "count"),
              sog_mean  =("SOG",        "mean"),
              sog_max   =("SOG",        "max"),
          )
          .reset_index()
    )

    # Per-vessel waypoints: list<FixedSizeList[2]<double>>
    fsl_type = pa.list_(pa.list_(pa.field("xy", pa.float64()), 2))
    waypoint_lists = [
        grp[["LON", "LAT"]].values.tolist()
        for _, grp in df.groupby("MMSI", sort=False)
    ]

    table = pa.table({
        "MMSI":       pa.array(meta["MMSI"].tolist()),
        "VesselName": pa.array(meta["VesselName"].fillna("").tolist()),
        "VesselType": pa.array(meta["VesselType"].fillna(0).astype(int).tolist()),
        "Length":     pa.array(meta["Length"].fillna(0.0).tolist()),
        "n_pings":    pa.array(meta["n_pings"].tolist()),
        "sog_mean":   pa.array(meta["sog_mean"].round(2).tolist()),
        "sog_max":    pa.array(meta["sog_max"].tolist()),
        "waypoints":  pa.array(waypoint_lists, type=fsl_type),
    })
    pq.write_table(table, parquet_path, compression="zstd")
    print(f"  Saved {parquet_path.name} ({parquet_path.stat().st_size // 1_048_576} MB), "
          f"{len(table):,} vessels")


# ---------------------------------------------------------------------------
# Dataset 4: MTBS wildfire perimeters
# ---------------------------------------------------------------------------

def download_mtbs() -> None:
    """Download MTBS fire perimeters and build flat + grouped parquet files.

    Source:
        https://www.mtbs.gov/direct-download
        Monitoring Trends in Burn Severity — composite perimeter shapefile,
        all mapped US wildfires and prescribed fires 1984–present.

    Produces:
        data/mtbs_perimeter_data.zip    (~362 MB, retained)
        data/mtbs_fires_flat.parquet    (~7 MB)
            One row per fire event.  Geometry is WKB-encoded (decoded
            automatically by read_parquet()).  Additional columns: event_id,
            fire_name, fire_type, year, acres.

        data/mtbs_fires_grouped.parquet (~7 MB)
            One row per distinct fire name.  The ``perimeters`` column is
            ``list<large_binary>`` — a variable-length list of WKB-encoded
            perimeter polygons, one per year the fire was mapped.  Additional
            columns: n_events, year_min, year_max, total_acres, years, acres.

    Geometries are simplified to 0.005° (~500 m) tolerance to reduce file
    size while retaining all analytically meaningful shape information.

    Used by: 04_mtbs_wildfire_perimeters.py
    """
    _ensure_data_dir()
    _import_or_die("geopandas")
    _import_or_die("shapely")

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely as shp

    zip_path      = DATA_DIR / "mtbs_perimeter_data.zip"
    flat_path     = DATA_DIR / "mtbs_fires_flat.parquet"
    grouped_path  = DATA_DIR / "mtbs_fires_grouped.parquet"

    _download(
        "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/"
        "data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip",
        zip_path,
        "MTBS wildfire perimeters (~362 MB)",
    )

    if flat_path.exists() and grouped_path.exists():
        print(f"  [skip] {flat_path.name} and {grouped_path.name} already exist")
        return

    print("  Loading shapefile …")
    gdf = gpd.read_file(
        zip_path,
        columns=["event_id", "incid_name", "incid_type",
                 "ig_date", "burnbndac", "geometry"],
    )
    gdf["year"] = (
        pd.to_datetime(gdf["ig_date"], errors="coerce")
          .dt.year.fillna(0).astype(int)
    )
    print(f"    {len(gdf):,} fire events, "
          f"{gdf['incid_name'].nunique():,} distinct names")

    print("  Simplifying geometries (tolerance=0.005°) and encoding WKB …")
    geoms_np  = np.array(gdf.geometry.values, dtype=object)
    simplified = shp.simplify(geoms_np, tolerance=0.005, preserve_topology=True)
    wkb_list   = [shp.to_wkb(g) for g in simplified]

    # --- flat table ---
    if not flat_path.exists():
        geo_meta = {
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["Polygon", "MultiPolygon"],
                }
            },
            "version": "1.0.0",
        }
        flat = pa.table({
            "event_id":  pa.array(gdf["event_id"].tolist()),
            "fire_name": pa.array(gdf["incid_name"].fillna("UNKNOWN").tolist()),
            "fire_type": pa.array(gdf["incid_type"].fillna("Unknown").tolist()),
            "year":      pa.array(gdf["year"].tolist()),
            "acres":     pa.array(gdf["burnbndac"].fillna(0.0).tolist()),
            "geometry":  pa.array(wkb_list, type=pa.large_binary()),
        }).replace_schema_metadata(
            {b"geo": json.dumps(geo_meta).encode()}
        )
        pq.write_table(flat, flat_path, compression="zstd")
        print(f"  Saved {flat_path.name} ({flat_path.stat().st_size // (1024*1024)} MB)")

    # --- grouped table ---
    if not grouped_path.exists():
        print("  Building grouped table (fire name → list of perimeters) …")
        name_to = defaultdict(lambda: {"yrs": [], "acs": [], "wkbs": []})
        for i in range(len(gdf)):
            row  = gdf.iloc[i]
            name = row["incid_name"] or "UNKNOWN"
            name_to[name]["yrs"].append(int(row["year"]))
            name_to[name]["acs"].append(
                float(row["burnbndac"]) if pd.notna(row["burnbndac"]) else 0.0
            )
            name_to[name]["wkbs"].append(wkb_list[i])

        fire_names = sorted(name_to.keys())
        grouped = pa.table({
            "fire_name":   pa.array(fire_names),
            "n_events":    pa.array(
                [len(name_to[n]["yrs"]) for n in fire_names], type=pa.int32()
            ),
            "year_min":    pa.array(
                [min(name_to[n]["yrs"]) for n in fire_names], type=pa.int32()
            ),
            "year_max":    pa.array(
                [max(name_to[n]["yrs"]) for n in fire_names], type=pa.int32()
            ),
            "total_acres": pa.array(
                [sum(name_to[n]["acs"]) for n in fire_names]
            ),
            "years":       pa.array(
                [name_to[n]["yrs"] for n in fire_names], type=pa.list_(pa.int32())
            ),
            "acres":       pa.array(
                [name_to[n]["acs"] for n in fire_names], type=pa.list_(pa.float64())
            ),
            "perimeters":  pa.array(
                [name_to[n]["wkbs"] for n in fire_names],
                type=pa.list_(pa.large_binary()),
            ),
        })
        pq.write_table(grouped, grouped_path, compression="zstd")
        print(f"  Saved {grouped_path.name} ({grouped_path.stat().st_size // (1024*1024)} MB), "
              f"{len(fire_names):,} fires")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Download and prepare all datasets for the akimbo-geo examples.

    Downloads are skipped if the target files already exist.  Run from the
    ``examples/`` directory (the ``data/`` subdirectory is created if absent).

    Total download: ~1.1 GB (zips retained alongside parquet derivatives).
    Derived parquet files: ~220 MB total.
    """
    steps = [
        ("US Counties 2023",                download_us_counties),
        ("California Census Blocks 2020",   download_la_county_blocks),
        ("NOAA AIS vessel positions",        download_ais),
        ("MTBS wildfire perimeters",         download_mtbs),
    ]
    for name, fn in steps:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        fn()

    print("\nAll datasets ready in:", DATA_DIR.resolve())


if __name__ == "__main__":
    run()
