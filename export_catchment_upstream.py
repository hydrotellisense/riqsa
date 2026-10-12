"""
Export per-reservoir CATCHMENT polygons and UPSTREAM river paths to GeoJSON
for the web atlas (index.html).

Sources
  Catchment : Dataset/Merged_Catchment/GBMIM_barrier_merged_watersheds_filled.gpkg
              Dataset/Merged_Catchment/GBMIM_reservoir_merged_watersheds_filled.gpkg
              (same files + same barrier-first rule as the runoff-correction script;
               matched on GDW_ID)
  Upstream  : Dataset/Upstream/Barrier/<ID>_upstream_path.gpkg
              Dataset/Upstream/Reservoir/<ID>_upstream_path.gpkg

Output (named by gww_id, like the existing watershed_/downstream_ files)
  geojson/reservoir/catchment_<gww_id>.geojson
  geojson/reservoir/upstream_<gww_id>.geojson

Run from the project root:
  python export_catchment_upstream.py
  python export_catchment_upstream.py --upstream-id gdw --simplify 0.0005 --precision 5
  python export_catchment_upstream.py --only 311 1024 --overwrite
"""

import os
import json
import argparse
import fiona
import geopandas as gpd

RESERVOIR_JSON      = "geojson/reservoir.json"
OUT_DIR             = "geojson/reservoir"            # actual (uncorrected) per-reservoir files
WATERSHED_BARRIER   = "Dataset/Merged_Catchment/GBMIM_barrier_merged_watersheds_filled.gpkg"
WATERSHED_RESERVOIR = "Dataset/Merged_Catchment/GBMIM_reservoir_merged_watersheds_filled.gpkg"
UPSTREAM_DIRS       = [("Barrier",   "Dataset/Upstream/Barrier"),
                       ("Reservoir", "Dataset/Upstream/Reservoir")]
UPSTREAM_PATTERN    = "{id}_upstream_path.gpkg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_id(v):
    """'311', 311, 311.0 -> '311'; None/NaN/'' -> None."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return None
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def load_reservoirs(path):
    """[(gww_id, gdw_id), ...] from reservoir.json."""
    with open(path) as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = list(records.values())
    out = []
    for r in records:
        gww = norm_id(r.get("gww_id"))
        gdw = norm_id(r.get("id"))
        if gww:
            out.append((gww, gdw))
    return out


def read_first_layer(path):
    layers = fiona.listlayers(path)
    gdf = gpd.read_file(path, layer=layers[0])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def write_geojson(gdf, path, simplify, precision):
    """Write a slim WGS84 GeoJSON. Returns True if anything was written."""
    gdf = gdf[gdf.geometry.notna()].copy()
    if simplify > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        return False
    if os.path.exists(path):
        os.remove(path)          # GeoJSON driver will not overwrite in place
    gdf.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=precision)
    return True


# ---------------------------------------------------------------------------
# Catchments
# ---------------------------------------------------------------------------

def load_catchments():
    """{gdw_id: geometry} — barrier GPKG first, reservoir fills the rest."""
    gdw_to_geom = {}
    for label, path in [("Barrier", WATERSHED_BARRIER), ("Reservoir", WATERSHED_RESERVOIR)]:
        if not os.path.exists(path):
            print(f"  [WARN] {label} GPKG not found: {path}")
            continue
        gdf = read_first_layer(path)
        if "GDW_ID" not in gdf.columns:
            print(f"  [WARN] {label} GPKG has no GDW_ID column")
            continue
        added = 0
        for gdw_raw, geom in zip(gdf["GDW_ID"], gdf.geometry):
            gdw = norm_id(gdw_raw)
            if gdw is None or gdw in gdw_to_geom or geom is None:
                continue
            gdw_to_geom[gdw] = geom
            added += 1
        print(f"  {label}: {added} catchments (total {len(gdw_to_geom)})")
    return gdw_to_geom


def export_catchments(reservoirs, args):
    print("\n[1] Catchments")
    gdw_to_geom = load_catchments()
    written = skipped = missing = 0
    for gww, gdw in reservoirs:
        out = os.path.join(OUT_DIR, f"catchment_{gww}.geojson")
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        geom = gdw_to_geom.get(gdw)
        if geom is None:
            missing += 1
            continue
        gdf = gpd.GeoDataFrame({"gww_id": [gww], "gdw_id": [gdw]},
                               geometry=[geom], crs="EPSG:4326")
        if write_geojson(gdf, out, args.simplify, args.precision):
            written += 1
    print(f"  written {written} · already existed {skipped} · no catchment {missing}")


# ---------------------------------------------------------------------------
# Upstream paths
# ---------------------------------------------------------------------------

def find_upstream_file(gww, gdw, mode):
    if mode == "gdw":
        ids = [gdw]
    elif mode == "gww":
        ids = [gww]
    else:                              # auto: try GDW_ID, then gww_id
        ids = [gdw, gww]
    for fid in ids:
        if not fid:
            continue
        for _, folder in UPSTREAM_DIRS:
            p = os.path.join(folder, UPSTREAM_PATTERN.format(id=fid))
            if os.path.exists(p):
                return p, fid
    return None, None


def export_upstream(reservoirs, args):
    print("\n[2] Upstream paths")
    written = skipped = missing = empty = 0
    matched_by = {"gdw": 0, "gww": 0}
    for gww, gdw in reservoirs:
        out = os.path.join(OUT_DIR, f"upstream_{gww}.geojson")
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        src, fid = find_upstream_file(gww, gdw, args.upstream_id)
        if src is None:
            missing += 1          # headwater or no file
            continue
        matched_by["gdw" if fid == gdw else "gww"] += 1
        try:
            gdf = read_first_layer(src)
        except Exception as e:
            print(f"  [WARN] gww={gww}: could not read {src} ({e})")
            continue
        gdf = gdf[["geometry"]].copy()       # geometry only keeps files small
        if write_geojson(gdf, out, args.simplify, args.precision):
            written += 1
        else:
            empty += 1
    print(f"  written {written} · already existed {skipped} · "
          f"no file {missing} · empty {empty}")
    print(f"  files matched by GDW_ID: {matched_by['gdw']} · by gww_id: {matched_by['gww']}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-id", choices=["auto", "gdw", "gww"], default="auto",
                    help="which ID the upstream GPKG filenames use (default: try GDW_ID, then gww_id)")
    ap.add_argument("--simplify", type=float, default=0.0,
                    help="simplify tolerance in degrees (e.g. 0.0005 ≈ 50 m); 0 = off")
    ap.add_argument("--precision", type=int, default=5,
                    help="decimal places for coordinates (5 ≈ 1 m)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="limit to these gww_ids (for testing)")
    ap.add_argument("--skip-catchment", action="store_true")
    ap.add_argument("--skip-upstream", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    reservoirs = load_reservoirs(RESERVOIR_JSON)
    if args.only:
        keep = {norm_id(x) for x in args.only}
        reservoirs = [(g, d) for g, d in reservoirs if g in keep]
    print(f"{len(reservoirs)} reservoirs from {RESERVOIR_JSON}")

    if not args.skip_catchment:
        export_catchments(reservoirs, args)
    if not args.skip_upstream:
        export_upstream(reservoirs, args)
    print(f"\nDone. Output: {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
