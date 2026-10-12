#!/usr/bin/env python3
"""Validate monthly website reservoir storage against CWC observations.

The script intentionally uses only the 10 reservoir matches confirmed by
coordinates. CWC weekly observations are aggregated to calendar months before
being joined to the website's monthly JSON series.

Important: the current website series may be surface-water area mislabeled as
storage. The default --website-unit unknown therefore creates the aligned
audit table but does not calculate volume-error metrics. Use --website-unit m3
or --website-unit mcm only after confirming that the JSON contains volume.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MATCHED_RESERVOIRS = {
    "BANSAGAR": "90228",
    "GANDHI SAGAR": "90381",
    "GOBIND SAGAR(BHAKRA)": "89238",
    "MATATILA": "88816",
    "PONG DAM": "89637",
    "RAMGANGA": "88176",
    "RANA PRATAP SAGAR": "89585",
    "RIHAND": "90276",
    "TEHRI": "86529",
    "THEIN": "88178",
}

SEASON_BY_MONTH = {
    1: "Dry",
    2: "Dry",
    3: "Dry",
    4: "Pre-Monsoon",
    5: "Pre-Monsoon",
    6: "Pre-Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Winter",
    11: "Winter",
    12: "Winter",
}

SEASON_ORDER = {
    "Dry": 1,
    "Pre-Monsoon": 2,
    "Monsoon": 3,
    "Winter": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the 10 coordinate-matched website reservoirs "
            "against CWC."
        )
    )
    parser.add_argument(
        "--cwc-file",
        "--cwc-csv",
        dest="cwc_file",
        required=True,
        type=Path,
        help=(
            "CWC source file. Supports the multi-tab .xlsx workbook "
            "and the long-form .csv file."
        ),
    )
    parser.add_argument(
        "--reservoir-json",
        required=True,
        type=Path,
        help="Local reservoir.json containing IDs and metadata.",
    )
    parser.add_argument(
        "--storage-json-dir",
        required=True,
        type=Path,
        help=(
            "Local directory containing storage_<gww_id>.json files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_output"),
    )
    parser.add_argument("--storage-field", default="Storage")
    parser.add_argument(
        "--website-unit",
        choices=("unknown", "m3", "mcm"),
        default="unknown",
        help=(
            "Unit of the website JSON values. Keep 'unknown' until "
            "the variable is confirmed. m3 is divided by 1,000,000; "
            "mcm is used unchanged."
        ),
    )
    parser.add_argument(
        "--monthly-stat",
        choices=("mean", "median"),
        default="mean",
        help="CWC monthly value used as the reference observation.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Create a time-series and 1:1 scatter plot for each reservoir.",
    )
    return parser.parse_args()


def true_flag(series: pd.Series) -> pd.Series:
    """Interpret common string/numeric Boolean representations."""
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes", "y"})


def load_cwc_source(source_path: Path) -> pd.DataFrame:
    suffix = source_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source_path, low_memory=False)

    if suffix in {".xlsx", ".xlsm"}:
        sheets = pd.read_excel(
            source_path,
            sheet_name=None,
            header=3,
            engine="openpyxl",
        )
        frames = []
        for sheet_name, sheet in sheets.items():
            sheet = sheet.dropna(how="all").copy()
            if sheet.empty:
                continue
            if "reservoir_name" not in sheet.columns:
                sheet["reservoir_name"] = sheet_name
            frames.append(sheet)

        if not frames:
            raise ValueError(
                "No data rows were found in the Excel workbook"
            )
        return pd.concat(frames, ignore_index=True)

    raise ValueError(
        "Unsupported CWC file type. Use .xlsx, .xlsm, or .csv"
    )


def load_cwc_monthly(source_path: Path) -> pd.DataFrame:
    df = load_cwc_source(source_path)
    required = {
        "observation_date",
        "reservoir_name",
        "current_live_storage_bcm",
        "live_capacity_at_frl_bcm",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CWC CSV is missing required columns: {missing}")

    df = df[df["reservoir_name"].isin(MATCHED_RESERVOIRS)].copy()
    df["observation_date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce",
    )
    df["current_live_storage_bcm"] = pd.to_numeric(
        df["current_live_storage_bcm"], errors="coerce"
    )
    df["live_capacity_at_frl_bcm"] = pd.to_numeric(
        df["live_capacity_at_frl_bcm"], errors="coerce"
    )

    # Remove records already flagged as physically inconsistent in the CWC CSV.
    invalid = pd.Series(False, index=df.index)
    for flag in ("storage_exceeds_capacity_flag", "level_above_frl_flag"):
        if flag in df.columns:
            invalid |= true_flag(df[flag])

    df = df[
        (~invalid)
        & df["observation_date"].notna()
        & df["current_live_storage_bcm"].notna()
    ].copy()

    # Primary CWC storage variable: live storage in BCM, converted to MCM.
    df["cwc_storage_mcm"] = df["current_live_storage_bcm"] * 1000.0
    df["cwc_live_capacity_mcm"] = df["live_capacity_at_frl_bcm"] * 1000.0
    df["month"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["reservoir_name", "month"], as_index=False)
        .agg(
            cwc_storage_mean_mcm=("cwc_storage_mcm", "mean"),
            cwc_storage_median_mcm=("cwc_storage_mcm", "median"),
            cwc_weekly_observation_count=("cwc_storage_mcm", "size"),
            cwc_live_capacity_mcm=("cwc_live_capacity_mcm", "median"),
        )
        .sort_values(["reservoir_name", "month"])
    )
    monthly["website_gww_id"] = monthly["reservoir_name"].map(
        MATCHED_RESERVOIRS
    )
    return monthly


def load_reservoir_metadata(json_path: Path) -> pd.DataFrame:
    with json_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("reservoir.json must contain a list of records")

    rows = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        gww_id = str(record.get("gww_id", "")).strip()
        if gww_id not in MATCHED_RESERVOIRS.values():
            continue
        rows.append(
            {
                "website_gww_id": gww_id,
                "website_gdw_id": record.get("id"),
                "website_latitude": record.get("lat"),
                "website_longitude": record.get("lng"),
                "website_country": record.get("country"),
                "website_basin": record.get("Basin"),
                "website_areal_impact": record.get("Areal_Impact"),
            }
        )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(
            "No confirmed matched GWW IDs were found in reservoir.json"
        )

    duplicated = metadata["website_gww_id"].duplicated(keep=False)
    if duplicated.any():
        ids = metadata.loc[duplicated, "website_gww_id"].tolist()
        raise ValueError(f"Duplicate GWW IDs in reservoir.json: {ids}")

    expected = set(MATCHED_RESERVOIRS.values())
    found = set(metadata["website_gww_id"])
    missing = sorted(expected - found)
    if missing:
        raise ValueError(
            "reservoir.json is missing matched GWW IDs: "
            f"{missing}"
        )
    return metadata


def load_storage_series(
    reservoir_name: str,
    gww_id: str,
    storage_dir: Path,
    field: str,
) -> pd.DataFrame:
    json_path = storage_dir / f"storage_{gww_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing local storage JSON: {json_path}"
        )

    with json_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if "dates" not in payload:
        raise ValueError(
            f"{json_path} does not contain a 'dates' array"
        )
    if field not in payload:
        available = sorted(k for k in payload if k != "dates")
        raise ValueError(
            f"{json_path} has no '{field}' field; "
            f"available fields: {available}"
        )
    if len(payload["dates"]) != len(payload[field]):
        raise ValueError(
            f"Date/value length mismatch in {json_path}"
        )

    out = pd.DataFrame(
        {
            "reservoir_name": reservoir_name,
            "website_gww_id": gww_id,
            "month": pd.to_datetime(payload["dates"], errors="coerce"),
            "website_value_raw": pd.to_numeric(
                payload[field],
                errors="coerce",
            ),
        }
    )
    return out.dropna(subset=["month", "website_value_raw"])


def load_all_storage_series(
    storage_dir: Path,
    field: str,
) -> pd.DataFrame:
    frames = [
        load_storage_series(
            name,
            gww_id,
            storage_dir,
            field,
        )
        for name, gww_id in MATCHED_RESERVOIRS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def finite_pair(
    obs: pd.Series,
    est: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    o = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(est, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(o) & np.isfinite(e)
    return o[keep], e[keep]


def calculate_metrics(
    group: pd.DataFrame,
    obs_col: str,
) -> dict[str, float | int | str]:
    obs, est = finite_pair(group[obs_col], group["website_storage_mcm"])
    n = len(obs)
    result: dict[str, float | int | str] = {
        "reservoir_name": str(group["reservoir_name"].iloc[0]),
        "website_gww_id": str(group["website_gww_id"].iloc[0]),
        "n_months": n,
    }
    if n < 2:
        return result

    error = est - obs
    mean_obs = float(np.mean(obs))
    std_obs = float(np.std(obs, ddof=1))
    std_est = float(np.std(est, ddof=1))
    mean_est = float(np.mean(est))
    if std_obs > 0 and std_est > 0:
        r = float(np.corrcoef(obs, est)[0, 1])
    else:
        r = math.nan
    denominator = float(np.sum((obs - mean_obs) ** 2))
    capacity_values = pd.to_numeric(
        group["cwc_live_capacity_mcm"],
        errors="coerce",
    )
    capacity = float(capacity_values.median())

    result.update(
        bias_mcm=float(np.mean(error)),
        mae_mcm=float(np.mean(np.abs(error))),
        rmse_mcm=float(np.sqrt(np.mean(error**2))),
        nrmse_capacity_pct=(
            float(np.sqrt(np.mean(error**2)) / capacity * 100.0)
            if capacity > 0
            else math.nan
        ),
        pbias_pct=(
            float(np.sum(error) / np.sum(obs) * 100.0)
            if np.sum(obs) != 0
            else math.nan
        ),
        pearson_r=r,
        nse=(
            1.0 - float(np.sum(error**2)) / denominator
            if denominator > 0
            else math.nan
        ),
        kge=(
            1.0
            - math.sqrt(
                (r - 1.0) ** 2
                + (std_est / std_obs - 1.0) ** 2
                + (mean_est / mean_obs - 1.0) ** 2
            )
            if np.isfinite(r) and std_obs > 0 and mean_obs != 0
            else math.nan
        ),
    )

    ordered = group.sort_values("month").copy()
    consecutive = ordered["month"].diff().dt.days.between(28, 31)
    ordered["cwc_delta_mcm"] = ordered[obs_col].diff().where(consecutive)
    ordered["website_delta_mcm"] = (
        ordered["website_storage_mcm"].diff().where(consecutive)
    )
    d_obs, d_est = finite_pair(
        ordered["cwc_delta_mcm"],
        ordered["website_delta_mcm"],
    )
    result["n_monthly_changes"] = len(d_obs)
    if len(d_obs) >= 2:
        delta_mse = np.mean((d_est - d_obs) ** 2)
        result["delta_rmse_mcm"] = float(np.sqrt(delta_mse))
        result["delta_pearson_r"] = (
            float(np.corrcoef(d_obs, d_est)[0, 1])
            if np.std(d_obs, ddof=1) > 0 and np.std(d_est, ddof=1) > 0
            else math.nan
        )
    return result


def add_pair_columns(merged: pd.DataFrame, obs_col: str) -> pd.DataFrame:
    merged = merged.sort_values(["reservoir_name", "month"]).copy()
    merged["year"] = merged["month"].dt.year
    merged["season"] = merged["month"].dt.month.map(
        SEASON_BY_MONTH
    )
    merged["cwc_storage_selected_mcm"] = merged[obs_col]
    merged["difference_mcm"] = (
        merged["website_storage_mcm"] - merged["cwc_storage_selected_mcm"]
    )
    merged["absolute_error_mcm"] = merged["difference_mcm"].abs()
    merged["percent_error"] = np.where(
        merged["cwc_storage_selected_mcm"] != 0,
        merged["difference_mcm"] / merged["cwc_storage_selected_mcm"] * 100.0,
        np.nan,
    )
    month_difference = merged.groupby("reservoir_name")["month"].diff()
    consecutive = month_difference.dt.days.between(28, 31)
    merged["cwc_storage_change_mcm"] = (
        merged.groupby("reservoir_name")["cwc_storage_selected_mcm"]
        .diff()
        .where(consecutive)
    )
    merged["website_storage_change_mcm"] = (
        merged.groupby("reservoir_name")["website_storage_mcm"]
        .diff()
        .where(consecutive)
    )
    return merged


def grouped_metrics(
    data: pd.DataFrame,
    obs_col: str,
    group_columns: list[str],
) -> pd.DataFrame:
    records = []
    grouped = data.groupby(group_columns, sort=False)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = calculate_metrics(group, obs_col)
        for column, value in zip(group_columns, keys):
            record[column] = value
        records.append(record)

    result = pd.DataFrame(records)
    if "season" in result.columns:
        result["season_order"] = result["season"].map(
            SEASON_ORDER
        )
        sort_columns = ["reservoir_name"]
        if "year" in result.columns:
            sort_columns.append("year")
        sort_columns.append("season_order")
        result = result.sort_values(sort_columns)
        result = result.drop(columns="season_order")
    else:
        result = result.sort_values("reservoir_name")
    return result


def make_plots(merged: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to use --make-plots") from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for name, group in merged.groupby("reservoir_name", sort=True):
        group = group.sort_values("month")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

        axes[0].plot(
            group["month"],
            group["cwc_storage_selected_mcm"],
            label="CWC",
            lw=2,
        )
        axes[0].plot(
            group["month"],
            group["website_storage_mcm"],
            label="Website",
            lw=2,
        )
        axes[0].set_title(f"{name}: monthly storage")
        axes[0].set_ylabel("Storage (MCM)")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        x = group["cwc_storage_selected_mcm"]
        y = group["website_storage_mcm"]
        axes[1].scatter(x, y, s=25, alpha=0.75)
        low = float(np.nanmin([x.min(), y.min()]))
        high = float(np.nanmax([x.max(), y.max()]))
        axes[1].plot([low, high], [low, high], "k--", lw=1, label="1:1")
        axes[1].set_xlabel("CWC storage (MCM)")
        axes[1].set_ylabel("Website storage (MCM)")
        axes[1].set_title("Agreement")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        clean_name = name.lower().replace("(", " ").replace(")", " ")
        safe_name = "_".join(clean_name.split())
        fig.savefig(
            plot_dir / f"{safe_name}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cwc = load_cwc_monthly(args.cwc_file)
    metadata = load_reservoir_metadata(args.reservoir_json)
    website = load_all_storage_series(
        args.storage_json_dir,
        args.storage_field,
    )
    merged = cwc.merge(
        website,
        on=["reservoir_name", "website_gww_id", "month"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        metadata,
        on="website_gww_id",
        how="left",
        validate="many_to_one",
    )

    if merged.empty:
        raise RuntimeError(
            "No overlapping CWC and website reservoir-month records "
            "were found"
        )

    audit_path = args.output_dir / "matched_monthly_pairs_audit.csv"
    merged.to_csv(audit_path, index=False, date_format="%Y-%m")

    print(f"Matched reservoirs: {merged['reservoir_name'].nunique()}")
    print(f"Matched reservoir-months: {len(merged)}")
    first_month = merged["month"].min()
    last_month = merged["month"].max()
    print(f"Common dates: {first_month:%Y-%m} to {last_month:%Y-%m}")
    print(f"Audit table: {audit_path}")

    if args.website_unit == "unknown":
        print(
            "\nSTOP: website units are unconfirmed, so volume-error "
            "metrics were not calculated.\n"
            "Confirm that the JSON field contains actual storage volume, "
            "then rerun with\n"
            "--website-unit m3 or --website-unit mcm."
        )
        return 0

    if args.website_unit == "m3":
        merged["website_storage_mcm"] = (
            merged["website_value_raw"] / 1_000_000.0
        )
    else:
        merged["website_storage_mcm"] = merged["website_value_raw"]

    obs_col = f"cwc_storage_{args.monthly_stat}_mcm"
    merged = add_pair_columns(merged, obs_col)

    pair_path = args.output_dir / "matched_monthly_validation.csv"
    merged.to_csv(pair_path, index=False, date_format="%Y-%m")

    summary = grouped_metrics(
        merged,
        obs_col,
        ["reservoir_name"],
    )
    summary_path = args.output_dir / "validation_summary_by_reservoir.csv"
    summary.to_csv(summary_path, index=False)

    season_summary = grouped_metrics(
        merged,
        obs_col,
        ["reservoir_name", "season"],
    )
    season_path = (
        args.output_dir
        / "validation_summary_by_reservoir_season.csv"
    )
    season_summary.to_csv(season_path, index=False)

    year_season_summary = grouped_metrics(
        merged,
        obs_col,
        ["reservoir_name", "year", "season"],
    )
    year_season_path = (
        args.output_dir
        / "validation_summary_by_reservoir_year_season.csv"
    )
    year_season_summary.to_csv(year_season_path, index=False)

    if args.make_plots:
        make_plots(merged, args.output_dir)

    print(f"Validation pairs: {pair_path}")
    print(f"Reservoir metrics: {summary_path}")
    print(f"Reservoir-season metrics: {season_path}")
    print(f"Reservoir-year-season metrics: {year_season_path}")
    if args.make_plots:
        print(f"Plots: {args.output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
