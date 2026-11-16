from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEASON_BY_MONTH = {
    1: "Dry", 2: "Dry", 3: "Dry",
    4: "Pre-Monsoon", 5: "Pre-Monsoon", 6: "Pre-Monsoon",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Winter", 11: "Winter", 12: "Winter",
}
SEASON_ORDER = {"Dry": 1, "Pre-Monsoon": 2, "Monsoon": 3, "Winter": 4}
DEFAULT_ROOT = Path(
    "/Users/protikbosepranto/Desktop/Home/Research/Hydrology Website"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate experimental storage against GRS ground truth."
    )
    parser.add_argument(
        "--match-csv",
        type=Path,
        default=DEFAULT_ROOT / "Dataset/gdw_grand_matched.csv",
        help="GDW/GWW/GRanD crosswalk CSV.",
    )
    parser.add_argument(
        "--grs-dir",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "Validation/Global_Reservoir_Storage_dataset"
        ),
        help="Extracted GRS folder containing <GRAND_ID>.csv files.",
    )
    parser.add_argument(
        "--experimental-json-dir",
        type=Path,
        default=DEFAULT_ROOT / "geojson/reservoir_corrected",
        help="Folder containing storage_<gww_id>.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "Validation/validation_output",
    )
    parser.add_argument(
        "--workbook-name",
        default="storage_validation_by_reservoir.xlsx",
        help="Excel workbook containing one worksheet per reservoir.",
    )
    parser.add_argument("--experimental-field", default="Storage",
                        help="Storage array key in each experimental JSON.")
    parser.add_argument(
        "--experimental-unit", choices=("m3", "mcm", "km3"), default="mcm",
        help="Unit of experimental values; default is mcm."
    )
    parser.add_argument("--exclude-interpolated", action="store_true",
                        help="Exclude GRS rows where Interpolation is Y.")
    parser.add_argument("--include-impoundment", action="store_true",
                        help="Include GRS rows marked as Impoundment=Y.")
    parser.add_argument("--make-plots", action="store_true",
                        help="Create one storage time-series plot per reservoir.")
    return parser.parse_args()


def true_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def id_text(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return str(int(number))


def find_column(columns, requested: str) -> str:
    by_lower = {str(col).strip().lower(): col for col in columns}
    key = requested.strip().lower()
    if key not in by_lower:
        raise KeyError(f"Missing column '{requested}'. Available: {list(columns)}")
    return by_lower[key]


def load_matches(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    required = {"gdw_id", "gww_id", "grand_id"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Match CSV is missing columns: {missing}")

    if "grand_match" in table.columns:
        table = table[true_values(table["grand_match"])].copy()
    table["gdw_id"] = table["gdw_id"].map(id_text)
    table["gww_id"] = table["gww_id"].map(id_text)
    table["grand_id"] = table["grand_id"].map(id_text)
    table = table[(table["gww_id"] != "") & (table["grand_id"] != "")].copy()
    if table.empty:
        raise ValueError("No rows with both gww_id and grand_id were found.")

    for col in ("reservoir_name", "dam_name", "country", "capacity_mcm"):
        if col not in table.columns:
            table[col] = np.nan
    table["reservoir_label"] = table["reservoir_name"].fillna(
        table["dam_name"]
    )
    fallback = "GDW " + table["gdw_id"]
    table["reservoir_label"] = table["reservoir_label"].fillna(fallback)
    table["reservoir_label"] = table["reservoir_label"].astype(str).str.strip()
    return table.drop_duplicates(subset=["gdw_id", "gww_id", "grand_id"])


def grs_path_for(row: pd.Series, grs_dir: Path) -> Path:
    filename = str(row.get("grs_file", "")).strip()
    if not filename or filename.lower() == "nan":
        filename = f"{row['grand_id']}.csv"
    direct = grs_dir / filename
    if direct.exists():
        return direct
    matches = list(grs_dir.rglob(filename))
    return matches[0] if len(matches) == 1 else direct


def experimental_path_for(gww_id: str, data_dir: Path) -> Path:
    filename = f"storage_{gww_id}.json"
    direct = data_dir / filename
    if direct.exists():
        return direct
    matches = list(data_dir.rglob(filename))
    return matches[0] if len(matches) == 1 else direct


def load_grs(path: Path, exclude_interpolated: bool,
             include_impoundment: bool) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    month_col = find_column(data.columns, "Month")
    storage_col = find_column(data.columns, "Storage(km3)")
    storage_km3 = pd.to_numeric(data[storage_col], errors="coerce")
    out = pd.DataFrame({
        "month": pd.to_datetime(data[month_col], errors="coerce"),
        "grs_storage_km3": storage_km3,
        "grs_storage_mcm": storage_km3 * 1000.0,
    })
    for source, target in (
        ("Interpolation", "grs_interpolation"),
        ("Impoundment", "grs_impoundment"),
    ):
        try:
            out[target] = data[find_column(data.columns, source)]
        except KeyError:
            out[target] = np.nan

    out["month"] = out["month"].dt.to_period("M").dt.to_timestamp()
    keep = out["month"].notna() & out["grs_storage_mcm"].notna()
    if exclude_interpolated:
        keep &= ~true_values(out["grs_interpolation"])
    if not include_impoundment:
        keep &= ~true_values(out["grs_impoundment"])
    out = out[keep].copy()
    return out.sort_values("month").drop_duplicates("month", keep="last")


def load_experimental(path: Path, field: str, unit: str) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "dates" not in payload:
        raise ValueError("JSON must contain a 'dates' array")
    actual_field = next(
        (key for key in payload if key.lower() == field.lower()), None
    )
    if actual_field is None:
        available = sorted(key for key in payload if key != "dates")
        raise ValueError(f"No '{field}' array; available fields: {available}")
    if len(payload["dates"]) != len(payload[actual_field]):
        raise ValueError("Date/value arrays have different lengths")

    factor = {"m3": 1e-6, "mcm": 1.0, "km3": 1000.0}[unit]
    out = pd.DataFrame({
        "month": pd.to_datetime(payload["dates"], errors="coerce"),
        "experimental_value_raw": pd.to_numeric(
            payload[actual_field], errors="coerce"
        ),
    })
    out["month"] = out["month"].dt.to_period("M").dt.to_timestamp()
    out["experimental_storage_mcm"] = out["experimental_value_raw"] * factor
    out = out.dropna(subset=["month", "experimental_storage_mcm"])
    return out.sort_values("month").drop_duplicates("month", keep="last")


def finite_pair(obs: pd.Series, est: pd.Series):
    obs = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    est = pd.to_numeric(est, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(obs) & np.isfinite(est)
    return obs[keep], est[keep]


def calculate_metrics(group: pd.DataFrame) -> dict:
    obs, est = finite_pair(
        group["grs_storage_mcm"], group["experimental_storage_mcm"]
    )
    first = group.iloc[0]
    result = {
        "reservoir_label": first["reservoir_label"],
        "gdw_id": first["gdw_id"], "gww_id": first["gww_id"],
        "grand_id": first["grand_id"], "n_months": len(obs),
    }
    if len(obs) == 0:
        return result

    error = est - obs
    mean_obs, mean_est = float(np.mean(obs)), float(np.mean(est))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    result.update({
        "grs_mean_storage_mcm": mean_obs,
        "experimental_mean_storage_mcm": mean_est,
        "grs_median_storage_mcm": float(np.median(obs)),
        "experimental_median_storage_mcm": float(np.median(est)),
        "grs_min_storage_mcm": float(np.min(obs)),
        "experimental_min_storage_mcm": float(np.min(est)),
        "grs_max_storage_mcm": float(np.max(obs)),
        "experimental_max_storage_mcm": float(np.max(est)),
        "mean_magnitude_ratio_exp_to_grs": (
            mean_est / mean_obs if mean_obs != 0 else math.nan
        ),
        "mean_magnitude_difference_mcm": mean_est - mean_obs,
        "bias_mcm": float(np.mean(error)),
        "mae_mcm": float(np.mean(np.abs(error))),
        "rmse_mcm": rmse,
        "rmse_pct_of_grs_mean": (
            rmse / abs(mean_obs) * 100 if mean_obs != 0 else math.nan
        ),
        "pbias_pct": (float(np.sum(error) / np.sum(obs) * 100)
                       if np.sum(obs) != 0 else math.nan),
    })

    ordered = group.sort_values("month")
    consecutive = ordered["month"].diff().dt.days.between(28, 31)
    d_obs = ordered["grs_storage_mcm"].diff().where(consecutive)
    d_est = ordered["experimental_storage_mcm"].diff().where(consecutive)
    d_obs, d_est = finite_pair(d_obs, d_est)
    result["n_monthly_changes"] = len(d_obs)
    if len(d_obs) >= 1:
        result["delta_rmse_mcm"] = float(
            np.sqrt(np.mean((d_est - d_obs) ** 2))
        )
    return result


def add_pair_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.sort_values(["gww_id", "month"]).copy()
    pairs["year"] = pairs["month"].dt.year
    pairs["season"] = pairs["month"].dt.month.map(SEASON_BY_MONTH)
    pairs["difference_mcm"] = (
        pairs["experimental_storage_mcm"] - pairs["grs_storage_mcm"]
    )
    pairs["absolute_error_mcm"] = pairs["difference_mcm"].abs()
    pairs["percent_error"] = np.where(
        pairs["grs_storage_mcm"] != 0,
        pairs["difference_mcm"] / pairs["grs_storage_mcm"] * 100,
        np.nan,
    )
    return pairs


def grouped_metrics(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(groups, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = calculate_metrics(group)
        row.update(dict(zip(groups, keys)))
        rows.append(row)
    result = pd.DataFrame(rows)
    if "season" in result.columns:
        result["_season"] = result["season"].map(SEASON_ORDER)
        result = result.sort_values(groups[:-1] + ["_season"])
        result = result.drop(columns="_season")
    return result


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "reservoir"


def excel_sheet_name(label: str, gww_id: str, used: set[str]) -> str:
    """Return a valid and unique Excel worksheet name."""
    raw_name = f"{label}_{gww_id}"
    clean = re.sub(r"[\\/*?:\[\]]", "_", raw_name)
    clean = re.sub(r"\s+", " ", clean).strip(" '")
    base = clean[:31] or f"GWW_{gww_id}"
    candidate = base
    counter = 2

    while candidate.casefold() in used:
        suffix = f"_{counter}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        counter += 1

    used.add(candidate.casefold())
    return candidate


def format_simple_table_sheet(worksheet) -> None:
    """Apply restrained formatting to Summary and Audit worksheets."""
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for column_number in range(1, worksheet.max_column + 1):
        letter = get_column_letter(column_number)
        values = (
            worksheet.cell(row_number, column_number).value
            for row_number in range(1, worksheet.max_row + 1)
        )
        width = max((len(str(value)) for value in values if value is not None),
                    default=10)
        worksheet.column_dimensions[letter].width = min(max(width + 2, 10), 35)


def write_reservoir_workbook(
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write summary tabs and one detailed worksheet per reservoir."""
    try:
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            "Creating the Excel workbook requires openpyxl. "
            "Install it with: pip install openpyxl"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    used_names = {"summary", "audit"}
    id_columns = ["reservoir_label", "gdw_id", "gww_id", "grand_id"]
    monthly_columns = [
        "month",
        "year",
        "season",
        "grs_storage_km3",
        "grs_storage_mcm",
        "experimental_value_raw",
        "experimental_storage_mcm",
        "difference_mcm",
        "absolute_error_mcm",
        "percent_error",
        "grs_interpolation",
        "grs_impoundment",
    ]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        audit.to_excel(writer, sheet_name="Audit", index=False)

        for _, group in pairs.groupby(
            ["gdw_id", "gww_id", "grand_id"],
            sort=False,
        ):
            group = group.sort_values("month").copy()
            first = group.iloc[0]
            sheet_name = excel_sheet_name(
                str(first["reservoir_label"]),
                str(first["gww_id"]),
                used_names,
            )

            metric_match = summary[
                (summary["gdw_id"] == first["gdw_id"])
                & (summary["gww_id"] == first["gww_id"])
                & (summary["grand_id"] == first["grand_id"])
            ]
            metric_row = metric_match.iloc[0] if not metric_match.empty else None

            information = [
                ("reservoir_label", first["reservoir_label"]),
                ("reservoir_name", first.get("reservoir_name", np.nan)),
                ("dam_name", first.get("dam_name", np.nan)),
                ("country", first.get("country", np.nan)),
                ("gdw_id", first["gdw_id"]),
                ("gww_id", first["gww_id"]),
                ("grand_id", first["grand_id"]),
                ("capacity_mcm", first.get("capacity_mcm", np.nan)),
            ]
            if metric_row is not None:
                for column in summary.columns:
                    if column not in id_columns:
                        information.append((column, metric_row[column]))

            information_frame = pd.DataFrame(
                information,
                columns=["reservoir_information", "value"],
            )
            information_frame.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=0,
            )

            available_columns = [
                column for column in monthly_columns if column in group.columns
            ]
            monthly_frame = group[available_columns]
            data_startrow = len(information_frame) + 3
            monthly_frame.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=data_startrow,
            )

            worksheet = writer.sheets[sheet_name]
            header_fill = PatternFill("solid", fgColor="D9EAF7")
            data_header_row = data_startrow + 1
            for header_row in (1, data_header_row):
                for cell in worksheet[header_row]:
                    cell.font = Font(bold=True)
                    cell.fill = header_fill

            worksheet.freeze_panes = f"A{data_header_row + 1}"
            last_column = get_column_letter(len(available_columns))
            last_data_row = data_header_row + len(monthly_frame)
            worksheet.auto_filter.ref = (
                f"A{data_header_row}:{last_column}{last_data_row}"
            )

            if "month" in available_columns:
                month_column = available_columns.index("month") + 1
                for row_number in range(data_header_row + 1,
                                        last_data_row + 1):
                    worksheet.cell(row_number, month_column).number_format = (
                        "mmm yyyy"
                    )

            for column_number in range(1, worksheet.max_column + 1):
                letter = get_column_letter(column_number)
                sample_end = min(worksheet.max_row, data_header_row + 30)
                values = (
                    worksheet.cell(row_number, column_number).value
                    for row_number in range(1, sample_end + 1)
                )
                width = max(
                    (len(str(value)) for value in values if value is not None),
                    default=10,
                )
                worksheet.column_dimensions[letter].width = min(
                    max(width + 2, 11),
                    32,
                )

        format_simple_table_sheet(writer.sheets["Summary"])
        format_simple_table_sheet(writer.sheets["Audit"])


def make_plots(pairs: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    keys = ["gdw_id", "gww_id", "grand_id"]
    for _, group in pairs.groupby(keys, sort=False):
        group = group.sort_values("month")
        first = group.iloc[0]
        label = str(first["reservoir_label"])
        fig, axis = plt.subplots(figsize=(11, 5))
        axis.plot(
            group["month"],
            group["grs_storage_mcm"],
            linewidth=1.8,
            label="GRS",
        )
        axis.plot(
            group["month"],
            group["experimental_storage_mcm"],
            linewidth=1.5,
            label="Experimental",
        )
        axis.set_xlabel("Month")
        axis.set_ylabel("Storage (MCM)")
        axis.set_title(
            f"{label} | GWW {first['gww_id']} | GRanD {first['grand_id']}"
        )
        axis.legend()
        axis.grid(alpha=0.3)
        fig.tight_layout()
        filename = f"{safe_name(label)}_gww_{first['gww_id']}.png"
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches = load_matches(args.match_csv)
    audits, pair_frames = [], []

    for _, row in matches.iterrows():
        grs_path = grs_path_for(row, args.grs_dir)
        exp_path = experimental_path_for(row["gww_id"],
                                         args.experimental_json_dir)
        audit = {
            "reservoir_label": row["reservoir_label"],
            "gdw_id": row["gdw_id"], "gww_id": row["gww_id"],
            "grand_id": row["grand_id"], "grs_path": str(grs_path),
            "experimental_path": str(exp_path),
            "grs_file_found": grs_path.exists(),
            "experimental_file_found": exp_path.exists(),
            "status": "pending", "n_grs_months": 0,
            "n_experimental_months": 0, "n_common_months": 0,
        }
        if not grs_path.exists() or not exp_path.exists():
            audit["status"] = "missing_grs" if not grs_path.exists() else "missing_experimental"
            if not grs_path.exists() and not exp_path.exists():
                audit["status"] = "missing_both"
            audits.append(audit)
            continue
        try:
            grs = load_grs(grs_path, args.exclude_interpolated,
                           args.include_impoundment)
            exp = load_experimental(exp_path, args.experimental_field,
                                    args.experimental_unit)
            merged = grs.merge(exp, on="month", how="inner")
            audit.update(n_grs_months=len(grs), n_experimental_months=len(exp),
                         n_common_months=len(merged))
            if merged.empty:
                audit["status"] = "no_common_months"
            else:
                audit["status"] = "validated"
                for col in (
                    "reservoir_label", "gdw_id", "gww_id", "grand_id",
                    "reservoir_name", "dam_name", "country", "capacity_mcm",
                ):
                    merged[col] = row.get(col, np.nan)
                pair_frames.append(merged)
        except Exception as exc:
            audit["status"] = "read_error"
            audit["error"] = str(exc)
        audits.append(audit)

    audit_path = args.output_dir / "file_match_audit.csv"
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(audit_path, index=False)
    if not pair_frames:
        raise ValueError(f"No comparable series found. Check {audit_path}")

    pairs = add_pair_columns(pd.concat(pair_frames, ignore_index=True))
    pair_path = args.output_dir / "monthly_storage_validation.csv"
    pairs.to_csv(pair_path, index=False)
    ids = ["reservoir_label", "gdw_id", "gww_id", "grand_id"]
    summary = grouped_metrics(pairs, ids)
    summary_path = args.output_dir / "validation_summary_by_reservoir.csv"
    summary.to_csv(summary_path, index=False)
    season_path = args.output_dir / "validation_summary_by_reservoir_season.csv"
    grouped_metrics(pairs, ids + ["season"]).to_csv(season_path, index=False)
    year_season_path = (
        args.output_dir / "validation_summary_by_reservoir_year_season.csv"
    )
    grouped_metrics(pairs, ids + ["year", "season"]).to_csv(
        year_season_path, index=False
    )
    workbook_path = args.output_dir / args.workbook_name
    write_reservoir_workbook(
        pairs,
        summary,
        audit_frame,
        workbook_path,
    )
    if args.make_plots:
        make_plots(pairs, args.output_dir)

    print(f"Matched crosswalk rows: {len(matches)}")
    print(f"Validated series: {len(summary)}")
    print(f"Matched monthly pairs: {len(pairs)}")
    print(f"Audit: {audit_path}")
    print(f"Monthly pairs: {pair_path}")
    print(f"Reservoir metrics: {summary_path}")
    print(f"Season metrics: {season_path}")
    print(f"Year-season metrics: {year_season_path}")
    print(f"Reservoir workbook: {workbook_path}")
    if args.make_plots:
        print(f"Plots: {args.output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)