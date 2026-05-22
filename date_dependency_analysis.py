"""Analyze whether Lotto Plus outcomes depend meaningfully on the draw date."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lotto_pipeline import MAIN_COLUMNS, TicketPredictor, infer_powerball_numbers, load_draws, score_predictions

REGULAR_WEEKDAYS = ("Wednesday", "Saturday")
SUMMARY_FEATURES = ("main_sum", "odd_count", "spread", "repeat_from_prev", "PB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws-path", default="draws.csv", help="Historical draw CSV.")
    parser.add_argument("--artifact-dir", default="artifacts", help="Output directory for reports.")
    parser.add_argument(
        "--holdout-draws",
        type=int,
        default=104,
        help="Number of latest stable-era draws to keep as a chronological holdout.",
    )
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    analysis = df.copy()
    analysis["weekday"] = analysis["Date"].dt.day_name()
    analysis["month"] = analysis["Date"].dt.month
    analysis["quarter"] = analysis["Date"].dt.quarter
    analysis["year"] = analysis["Date"].dt.year
    analysis["day_of_year"] = analysis["Date"].dt.dayofyear
    analysis["draw_index"] = np.arange(len(analysis), dtype=int)
    analysis["main_sum"] = analysis[MAIN_COLUMNS].sum(axis=1)
    analysis["odd_count"] = analysis[MAIN_COLUMNS].apply(lambda row: int(sum(int(value) % 2 for value in row)), axis=1)
    analysis["spread"] = analysis["5"] - analysis["1"]
    analysis["main_max"] = analysis[MAIN_COLUMNS].max(axis=1)
    analysis["has_36"] = (analysis[MAIN_COLUMNS] == 36).any(axis=1)

    previous_sets: list[set[int] | None] = [None]
    previous_sets.extend(set(map(int, row)) for row in analysis[MAIN_COLUMNS].iloc[:-1].to_numpy())
    repeat_from_previous: list[int] = []
    for current_row, previous_row in zip(analysis[MAIN_COLUMNS].to_numpy(), previous_sets):
        current_set = set(map(int, current_row))
        if previous_row is None:
            repeat_from_previous.append(0)
        else:
            repeat_from_previous.append(len(current_set & previous_row))
    analysis["repeat_from_prev"] = repeat_from_previous
    return analysis


def benjamini_hochberg(records: list[dict[str, object]], pvalue_key: str = "pvalue") -> list[dict[str, object]]:
    if not records:
        return records

    ranked = sorted(records, key=lambda row: float(row[pvalue_key]))
    total = len(ranked)
    adjusted = 1.0

    for reverse_rank, row in enumerate(reversed(ranked), start=1):
        rank = total - reverse_rank + 1
        raw_pvalue = float(row[pvalue_key])
        adjusted = min(adjusted, raw_pvalue * total / rank)
        row["qvalue"] = adjusted
        row["significant_fdr_05"] = adjusted <= 0.05

    return ranked


def autocorrelation(series: pd.Series, max_lag: int = 52) -> dict[str, object]:
    values = series.to_numpy(dtype=float)
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return {"significance_bound": None, "top_lags": []}

    bound = 1.96 / np.sqrt(len(values))
    lag_rows: list[dict[str, float]] = []
    for lag in range(1, min(max_lag, len(values) - 1) + 1):
        numerator = float(np.dot(centered[:-lag], centered[lag:]))
        acf_value = numerator / denominator
        lag_rows.append(
            {
                "lag": lag,
                "acf": acf_value,
                "significant": abs(acf_value) > bound,
            }
        )

    lag_rows.sort(key=lambda row: abs(float(row["acf"])), reverse=True)
    return {
        "significance_bound": float(bound),
        "top_lags": lag_rows[:5],
    }


def structural_change_summary(df: pd.DataFrame) -> dict[str, object]:
    last_seen_by_number: dict[int, str] = {}
    historical_support = sorted({int(value) for value in df[MAIN_COLUMNS].to_numpy().reshape(-1)})
    for number in historical_support:
        mask = (df[MAIN_COLUMNS] == number).any(axis=1)
        last_seen_by_number[number] = df.loc[mask, "Date"].max().strftime("%Y-%m-%d")

    has_36_mask = (df[MAIN_COLUMNS] == 36).any(axis=1)
    last_36_date = df.loc[has_36_mask, "Date"].max() if has_36_mask.any() else None
    stable_start = None if last_36_date is None else last_36_date + pd.Timedelta(days=1)

    yearly_max = (
        df.assign(year=df["Date"].dt.year)
        .groupby("year")[MAIN_COLUMNS]
        .max()
        .max(axis=1)
        .astype(int)
        .to_dict()
    )

    return {
        "historical_main_support_max": int(df[MAIN_COLUMNS].max().max()),
        "recent_main_support_max": int(df.tail(500)[MAIN_COLUMNS].max().max()),
        "last_occurrence_by_number": {str(key): value for key, value in last_seen_by_number.items()},
        "last_occurrence_of_36": None if last_36_date is None else last_36_date.strftime("%Y-%m-%d"),
        "stable_era_start": None if stable_start is None else stable_start.strftime("%Y-%m-%d"),
        "yearly_main_support_max": {str(key): int(value) for key, value in yearly_max.items()},
    }


def summary_tests(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    regular = df[df["weekday"].isin(REGULAR_WEEKDAYS)].copy()
    weekday_rows: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []

    for feature in SUMMARY_FEATURES:
        wednesday_values = regular.loc[regular["weekday"] == "Wednesday", feature]
        saturday_values = regular.loc[regular["weekday"] == "Saturday", feature]
        statistic, pvalue = mannwhitneyu(wednesday_values, saturday_values, alternative="two-sided")
        weekday_rows.append(
            {
                "feature": feature,
                "weekday_a": "Wednesday",
                "weekday_b": "Saturday",
                "weekday_a_mean": float(wednesday_values.mean()),
                "weekday_b_mean": float(saturday_values.mean()),
                "mean_difference": float(wednesday_values.mean() - saturday_values.mean()),
                "statistic": float(statistic),
                "pvalue": float(pvalue),
            }
        )

        month_groups = [group[feature].to_numpy(dtype=float) for _, group in df.groupby("month")]
        statistic, pvalue = kruskal(*month_groups)
        monthly_means = df.groupby("month")[feature].mean().round(4).to_dict()
        month_rows.append(
            {
                "feature": feature,
                "monthly_means": {str(key): float(value) for key, value in monthly_means.items()},
                "monthly_mean_range": float(max(monthly_means.values()) - min(monthly_means.values())),
                "statistic": float(statistic),
                "pvalue": float(pvalue),
            }
        )

    return {
        "weekday_regular_schedule": benjamini_hochberg(weekday_rows),
        "month_of_year": benjamini_hochberg(month_rows),
    }


def number_dependency_tests(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    regular = df[df["weekday"].isin(REGULAR_WEEKDAYS)].copy()
    month_rows: list[dict[str, object]] = []
    weekday_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []

    main_support = sorted({int(value) for value in df[MAIN_COLUMNS].to_numpy().reshape(-1)})
    powerball_support = infer_powerball_numbers(df)

    def run_indicator_tests(label_prefix: str, occurrence_frame: pd.DataFrame) -> None:
        for number in occurrence_frame.columns:
            number_label = f"{label_prefix}_{number}"
            occurrence = occurrence_frame[number].astype(int)

            month_table = pd.crosstab(df["month"], occurrence)
            if month_table.shape[1] == 2:
                chi2, pvalue, _, _ = chi2_contingency(month_table)
                month_rows.append(
                    {
                        "label": number_label,
                        "occurrence_rate": float(occurrence.mean()),
                        "statistic": float(chi2),
                        "pvalue": float(pvalue),
                    }
                )

            regular_occurrence = occurrence.loc[regular.index]
            weekday_table = pd.crosstab(regular["weekday"], regular_occurrence)
            if weekday_table.shape[1] == 2:
                chi2, pvalue, _, _ = chi2_contingency(weekday_table)
                weekday_rows.append(
                    {
                        "label": number_label,
                        "occurrence_rate": float(regular_occurrence.mean()),
                        "statistic": float(chi2),
                        "pvalue": float(pvalue),
                    }
                )

            correlation, pvalue = spearmanr(df["draw_index"], occurrence)
            trend_rows.append(
                {
                    "label": number_label,
                    "occurrence_rate": float(occurrence.mean()),
                    "correlation": float(0.0 if np.isnan(correlation) else correlation),
                    "pvalue": float(pvalue),
                }
            )

    main_occurrence = pd.DataFrame(
        {number: (df[MAIN_COLUMNS] == number).any(axis=1) for number in main_support},
        index=df.index,
    )
    powerball_occurrence = pd.DataFrame(
        {number: (df["PB"] == number) for number in powerball_support},
        index=df.index,
    )

    run_indicator_tests("main", main_occurrence)
    run_indicator_tests("pb", powerball_occurrence)

    return {
        "month_of_year": benjamini_hochberg(month_rows),
        "weekday_regular_schedule": benjamini_hochberg(weekday_rows),
        "time_trend": benjamini_hochberg(trend_rows),
    }


def build_date_only_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["year"] = df["Date"].dt.year.astype(float)
    features["quarter"] = df["Date"].dt.quarter.astype(float)
    features["month_sin"] = np.sin(2.0 * np.pi * df["Date"].dt.month / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * df["Date"].dt.month / 12.0)
    features["day_of_year_sin"] = np.sin(2.0 * np.pi * df["Date"].dt.dayofyear / 366.0)
    features["day_of_year_cos"] = np.cos(2.0 * np.pi * df["Date"].dt.dayofyear / 366.0)
    features["is_wednesday"] = (df["Date"].dt.day_name() == "Wednesday").astype(float)
    features["is_saturday"] = (df["Date"].dt.day_name() == "Saturday").astype(float)
    features["is_rescheduled"] = (~df["Date"].dt.day_name().isin(REGULAR_WEEKDAYS)).astype(float)
    return features


def random_ticket_expectation(main_support_size: int, powerball_support_size: int) -> dict[str, float]:
    expected_main_hits = 25.0 / float(main_support_size)
    expected_powerball_hits = 1.0 / float(powerball_support_size)
    return {
        "mean_main_hits": expected_main_hits,
        "powerball_accuracy": expected_powerball_hits,
        "mean_total_hits": expected_main_hits + expected_powerball_hits,
    }


def date_only_model_analysis(df: pd.DataFrame, holdout_draws: int) -> dict[str, object]:
    feature_frame = build_date_only_features(df)
    y = df[[*MAIN_COLUMNS, "PB"]].to_numpy(dtype=int)

    if len(feature_frame) <= holdout_draws:
        raise ValueError("Not enough rows for the requested holdout size.")

    train_end = len(feature_frame) - holdout_draws
    X_train = feature_frame.iloc[:train_end]
    X_holdout = feature_frame.iloc[train_end:]
    y_train = y[:train_end]
    y_holdout = y[train_end:]

    estimator = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
        ]
    )
    predictor = TicketPredictor(estimator=estimator, estimator_name="date_only_knn")
    predictor.fit(X_train, y_train)

    allowed_main_numbers = sorted({int(value) for value in df[MAIN_COLUMNS].to_numpy().reshape(-1)})
    allowed_powerball_numbers = infer_powerball_numbers(df)
    predictions = predictor.predict(
        X_holdout,
        allowed_main_numbers=allowed_main_numbers,
        allowed_powerball_numbers=allowed_powerball_numbers,
    )

    metrics = score_predictions(y_holdout, predictions)
    return {
        "holdout_draws": holdout_draws,
        "holdout_start_date": df.iloc[train_end]["Date"].strftime("%Y-%m-%d"),
        "holdout_end_date": df.iloc[-1]["Date"].strftime("%Y-%m-%d"),
        "metrics": metrics,
        "random_ticket_expectation": random_ticket_expectation(
            main_support_size=max(allowed_main_numbers),
            powerball_support_size=max(allowed_powerball_numbers),
        ),
    }


def frequency_baseline_analysis(df: pd.DataFrame, holdout_draws: int) -> dict[str, object]:
    if len(df) <= holdout_draws:
        raise ValueError("Not enough rows for the requested holdout size.")

    train_end = len(df) - holdout_draws
    train = df.iloc[:train_end]
    holdout = df.iloc[train_end:]

    main_ticket = sorted(
        pd.Series(train[MAIN_COLUMNS].to_numpy().reshape(-1))
        .value_counts()
        .head(5)
        .index.astype(int)
        .tolist()
    )
    powerball = int(train["PB"].value_counts().idxmax())

    predictions = np.tile(np.asarray(main_ticket + [powerball], dtype=int), (len(holdout), 1))
    metrics = score_predictions(holdout[[*MAIN_COLUMNS, "PB"]].to_numpy(dtype=int), predictions)
    return {
        "holdout_draws": holdout_draws,
        "holdout_start_date": holdout.iloc[0]["Date"].strftime("%Y-%m-%d"),
        "holdout_end_date": holdout.iloc[-1]["Date"].strftime("%Y-%m-%d"),
        "ticket": {"main_numbers": main_ticket, "powerball": powerball},
        "metrics": metrics,
    }


def build_report_lines(results: dict[str, object]) -> list[str]:
    structural = results["structural_changes"]
    weekday_tests = results["stable_era_summary_tests"]["weekday_regular_schedule"]
    month_tests = results["stable_era_summary_tests"]["month_of_year"]
    number_month = results["stable_era_number_tests"]["month_of_year"]
    number_weekday = results["stable_era_number_tests"]["weekday_regular_schedule"]
    number_trend = results["stable_era_number_tests"]["time_trend"]
    date_only = results["stable_era_date_only_model"]
    frequency_baseline = results["stable_era_frequency_baseline"]

    significant_month = [row["label"] for row in number_month if row["significant_fdr_05"]]
    significant_weekday = [row["label"] for row in number_weekday if row["significant_fdr_05"]]
    significant_trend = [row["label"] for row in number_trend if row["significant_fdr_05"]]

    lines = [
        "# Date Dependency Analysis",
        "",
        f"- Rows analyzed: {results['dataset']['rows']}",
        f"- Date range: {results['dataset']['start_date']} to {results['dataset']['end_date']}",
        f"- Stable era start: {structural['stable_era_start']}",
        f"- Historical main-number support max: {structural['historical_main_support_max']}",
        f"- Recent main-number support max: {structural['recent_main_support_max']}",
        f"- Last observed `36`: {structural['last_occurrence_of_36']}",
        "",
        "## Key Findings",
        "",
        "- The clearest date dependency is structural, not predictive: main number `36` appears historically but disappears after September 2012.",
        f"- Stable-era month test significant features after FDR: {', '.join(row['feature'] for row in month_tests if row['significant_fdr_05']) or 'none'}.",
        f"- Stable-era Wednesday vs Saturday test significant features after FDR: {', '.join(row['feature'] for row in weekday_tests if row['significant_fdr_05']) or 'none'}.",
        f"- Stable-era individual number month dependencies after FDR: {', '.join(significant_month) or 'none'}.",
        f"- Stable-era individual number weekday dependencies after FDR: {', '.join(significant_weekday) or 'none'}.",
        f"- Stable-era individual number time trends after FDR: {', '.join(significant_trend) or 'none'}.",
        "",
        "## Date-Only Holdout Model",
        "",
        f"- Holdout window: {date_only['holdout_start_date']} to {date_only['holdout_end_date']} ({date_only['holdout_draws']} draws)",
        f"- Date-only mean_main_hits: {date_only['metrics']['mean_main_hits']:.4f}",
        f"- Date-only powerball_accuracy: {date_only['metrics']['powerball_accuracy']:.4f}",
        f"- Date-only mean_total_hits: {date_only['metrics']['mean_total_hits']:.4f}",
        f"- Static frequency-baseline mean_total_hits: {frequency_baseline['metrics']['mean_total_hits']:.4f}",
        f"- Random-ticket expected mean_total_hits: {date_only['random_ticket_expectation']['mean_total_hits']:.4f}",
    ]
    return lines


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_analysis_frame(load_draws(args.draws_path))
    structural = structural_change_summary(df)
    stable_start = pd.Timestamp(structural["stable_era_start"])
    stable_df = df[df["Date"] >= stable_start].reset_index(drop=True)

    results = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "rows": int(len(df)),
            "start_date": df["Date"].min().strftime("%Y-%m-%d"),
            "end_date": df["Date"].max().strftime("%Y-%m-%d"),
            "weekday_counts": {str(key): int(value) for key, value in df["weekday"].value_counts().sort_index().to_dict().items()},
        },
        "structural_changes": structural,
        "full_history_autocorrelation": {
            feature: autocorrelation(df[feature]) for feature in SUMMARY_FEATURES
        },
        "stable_era_summary_tests": summary_tests(stable_df),
        "stable_era_number_tests": number_dependency_tests(stable_df),
        "stable_era_date_only_model": date_only_model_analysis(stable_df, holdout_draws=args.holdout_draws),
        "stable_era_frequency_baseline": frequency_baseline_analysis(stable_df, holdout_draws=args.holdout_draws),
    }

    json_path = artifact_dir / "date_dependency_summary.json"
    json_path.write_text(json.dumps(results, indent=2, default=json_default), encoding="utf-8")

    report_path = artifact_dir / "date_dependency_report.md"
    report_path.write_text("\n".join(build_report_lines(results)) + "\n", encoding="utf-8")

    print(f"Saved JSON summary to {json_path}")
    print(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()
