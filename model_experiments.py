"""Benchmark multiple multiclass models and save the best ticket predictor."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lotto_pipeline import (
    RANDOM_STATE,
    TicketPredictor,
    build_supervised_dataset,
    infer_current_main_numbers,
    infer_powerball_numbers,
    load_draws,
    save_model_bundle,
    score_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws-path", default="draws.csv", help="Historical draw CSV.")
    parser.add_argument("--artifact-dir", default="artifacts", help="Where to save model outputs.")
    parser.add_argument(
        "--holdout-draws",
        type=int,
        default=104,
        help="Number of latest draws to reserve as the final holdout set.",
    )
    parser.add_argument("--cv-splits", type=int, default=3, help="Number of time-series CV folds.")
    return parser.parse_args()


def model_specs() -> list[tuple[str, object]]:
    return [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=600,
                            C=0.5,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "knn",
            Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=10,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_depth=5,
                learning_rate=0.05,
                max_iter=120,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)

    df = load_draws(args.draws_path)
    X, y, reference = build_supervised_dataset(df)
    current_main_numbers = infer_current_main_numbers(df)
    powerball_numbers = infer_powerball_numbers(df)

    if len(X) <= args.holdout_draws:
        raise ValueError("Not enough samples for the requested holdout size.")

    train_val_end = len(X) - args.holdout_draws
    X_train_val = X.iloc[:train_val_end].reset_index(drop=True)
    y_train_val = y[:train_val_end]
    X_holdout = X.iloc[train_val_end:].reset_index(drop=True)
    y_holdout = y[train_val_end:]
    holdout_reference = reference.iloc[train_val_end:].reset_index(drop=True)

    cv = TimeSeriesSplit(n_splits=args.cv_splits)
    benchmark_rows: list[dict[str, float | str | int]] = []

    for model_name, estimator in model_specs():
        print(f"Evaluating {model_name}...", flush=True)
        fold_metrics: list[dict[str, float]] = []

        for fold_number, (train_index, test_index) in enumerate(cv.split(X_train_val), start=1):
            predictor = TicketPredictor(estimator=estimator, estimator_name=model_name)
            predictor.fit(X_train_val.iloc[train_index], y_train_val[train_index])
            predictions = predictor.predict(X_train_val.iloc[test_index])
            metrics = score_predictions(y_train_val[test_index], predictions)
            metrics["fold"] = fold_number
            fold_metrics.append(metrics)

        mean_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        benchmark_row: dict[str, float | str | int] = {
            "model": model_name,
            "cv_folds": args.cv_splits,
        }
        benchmark_row.update({f"cv_{key}": value for key, value in mean_metrics.items() if key != "fold"})
        benchmark_rows.append(benchmark_row)
        print(
            f"  mean_total_hits={benchmark_row['cv_mean_total_hits']:.4f}, "
            f"mean_main_hits={benchmark_row['cv_mean_main_hits']:.4f}, "
            f"powerball_accuracy={benchmark_row['cv_powerball_accuracy']:.4f}"
        , flush=True)

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
        ["cv_mean_total_hits", "cv_powerball_accuracy", "cv_mean_main_hits"],
        ascending=False,
    )
    best_model_name = str(benchmark_df.iloc[0]["model"])
    best_estimator = dict(model_specs())[best_model_name]

    best_train_val_predictor = TicketPredictor(estimator=best_estimator, estimator_name=best_model_name)
    best_train_val_predictor.fit(X_train_val, y_train_val)
    holdout_predictions = best_train_val_predictor.predict(
        X_holdout,
        allowed_main_numbers=current_main_numbers,
        allowed_powerball_numbers=powerball_numbers,
    )
    holdout_metrics = score_predictions(y_holdout, holdout_predictions)

    final_predictor = TicketPredictor(estimator=best_estimator, estimator_name=best_model_name)
    final_predictor.fit(X, y)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = artifact_dir / "model_benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    holdout_predictions_path = artifact_dir / "holdout_predictions.csv"
    holdout_output = holdout_reference.copy()
    holdout_output["pred_1"] = holdout_predictions[:, 0]
    holdout_output["pred_2"] = holdout_predictions[:, 1]
    holdout_output["pred_3"] = holdout_predictions[:, 2]
    holdout_output["pred_4"] = holdout_predictions[:, 3]
    holdout_output["pred_5"] = holdout_predictions[:, 4]
    holdout_output["pred_pb"] = holdout_predictions[:, 5]
    holdout_output.to_csv(holdout_predictions_path, index=False)

    bundle = {
        "model_name": best_model_name,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "feature_columns": list(X.columns),
        "allowed_main_numbers": current_main_numbers,
        "allowed_powerball_numbers": powerball_numbers,
        "holdout_metrics": holdout_metrics,
        "cv_results": benchmark_df.to_dict(orient="records"),
        "model": final_predictor,
    }
    model_path = artifact_dir / "best_ticket_predictor.joblib"
    save_model_bundle(bundle, model_path)

    summary_path = artifact_dir / "best_model_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "holdout_metrics": holdout_metrics,
                "benchmark_path": str(benchmark_path),
                "model_path": str(model_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nBenchmark complete.", flush=True)
    print(f"Best model: {best_model_name}", flush=True)
    print(
        "Holdout metrics: "
        f"mean_total_hits={holdout_metrics['mean_total_hits']:.4f}, "
        f"mean_main_hits={holdout_metrics['mean_main_hits']:.4f}, "
        f"powerball_accuracy={holdout_metrics['powerball_accuracy']:.4f}, "
        f"exact_match_rate={holdout_metrics['exact_match_rate']:.4f}"
    , flush=True)
    print(f"Saved benchmark table to {benchmark_path}", flush=True)
    print(f"Saved trained model to {model_path}", flush=True)


if __name__ == "__main__":
    main()
