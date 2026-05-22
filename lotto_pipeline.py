"""Shared data loading, feature engineering, evaluation, and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

MAIN_COLUMNS = ["1", "2", "3", "4", "5"]
TARGET_COLUMNS = MAIN_COLUMNS + ["PB"]
DRAW_DATE_FORMAT = "%d-%b-%y"
FEATURE_WINDOWS = (5, 10, 20, 50)
LAG_DRAWS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class DrawContext:
    draw_number: int
    draw_date: pd.Timestamp
    jackpot: float


def load_draws(csv_path: str | Path = "draws.csv") -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing draw history file: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(
            f"{csv_path} is empty. Run `python scrapehistory.py` before training or predicting."
        )

    numeric_columns = ["Draw Number", "Jackpot", "PB", "Multiplier", "Winners", *MAIN_COLUMNS]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], format=DRAW_DATE_FORMAT)
    df = df.sort_values("Draw Number").drop_duplicates("Draw Number").reset_index(drop=True)
    return df


def infer_current_main_numbers(df: pd.DataFrame, recent_draws: int = 500) -> list[int]:
    recent_max = int(df.tail(recent_draws)[MAIN_COLUMNS].max().max())
    return list(range(1, recent_max + 1))


def infer_powerball_numbers(df: pd.DataFrame) -> list[int]:
    max_value = int(df["PB"].max())
    return list(range(1, max_value + 1))


def infer_next_draw_date(latest_draw_date: pd.Timestamp) -> pd.Timestamp:
    weekday = latest_draw_date.dayofweek
    if weekday == 2:  # Wednesday -> Saturday
        return latest_draw_date + timedelta(days=3)
    if weekday == 5:  # Saturday -> Wednesday
        return latest_draw_date + timedelta(days=4)
    return latest_draw_date + timedelta(days=3)


def build_draw_context(
    df: pd.DataFrame,
    draw_number: int | None = None,
    draw_date: str | pd.Timestamp | None = None,
    jackpot: float | None = None,
) -> DrawContext:
    latest = df.iloc[-1]
    next_draw_number = int(draw_number or int(latest["Draw Number"]) + 1)
    next_draw_date = (
        pd.to_datetime(draw_date, format=DRAW_DATE_FORMAT)
        if isinstance(draw_date, str)
        else draw_date
    )
    if next_draw_date is None:
        next_draw_date = infer_next_draw_date(pd.Timestamp(latest["Date"]))

    jackpot_series = df["Jackpot"].dropna() if "Jackpot" in df.columns else pd.Series(dtype=float)
    latest_jackpot = float(jackpot_series.iloc[-1]) if not jackpot_series.empty else 0.0
    return DrawContext(
        draw_number=next_draw_number,
        draw_date=pd.Timestamp(next_draw_date),
        jackpot=float(jackpot if jackpot is not None else latest_jackpot),
    )


def _history_feature_dict(
    main_history: np.ndarray,
    powerball_history: np.ndarray,
    context: DrawContext,
    last_seen_main: dict[int, int],
    last_seen_powerball: dict[int, int],
    main_candidates: Iterable[int],
    powerball_candidates: Iterable[int],
    sample_index: int,
) -> dict[str, float]:
    feature_row: dict[str, float] = {}
    main_candidates = list(main_candidates)
    powerball_candidates = list(powerball_candidates)

    feature_row["draw_number"] = float(context.draw_number)

    for lag in range(1, LAG_DRAWS + 1):
        lag_main = main_history[-lag]
        lag_powerball = powerball_history[-lag]
        for index, value in enumerate(lag_main, start=1):
            feature_row[f"lag_{lag}_main_{index}"] = float(value)
        feature_row[f"lag_{lag}_pb"] = float(lag_powerball)
        feature_row[f"lag_{lag}_sum"] = float(np.sum(lag_main))

    for window in FEATURE_WINDOWS:
        recent_main = main_history[-window:].reshape(-1)
        recent_powerball = powerball_history[-window:]
        main_counts = np.bincount(recent_main.astype(int), minlength=max(main_candidates) + 1)
        powerball_counts = np.bincount(
            recent_powerball.astype(int),
            minlength=max(powerball_candidates) + 1,
        )

        feature_row[f"main_sum_mean_{window}"] = float(np.mean(main_history[-window:].sum(axis=1)))
        feature_row[f"main_sum_std_{window}"] = float(np.std(main_history[-window:].sum(axis=1)))
        feature_row[f"powerball_mean_{window}"] = float(np.mean(recent_powerball))

        for number in main_candidates:
            feature_row[f"main_freq_{window}_{number}"] = float(main_counts[number])

        for number in powerball_candidates:
            feature_row[f"pb_freq_{window}_{number}"] = float(powerball_counts[number])

    for number in main_candidates:
        previous_index = last_seen_main[number]
        gap = sample_index - previous_index if previous_index >= 0 else sample_index + 1
        feature_row[f"main_gap_{number}"] = float(gap)

    for number in powerball_candidates:
        previous_index = last_seen_powerball[number]
        gap = sample_index - previous_index if previous_index >= 0 else sample_index + 1
        feature_row[f"pb_gap_{number}"] = float(gap)

    return feature_row


def build_supervised_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    df = df.copy()

    main_array = df[MAIN_COLUMNS].to_numpy(dtype=int)
    powerball_array = df["PB"].to_numpy(dtype=int)

    main_candidates = list(range(1, int(np.nanmax(main_array)) + 1))
    powerball_candidates = list(range(1, int(np.nanmax(powerball_array)) + 1))
    start_index = max(max(FEATURE_WINDOWS), LAG_DRAWS)

    last_seen_main = {number: -1 for number in main_candidates}
    last_seen_powerball = {number: -1 for number in powerball_candidates}
    for history_index in range(start_index):
        for number in main_array[history_index]:
            last_seen_main[int(number)] = history_index
        last_seen_powerball[int(powerball_array[history_index])] = history_index

    feature_rows: list[dict[str, float]] = []
    targets: list[np.ndarray] = []
    references: list[dict[str, object]] = []

    for sample_index in range(start_index, len(df)):
        history_main = main_array[:sample_index]
        history_powerball = powerball_array[:sample_index]
        context = DrawContext(
            draw_number=int(df.iloc[sample_index]["Draw Number"]),
            draw_date=pd.Timestamp(df.iloc[sample_index]["Date"]),
            jackpot=float(df.iloc[sample_index]["Jackpot"]),
        )

        feature_rows.append(
            _history_feature_dict(
                main_history=history_main,
                powerball_history=history_powerball,
                context=context,
                last_seen_main=last_seen_main,
                last_seen_powerball=last_seen_powerball,
                main_candidates=main_candidates,
                powerball_candidates=powerball_candidates,
                sample_index=sample_index,
            )
        )
        targets.append(df.iloc[sample_index][TARGET_COLUMNS].to_numpy(dtype=int))
        references.append(
            {
                "Draw Number": int(df.iloc[sample_index]["Draw Number"]),
                "Date": df.iloc[sample_index]["Date"],
            }
        )

        for number in main_array[sample_index]:
            last_seen_main[int(number)] = sample_index
        last_seen_powerball[int(powerball_array[sample_index])] = sample_index

    X = pd.DataFrame(feature_rows)
    y = np.vstack(targets)
    ref = pd.DataFrame(references)
    return X, y, ref


def build_next_draw_features(df: pd.DataFrame, context: DrawContext) -> pd.DataFrame:
    df = df.copy()
    main_array = df[MAIN_COLUMNS].to_numpy(dtype=int)
    powerball_array = df["PB"].to_numpy(dtype=int)

    main_candidates = list(range(1, int(np.nanmax(main_array)) + 1))
    powerball_candidates = list(range(1, int(np.nanmax(powerball_array)) + 1))

    last_seen_main = {number: -1 for number in main_candidates}
    last_seen_powerball = {number: -1 for number in powerball_candidates}
    for history_index in range(len(df)):
        for number in main_array[history_index]:
            last_seen_main[int(number)] = history_index
        last_seen_powerball[int(powerball_array[history_index])] = history_index

    feature_row = _history_feature_dict(
        main_history=main_array,
        powerball_history=powerball_array,
        context=context,
        last_seen_main=last_seen_main,
        last_seen_powerball=last_seen_powerball,
        main_candidates=main_candidates,
        powerball_candidates=powerball_candidates,
        sample_index=len(df),
    )
    return pd.DataFrame([feature_row])


class TicketPredictor:
    """Train one multiclass estimator per ball and emit a valid lottery ticket."""

    def __init__(self, estimator: BaseEstimator, estimator_name: str):
        self.estimator = estimator
        self.estimator_name = estimator_name

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "TicketPredictor":
        X_values = np.asarray(X)
        y_values = np.asarray(y)

        self.main_models_ = []
        for column_index in range(5):
            model = clone(self.estimator)
            model.fit(X_values, y_values[:, column_index])
            self.main_models_.append(model)

        self.powerball_model_ = clone(self.estimator)
        self.powerball_model_.fit(X_values, y_values[:, 5])
        self.main_numbers_ = sorted({int(value) for value in y_values[:, :5].reshape(-1)})
        self.powerball_numbers_ = sorted({int(value) for value in y_values[:, 5]})
        return self

    def _aligned_probabilities(
        self,
        estimator: BaseEstimator,
        X_values: np.ndarray,
        allowed_numbers: list[int],
    ) -> np.ndarray:
        probabilities = estimator.predict_proba(X_values)
        aligned = np.zeros((X_values.shape[0], len(allowed_numbers)), dtype=float)
        class_lookup = {int(label): index for index, label in enumerate(estimator.classes_)}
        for allowed_index, number in enumerate(allowed_numbers):
            if number in class_lookup:
                aligned[:, allowed_index] = probabilities[:, class_lookup[number]]
        return aligned

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        allowed_main_numbers: list[int] | None = None,
        allowed_powerball_numbers: list[int] | None = None,
    ) -> np.ndarray:
        X_values = np.asarray(X)
        allowed_main_numbers = allowed_main_numbers or self.main_numbers_
        allowed_powerball_numbers = allowed_powerball_numbers or self.powerball_numbers_

        main_non_selection = np.ones((X_values.shape[0], len(allowed_main_numbers)), dtype=float)
        for estimator in self.main_models_:
            aligned = self._aligned_probabilities(estimator, X_values, allowed_main_numbers)
            main_non_selection *= 1.0 - aligned

        main_scores = 1.0 - main_non_selection
        top_indices = np.argsort(main_scores, axis=1)[:, -5:]
        predicted_main = np.sort(np.take(np.asarray(allowed_main_numbers), top_indices), axis=1)

        powerball_scores = self._aligned_probabilities(
            self.powerball_model_,
            X_values,
            allowed_powerball_numbers,
        )
        powerball_predictions = np.take(
            np.asarray(allowed_powerball_numbers),
            np.argmax(powerball_scores, axis=1),
        ).reshape(-1, 1)

        return np.hstack([predicted_main, powerball_predictions])


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_main = y_true[:, :5]
    pred_main = y_pred[:, :5]
    true_powerball = y_true[:, 5]
    pred_powerball = y_pred[:, 5]

    main_hits = np.array(
        [len(set(map(int, truth)) & set(map(int, pred))) for truth, pred in zip(true_main, pred_main)],
        dtype=float,
    )
    powerball_hits = (true_powerball == pred_powerball).astype(float)
    exact_match = ((main_hits == 5.0) & (powerball_hits == 1.0)).astype(float)

    return {
        "mean_main_hits": float(main_hits.mean()),
        "main_hit_rate": float(main_hits.mean() / 5.0),
        "powerball_accuracy": float(powerball_hits.mean()),
        "mean_total_hits": float((main_hits + powerball_hits).mean()),
        "exact_match_rate": float(exact_match.mean()),
    }


def save_model_bundle(bundle: dict[str, object], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_model_bundle(model_path: str | Path) -> dict[str, object]:
    return joblib.load(model_path)
