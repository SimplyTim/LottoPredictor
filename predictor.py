"""Programmatic wrapper around the saved next-draw ticket predictor."""

from __future__ import annotations

from pathlib import Path

from lotto_pipeline import (
    build_draw_context,
    build_next_draw_features,
    load_draws,
    load_model_bundle,
)
from next_draw_predictor import fetch_next_estimated_jackpot


def predict_next_draw(
    draws_path: str | Path = "draws.csv",
    model_path: str | Path = "artifacts/best_ticket_predictor.joblib",
    draw_number: int | None = None,
    draw_date: str | None = None,
    estimated_jackpot: float | None = None,
) -> dict[str, object]:
    bundle = load_model_bundle(model_path)
    predictor = bundle["model"]

    df = load_draws(draws_path)
    jackpot = estimated_jackpot if estimated_jackpot is not None else fetch_next_estimated_jackpot()
    context = build_draw_context(
        df,
        draw_number=draw_number,
        draw_date=draw_date,
        jackpot=jackpot,
    )
    feature_frame = build_next_draw_features(df, context)[bundle["feature_columns"]]
    prediction = predictor.predict(
        feature_frame,
        allowed_main_numbers=bundle["allowed_main_numbers"],
        allowed_powerball_numbers=bundle["allowed_powerball_numbers"],
    )[0]

    return {
        "model_name": bundle["model_name"],
        "draw_number": context.draw_number,
        "draw_date": context.draw_date,
        "estimated_jackpot": jackpot,
        "main_numbers": [int(value) for value in prediction[:5]],
        "powerball": int(prediction[5]),
    }
