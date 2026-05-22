"""Programmatic wrapper around the saved next-draw ticket predictor."""

from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from lotto_pipeline import (
    build_draw_context,
    build_next_draw_features,
    load_draws,
    load_model_bundle,
)

RESULTS_URL = "https://www.nlcbplaywhelotto.com/nlcb-lotto-plus-results/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def fetch_next_estimated_jackpot() -> float | None:
    try:
        response = requests.get(RESULTS_URL, timeout=30, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text("\n", strip=True)
        match = re.search(r"Next Estimated Jackpot\s+([\d.]+)\s+MILLION", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) * 1_000_000.0
    except requests.RequestException:
        return None
    return None


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


def format_prediction(result: dict[str, object]) -> str:
    main_numbers = " ".join(f"{int(value):02d}" for value in result["main_numbers"])
    lines = [
        f"Model: {result['model_name']}",
        f"Next draw #: {int(result['draw_number'])}",
        f"Next draw date: {result['draw_date'].strftime('%d-%b-%y')}",
    ]
    if result["estimated_jackpot"] is not None:
        lines.append(f"Estimated jackpot: ${float(result['estimated_jackpot']):,.2f}")
    lines.append(f"Predicted numbers: {main_numbers} | PB {int(result['powerball']):02d}")
    return "\n".join(lines)
