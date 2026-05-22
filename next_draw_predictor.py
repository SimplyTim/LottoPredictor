"""Load the best saved model and predict the next Lotto Plus draw."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from lotto_pipeline import (
    DRAW_DATE_FORMAT,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws-path", default="draws.csv", help="Historical draw CSV.")
    parser.add_argument(
        "--model-path",
        default="artifacts/best_ticket_predictor.joblib",
        help="Saved model bundle.",
    )
    parser.add_argument("--draw-number", type=int, help="Override the upcoming draw number.")
    parser.add_argument(
        "--draw-date",
        help=f"Override the upcoming draw date in {DRAW_DATE_FORMAT} format.",
    )
    parser.add_argument(
        "--estimated-jackpot",
        type=float,
        help="Override the upcoming estimated jackpot.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(Path(args.model_path))
    predictor = bundle["model"]
    feature_columns = bundle["feature_columns"]
    allowed_main_numbers = bundle["allowed_main_numbers"]
    allowed_powerball_numbers = bundle["allowed_powerball_numbers"]

    df = load_draws(args.draws_path)
    estimated_jackpot = args.estimated_jackpot
    if estimated_jackpot is None:
        estimated_jackpot = fetch_next_estimated_jackpot()

    context = build_draw_context(
        df,
        draw_number=args.draw_number,
        draw_date=args.draw_date,
        jackpot=estimated_jackpot,
    )
    feature_frame = build_next_draw_features(df, context)
    feature_frame = feature_frame[feature_columns]

    prediction = predictor.predict(
        feature_frame,
        allowed_main_numbers=allowed_main_numbers,
        allowed_powerball_numbers=allowed_powerball_numbers,
    )[0]

    main_numbers = " ".join(f"{int(value):02d}" for value in prediction[:5])
    powerball = int(prediction[5])

    print(f"Model: {bundle['model_name']}")
    print(f"Next draw #: {context.draw_number}")
    print(f"Next draw date: {context.draw_date.strftime(DRAW_DATE_FORMAT)}")
    if estimated_jackpot is not None:
        print(f"Estimated jackpot: ${estimated_jackpot:,.2f}")
    print(f"Predicted numbers: {main_numbers} | PB {powerball:02d}")


if __name__ == "__main__":
    main()
