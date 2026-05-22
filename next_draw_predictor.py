"""CLI entry point for next-draw Lotto Plus prediction."""

from __future__ import annotations

import argparse

from lotto_pipeline import DRAW_DATE_FORMAT
from predictor import format_prediction, predict_next_draw


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


def main() -> None:
    args = parse_args()
    result = predict_next_draw(
        draws_path=args.draws_path,
        model_path=args.model_path,
        draw_number=args.draw_number,
        draw_date=args.draw_date,
        estimated_jackpot=args.estimated_jackpot,
    )
    print(format_prediction(result))


if __name__ == "__main__":
    main()
