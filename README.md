# LottoPredictor

This project scrapes Trinidad and Tobago Lotto Plus history, engineers lag and rolling-history features, benchmarks five multiclass models, and saves the best one for next-draw predictions.

## Current workflow

1. Refresh the draw history:

```bash
python scrapehistory.py
```

2. Benchmark five models and save the best performer:

```bash
python model_experiments.py
```

3. Predict the next draw with the saved model:

```bash
python next_draw_predictor.py
```

## Models benchmarked

- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Extra Trees
- HistGradientBoosting

The experiments use time-series cross-validation and a final chronological holdout set instead of a random train/test split.

## Latest verified benchmark

Best model: `knn`

Holdout metrics:

- `mean_total_hits`: `0.8462`
- `mean_main_hits`: `0.7500`
- `powerball_accuracy`: `0.0962`
- `exact_match_rate`: `0.0000`

This is consistent with lottery outcomes behaving close to random: the pipeline can surface weak recency/frequency structure, but it still does not produce exact-ticket wins on holdout data.

## Generated artifacts

- `artifacts/model_benchmark.csv`
- `artifacts/holdout_predictions.csv`
- `artifacts/best_model_summary.json`
- `artifacts/best_ticket_predictor.joblib`
