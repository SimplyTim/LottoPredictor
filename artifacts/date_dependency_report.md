# Date Dependency Analysis

- Rows analyzed: 2488
- Date range: 2001-07-04 to 2026-05-20
- Stable era start: 2012-09-02
- Historical main-number support max: 36
- Recent main-number support max: 35
- Last observed `36`: 2012-09-01

## Key Findings

- The clearest date dependency is structural, not predictive: main number `36` appears historically but disappears after September 2012.
- Stable-era month test significant features after FDR: none.
- Stable-era Wednesday vs Saturday test significant features after FDR: none.
- Stable-era individual number month dependencies after FDR: pb_3.
- Stable-era individual number weekday dependencies after FDR: none.
- Stable-era individual number time trends after FDR: none.

## Date-Only Holdout Model

- Holdout window: 2025-05-03 to 2026-05-20 (104 draws)
- Date-only mean_main_hits: 0.8462
- Date-only powerball_accuracy: 0.0865
- Date-only mean_total_hits: 0.9327
- Static frequency-baseline mean_total_hits: 0.8558
- Random-ticket expected mean_total_hits: 0.8143
