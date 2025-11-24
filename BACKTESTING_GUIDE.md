# Backtesting Guide

Quick guide to using the backtesting feature for evaluating forecast accuracy.

## What is Backtesting?

Backtesting simulates making forecasts in the past using only data available at that time, then comparing them to what actually happened. This lets you:
- Evaluate which model (ETS, SeasonalNaive, or Ensemble) is most accurate
- Test different configurations (outlier thresholds, minimum data requirements)
- Build confidence in your forecasting approach before using it for production

## Quick Start

```bash
# Simulate forecasting January 2025 using only data through October 2024
# Compare to actual January 2025 volumes
python cli.py forecast 2025-01 \
  --by site \
  --train-through 2024-10 \
  --compare-actuals \
  --output backtest_jan2025.xlsx
```

## The Two Backtesting Flags

### `--train-through YYYY-MM`
Limits training data to only through the specified month. This simulates making a forecast in the past.

**Example:** `--train-through 2024-10` means models only see data from 2022-01 through 2024-10.

### `--compare-actuals`
Automatically fetches actual volumes for the target month and computes error metrics.

**What it adds:**
- Console output with MAE and MAPE for each model
- Excel columns: actual_volume, error, abs_error, ape
- Enables data-driven model selection

## Understanding the Output

### Console Metrics

```
ACTUALS CHECK
  ETS          MAE:     45,234 | MAPE:   7.23% | n=385
  SeasonalNaive MAE:    52,891 | MAPE:   8.45% | n=385
  ENSEMBLE     MAE:     43,112 | MAPE:   6.89% | n=385
```

- **MAE (Mean Absolute Error)**: Average error in gallons. Lower is better.
- **MAPE (Mean Absolute Percentage Error)**: Average % error. Standard accuracy metric.
  - <5%: Excellent
  - 5-10%: Good
  - 10-20%: Acceptable
  - >20%: Poor (investigate)
- **n**: Number of sites/combinations with both forecast and actual data

### Excel Output Columns

When using `--compare-actuals`, the Excel file includes:
- `actual_volume`: What actually happened
- `error`: forecast_volume - actual_volume (positive = over-forecast)
- `abs_error`: Absolute error
- `ape`: Absolute percentage error (for manual analysis)

## Common Workflows

### 1. Test Last 3 Months

```bash
# November 2024
python cli.py forecast 2024-11 \
  --train-through 2024-08 \
  --compare-actuals \
  --by site \
  --output backtest_nov.xlsx

# December 2024
python cli.py forecast 2024-12 \
  --train-through 2024-09 \
  --compare-actuals \
  --by site \
  --output backtest_dec.xlsx

# January 2025
python cli.py forecast 2025-01 \
  --train-through 2024-10 \
  --compare-actuals \
  --by site \
  --output backtest_jan.xlsx

# Compare MAPE across months to find which model is most consistent
```

### 2. Evaluate Model Changes

```bash
# Before making a change to outlier thresholds
python cli.py forecast 2024-12 \
  --train-through 2024-09 \
  --compare-actuals \
  --by site \
  --output baseline.xlsx

# Make changes to forecaster.py (e.g., adjust MAD thresholds)

# Test with same parameters
python cli.py forecast 2024-12 \
  --train-through 2024-09 \
  --compare-actuals \
  --by site \
  --output after_changes.xlsx

# Compare MAPE: Did it improve?
```

### 3. Test Data Requirements

```bash
# Strict (24 months minimum)
python cli.py forecast 2025-01 \
  --train-through 2024-10 \
  --compare-actuals \
  --min-months 24 \
  --by site \
  --output strict_24mo.xlsx

# Lenient (12 months minimum)
python cli.py forecast 2025-01 \
  --train-through 2024-10 \
  --compare-actuals \
  --min-months 12 \
  --by site \
  --output lenient_12mo.xlsx

# Compare:
# - How many more sites with 12mo? (check Forecasts sheet)
# - Is MAPE worse with less data? (check console output)
```

### 4. Model Selection

```bash
# Test each model individually
python cli.py forecast 2024-12 \
  --train-through 2024-09 \
  --compare-actuals \
  --model ets \
  --by site \
  --output ets_only.xlsx

python cli.py forecast 2024-12 \
  --train-through 2024-09 \
  --compare-actuals \
  --model snaive \
  --by site \
  --output snaive_only.xlsx

# Test ensemble
python cli.py forecast 2024-12 \
  --train-through 2024-09 \
  --compare-actuals \
  --by site \
  --output ensemble.xlsx

# Compare MAPE to decide which to use for production
```

## Best Practices

### Choosing Training Window
- Use `--train-through` to simulate realistic forecasting (2-3 months before target)
- Example: Forecasting January 2025 in real life, you'd have data through October 2024
- Use `--train-through 2024-10` to simulate this

### Aggregation Level
- Start with `--by site` for faster testing
- Use `--by site_grade` for detailed analysis but expect longer runtime
- Use `--by grade` or `total` for quick sanity checks

### Interpreting Results
- ENSEMBLE typically has lowest MAPE (median is robust)
- If one model consistently outperforms, consider using only that model
- High MAPE (>20%) at a site might indicate:
  - Data quality issues
  - Recent operational changes (remodel, competition)
  - Seasonal patterns not captured by 24 months of history

### Multiple Test Periods
- Always test multiple months, not just one
- Models might perform differently in different seasons
- Look for consistency across 3-6 test periods

## Troubleshooting

**"No actuals available for that target month to compare"**
- The target month you're forecasting doesn't have actual data yet
- Either the month is in the future or data hasn't been loaded
- Solution: Choose a past month with known actuals, or run without `--compare-actuals`

**MAPE is very high (>30%)**
- Check "Skipped" sheet - might be forecasting sites with insufficient data
- Increase `--min-months` threshold
- Check for data quality issues: `python cli.py status --detailed`

**Different n values between models**
- Normal: Models might fail for different sites due to data characteristics
- ETS requires more stable patterns than SeasonalNaive
- ENSEMBLE uses median of available models, so n is sites where at least one model succeeded

## Technical Details

### How It Works Under the Hood
1. `--train-through 2024-10` sets end_date filter on all database queries
2. Models fit on historical data through October 2024 only
3. Forecast generated for target month (e.g., January 2025)
4. `--compare-actuals` queries database for January 2025 actual sales
5. Actuals aggregated at same level as forecast (site/grade/site_grade/total)
6. Left join on (site_id, grade, target_month)
7. Error metrics computed and displayed

### Error Calculation
- error = forecast_volume - actual_volume
- abs_error = |error|
- ape = abs_error / actual_volume (excluding zeros)
- MAE = mean(abs_error)
- MAPE = mean(ape)

### What Gets Excluded
- Sites where actual_volume is NaN or 0 (division by zero in MAPE)
- Any forecast that failed to generate (logged in "Skipped" sheet)
- Estimated volumes (same as training: exclude_estimated=True)

## Examples by Use Case

**"Which model should I use for production?"**
```bash
# Backtest last 6 months, review console MAPE for each model
for month in {07..12}; do
  python cli.py forecast 2024-$month \
    --train-through 2024-$(printf "%02d" $((month-3))) \
    --compare-actuals \
    --by site \
    --output backtest_2024-$month.xlsx
done
```

**"Should I use 12 or 24 months minimum?"**
```bash
python cli.py forecast 2024-12 --train-through 2024-09 --compare-actuals --min-months 12 --output test_12mo.xlsx
python cli.py forecast 2024-12 --train-through 2024-09 --compare-actuals --min-months 24 --output test_24mo.xlsx
# Compare: More sites with 12mo, but is accuracy worse?
```

**"Did my code change improve forecasts?"**
```bash
# Baseline
git checkout main
python cli.py forecast 2024-12 --train-through 2024-09 --compare-actuals --output baseline.xlsx

# Your changes
git checkout your-feature-branch
python cli.py forecast 2024-12 --train-through 2024-09 --compare-actuals --output improved.xlsx

# Compare MAPE in console output
```

## See Also

- README.md - General system documentation
- EXAMPLES.md - Production forecasting examples (without backtesting)
- CLAUDE.md - Technical architecture and implementation details
