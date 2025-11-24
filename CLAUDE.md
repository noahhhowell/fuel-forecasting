# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fuel volume forecasting system for ~400 gas stations. It forecasts monthly fuel volumes by site and grade using historical daily sales data from PDI (Excel/CSV exports). The system uses SQLite for local storage with automatic deduplication, and supports ETS (Holt-Winters) and Seasonal Naive models with ensemble (median) forecasting.

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Activate virtual environment (optional, uv run handles this)
.\.venv\Scripts\activate

# Load data from Excel/CSV
python cli.py load --file data/sales_2024.xlsx
python cli.py load --file data/sales_2024.csv
python cli.py load --directory data/  # Load all files in directory

# Check database status
python cli.py status
python cli.py status --detailed  # Shows data quality by site

# Generate forecasts
python cli.py forecast 2025-07  # Aggregate forecast (console output)
python cli.py forecast 2025-07 --by site --output forecasts/2025-07_sites.xlsx
python cli.py forecast 2025-07 --by site_grade --output forecasts/2025-07_detailed.csv

# Export historical data
python cli.py export --output fuel_data.csv

# All commands work with uv run prefix
uv run python cli.py forecast 2025-07
```

## Backtesting (Available on backtesting branch)

The backtesting feature allows you to evaluate forecast accuracy by simulating historical forecasts using only data available at that time, then comparing to actual outcomes.

### How Backtesting Works

Backtesting uses two flags together:
- `--train-through YYYY-MM`: Limits training data to only through the specified month (simulates making a forecast in the past)
- `--compare-actuals`: Automatically fetches actual volumes for the target month and computes error metrics

### Backtesting Workflow

```bash
# Example: Simulate forecasting January 2025 using only data through October 2024
# Then compare forecast to actual January 2025 volumes

# 1. Generate backtest forecast
python cli.py forecast 2025-01 \
  --by site \
  --train-through 2024-10 \
  --compare-actuals \
  --output backtest_jan2025.xlsx

# Output shows error metrics in console:
# ACTUALS CHECK
#   ETS          MAE:     45,234 | MAPE:   7.23% | n=385
#   SeasonalNaive MAE:    52,891 | MAPE:   8.45% | n=385
#   ENSEMBLE     MAE:     43,112 | MAPE:   6.89% | n=385

# 2. Excel output includes additional columns:
# - actual_volume: What actually happened in January 2025
# - error: forecast_volume - actual_volume
# - abs_error: Absolute error
# - ape: Absolute percentage error (for MAPE calculation)
```

### Common Backtesting Scenarios

**Test last 6 months to find optimal model:**
```bash
# Backtest each of the last 6 months
for month in 2024-{07..12}; do
  train_through=$(date -d "$month -3 months" +%Y-%m)
  python cli.py forecast $month \
    --train-through $train_through \
    --compare-actuals \
    --by site \
    --output backtest_$month.xlsx
done

# Review which model (ETS, SeasonalNaive, or ENSEMBLE) had lowest MAPE
```

**Evaluate new model configuration:**
```bash
# After changing outlier thresholds or model parameters
python cli.py forecast 2024-11 \
  --train-through 2024-08 \
  --compare-actuals \
  --by site_grade \
  --output validation_test.xlsx

# Check if changes improved MAPE vs baseline
```

**Test with different data requirements:**
```bash
# Compare forecast quality with different minimum months
python cli.py forecast 2025-01 \
  --train-through 2024-10 \
  --compare-actuals \
  --min-months 12 \
  --output backtest_12mo.xlsx

python cli.py forecast 2025-01 \
  --train-through 2024-10 \
  --compare-actuals \
  --min-months 24 \
  --output backtest_24mo.xlsx

# More sites with 12mo, but possibly lower accuracy
```

### Understanding Error Metrics

The backtesting output provides three metrics:
- **MAE (Mean Absolute Error)**: Average absolute difference in gallons. Lower is better. Directly interpretable in volume units.
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error. Lower is better. Standard forecast accuracy metric.
  - <5%: Excellent
  - 5-10%: Good
  - 10-20%: Acceptable
  - >20%: Poor (investigate data quality or model assumptions)
- **n**: Number of forecasts compared (sites with both forecast and actuals)

The console output shows metrics per model, making it easy to compare ETS vs SeasonalNaive vs ENSEMBLE.

### Implementation Details

**Backtesting adds these methods to forecaster.py:**
- `_get_history_end_date()`: Converts YYYY-MM to end-of-month timestamp for filtering
- `_get_actuals_for_month()`: Aggregates actual sales for target month at the requested level (total/grade/site/site_grade)
- `_attach_actuals()`: Left-joins actuals to forecasts, computes error columns, and calculates summary metrics by model

**Key behaviors:**
- If actuals don't exist for the target month, `--compare-actuals` is silently ignored (no error)
- Error metrics exclude any rows where actual_volume is NaN or zero (avoids division-by-zero in MAPE)
- Actuals use same filtering as training (exclude_estimated=True)
- Summary metrics are stored in DataFrame.attrs["actuals_summary"] for programmatic access

**Workflow under the hood:**
1. `--train-through 2024-10` → Sets end_date filter on all data queries
2. Models fit on 2022-01 through 2024-10 data only
3. Forecast generated for 2025-01
4. `--compare-actuals` → Queries database for 2025-01 actuals (entire month)
5. Aggregates actuals at same level as forecasts (site/grade/site_grade/total)
6. Merges on (site_id, grade, target_month)
7. Computes error columns
8. Calculates MAE/MAPE by model
9. Displays summary in console and adds columns to Excel output
```

## Architecture

### Core Modules

**database.py** - SQLite data layer with automatic deduplication
- `FuelDatabase`: Manages sales data with composite primary key (site_id, grade, day)
- Handles Excel (row 5 headers by default) and CSV ingestion
- Column normalization (flexible casing/spacing)
- Metadata tracking for load history
- Query methods for forecasting preparation

**models.py** - Forecasting models with base abstraction
- `ForecastModel`: Abstract base class with fit/predict interface
- `ETSModel`: Holt-Winters exponential smoothing with adaptive seasonality (add vs mul based on CV heuristic), damped trend, and automatic seasonality detection (requires 24+ months)
- `SeasonalNaiveModel`: Same-month-last-year baseline with frequency enforcement to prevent seasonal drift
- Both models clamp forecasts to non-negative values

**forecaster.py** - High-level forecasting orchestration
- `FuelForecaster`: Main interface with progress tracking
- Monthly data preparation with MAD-based outlier handling (clips vs removes to preserve time series)
- Multi-level forecasting: total, grade, site, site_grade
- Ensemble forecasting (median of available models)
- Data sufficiency checks (default: 24 months minimum)
- Excel/CSV export with three outputs: Forecasts, Skipped (insufficient data), Summary

**cli.py** - Command-line interface with argparse
- Four main commands: load, status, forecast, export
- Forecast levels controlled by `--by` flag
- Model selection via `--model` (default: all + ensemble)
- Data quality thresholds: `--min-months`, `--include-all`
- Backtesting flags (backtesting branch): `--train-through YYYY-MM`, `--compare-actuals`

### Data Flow

1. **Ingestion**: Excel/CSV → normalize columns → FuelDatabase → SQLite (dedupe on insert via composite PK)
2. **Preparation**: Query DB → aggregate to monthly → MAD outlier handling → validate sufficiency
3. **Training**: Fit ETS and SeasonalNaive models (adaptive seasonality, damped trend)
4. **Forecasting**: Generate predictions → compute ensemble (median) → format results
5. **Export**: Excel (3 sheets) or CSV (3 files) with column order: site_id, grade, target_month, model, forecast_volume

### Key Design Decisions

**Outlier Handling**: Uses MAD (Median Absolute Deviation) instead of IQR for robustness to asymmetric distributions. Outliers are capped at boundaries (not removed) to preserve time series structure. Lower bound: max(0, median - 3*MAD). Upper bound: median * 5 (allows holiday spikes but clips data errors).

**Seasonal Naive Implementation**: Enforces monthly frequency with forward-fill to prevent "12 months ago" misalignment when months are missing. This was a critical fix to ensure accurate year-over-year comparisons.

**ETS Seasonality**: Automatically picks additive vs multiplicative based on CV (std/mean) with 0.3 threshold. Uses damped trend to prevent unrealistic long-term extrapolation. Disables seasonality if <24 months of data.

**Data Sufficiency**: Default 24-month minimum ensures reliable seasonal patterns. Adjustable via `--min-months` or `--include-all` for new stores.

**Ensemble Strategy**: Median (not mean) of available models for robustness to model-specific failures or outliers.

## Important Behaviors

- **Deduplication**: Automatic on insert via composite primary key (site_id, grade, day). Loading same file twice skips duplicates, not an error.
- **Excel Headers**: Default skiprows=4 (row 5 headers). Override with `--header-row` if needed.
- **Column Flexibility**: Database normalizes column names (strip whitespace, case-insensitive matching) to handle PDI export variations.
- **Negative Volumes**: Both models clamp forecasts to non-negative values (volumes cannot be negative).
- **Progress Tracking**: Site-level and site-grade forecasts show progress every 20 iterations for long-running jobs.
- **Skipped Sites**: Sites with insufficient data are tracked in separate "Skipped" sheet/file, not silently omitted.

## Testing Notes

No formal test suite exists. Testing approaches:

**Manual smoke testing:**
1. Load sample data: `python cli.py load --file data/sample.xlsx`
2. Check status: `python cli.py status --detailed`
3. Generate test forecast: `python cli.py forecast 2025-07 --by site --output test.xlsx`
4. Validate output Excel has 3 sheets: Forecasts, Skipped, Summary

**Automated backtesting (backtesting branch only):**
Use `--train-through` and `--compare-actuals` flags to evaluate historical forecast accuracy with automatic error metrics (MAE, MAPE). See Backtesting section above for detailed workflows.

## Common Patterns

**Adding New Forecast Levels**: Modify `forecaster.py` → `forecast()` method → add elif branch for new `--by` level → implement grouping logic → ensure progress tracking.

**Adding New Models**:
1. Create class in `models.py` inheriting from `ForecastModel`
2. Implement `fit()` and `predict()` methods
3. Add to `get_available_models()` dict
4. Model automatically available in CLI via `--model` flag

**Modifying Outlier Logic**: Edit `forecaster.py` → `_handle_outliers()` method. Current logic uses MAD with 3.0 threshold (lower) and 5.0 * median (upper).

**Changing Default Thresholds**: Modify `FuelForecaster.__init__()` for min_months_data, or `forecaster.py` MAD multipliers in `_handle_outliers()`.

## Database Schema

```sql
CREATE TABLE sales (
    site_id TEXT NOT NULL,
    grade TEXT NOT NULL,
    day DATE NOT NULL,
    brand TEXT,
    site TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    owner TEXT,
    b_unit TEXT,
    stock REAL,
    delivered REAL,
    volume REAL,
    is_estimated BOOLEAN,
    total_sales REAL,
    target REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (site_id, grade, day)
);

CREATE TABLE load_metadata (
    load_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    rows_loaded INTEGER,
    rows_duplicates INTEGER,
    load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Dependencies

Core: pandas, numpy, openpyxl (Excel I/O), statsmodels (ETS)
Database: sqlite3 (stdlib)
Package manager: uv (not pip)

No requirements.txt or pyproject.toml exists - dependencies managed via uv.lock only.

## File Organization

- **data/**: Input Excel/CSV files (gitignored)
- **forecasts/**: Output forecast files (gitignored)
- **fuel_sales.db**: SQLite database (gitignored)
- All source code at repository root (no src/ subdirectory)
- Documentation: README.md (quick start), EXAMPLES.md (scenarios), QUICKREF.md (cheatsheet), SETUP.md (Windows setup)

## Known Limitations

- Single forecast horizon only (cannot generate multiple horizons in one run)
- No confidence intervals (deterministic point forecasts only)
- Site closures/openings handled via data availability check, not explicit lifecycle tracking
- No parallel processing (site-grade forecasts are sequential)
- Backtesting available on backtesting branch only (not merged to main)
