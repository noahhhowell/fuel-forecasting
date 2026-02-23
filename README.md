# Fuel Forecasting System

Forecast gas station fuel volumes using ETS (Holt-Winters) and Seasonal Naive models with a weighted ensemble. Supports Excel/CSV ingest, SQLite storage, and Excel/CSV forecast exports.

## How It Works

The system combines two forecasting models into a single **ENSEMBLE** prediction for each site:

1. **ETS** (Holt-Winters) — Exponential smoothing that captures trend and seasonality.
2. **Seasonal Naive** — Uses same-month-last-year values as the forecast.

By default, the ensemble uses fixed weights (70% ETS, 30% Seasonal Naive). If you run **calibration** first, the system learns the best weight for each site based on which model has been more accurate there historically. See the [Calibration](#calibration) section for details.

The **ENSEMBLE** forecast is what you should use for decisions. Individual model outputs (ets, snaive) are included in exports for transparency.

## Setup

### Prerequisites

- Windows 10/11
- Python 3.9+

### Install

```powershell
# Install uv if you don't have it
pip install uv

# Install project dependencies (creates .venv automatically)
uv sync

# Create data directories
mkdir data forecasts

# Verify it works
uv run python cli.py --help
```

### Running Commands

You can either prefix commands with `uv run`:

```powershell
uv run python cli.py forecast 2026-03
```

Or activate the venv and run directly:

```powershell
.\.venv\Scripts\activate
python cli.py forecast 2026-03
```

The rest of this guide omits the `uv run` prefix for brevity.

## Loading Data

Put Excel/CSV exports from PDI into the `data/` folder, then load them.

```bash
# Single file (Excel skips first 4 rows by default to find headers on row 5)
python cli.py load --file data/FuelVolume_2024.xlsx

# CSV file (headers expected on row 1)
python cli.py load --file data/fuel_data.csv

# All files in a directory at once
python cli.py load --directory ./data
```

Deduplication is automatic (primary key: site_id + grade + day). Re-loading the same file is safe.

If your Excel headers aren't on row 5, use `--header-row`:

```bash
python cli.py load --file data/custom.xlsx --header-row 0
```

## Checking Status

```bash
# Quick summary
python cli.py status

# Detailed data quality report (shows sites with insufficient data)
python cli.py status --detailed
```

## Generating Forecasts

```bash
# Basic forecast (by site, printed to console)
python cli.py forecast 2026-03

# Save to Excel
python cli.py forecast 2026-03 --by site --output forecasts/2026-03_site.xlsx

# Save to CSV
python cli.py forecast 2026-03 --by site --output forecasts/2026-03_site.csv
```

If calibration has been run, forecasts automatically use the calibrated per-site weights. To force the default fixed weights instead, add `--no-calibration`:

```bash
python cli.py forecast 2026-03 --no-calibration
```

### Forecast Levels

| Level | What it does | Output rows |
|-------|--------------|-------------|
| `grade` | One forecast per fuel grade (UNL, PRE, DSL) | 3 x #grades |
| `site` | One forecast per site, all grades combined | 3 x #sites |
| `site_grade` | One forecast per site-grade combination | 3 x #combos |

```bash
python cli.py forecast 2026-03 --by grade --output grades.xlsx
python cli.py forecast 2026-03 --by site --output sites.xlsx
python cli.py forecast 2026-03 --by site_grade --output detailed.xlsx
```

### Forecast Options

| Flag | What it does |
|------|--------------|
| `--by` | Aggregation level: `grade`, `site`, `site_grade` (default: `site`) |
| `--output` | Save to Excel (.xlsx) or CSV (.csv) |
| `--model ets` | Use only ETS model |
| `--model snaive` | Use only Seasonal Naive model |
| `--min-months 12` | Lower the data threshold (default: 24 months) |
| `--include-all` | Include sites with insufficient data |
| `--no-calibration` | Ignore calibrated weights; use fixed 70/30 blend |

## Calibration

Calibration teaches the system which model works best for each site. Without calibration, every site gets the same 70/30 ETS/Seasonal Naive blend. After calibration, each site gets its own optimized weights based on historical accuracy.

### What It Does

1. **Backtests** all sites over recent months to measure each model's accuracy per site.
2. **Computes per-site weights** using inverse-MAPE scoring — models that were more accurate get more weight. Every model is guaranteed at least a 10% floor so no model is completely ignored.
3. **Applies shrinkage** for sites with limited data — if a site has fewer than 6 backtest months, its weights are blended toward the global average to avoid overfitting to a small sample.
4. **Builds prediction intervals** by measuring how far off past forecasts were (residual distributions), which can be used for confidence ranges.
5. **Stores everything** in the database, tagged with a run ID. Only the latest calibration run is used for forecasting — older runs are kept in the database but never mixed into new forecasts.

### Running Calibration

```bash
# Default: 12-month backtest, 2-month-ahead horizon
python cli.py calibrate

# Longer backtest window
python cli.py calibrate --months 18

# Save a detailed report to Excel
python cli.py calibrate --output calibration_report.xlsx
```

### Calibration Options

| Flag | Default | What it does |
|------|---------|--------------|
| `--months` | 12 | Number of recent months to backtest |
| `--min-months` | 24 | Minimum months of data required before the backtest window |
| `--horizon` | 2 | How many months ahead to forecast in each backtest step |
| `--weight-floor` | 0.10 | Minimum weight any model can receive (prevents dropping a model entirely) |
| `--output` | None | Save calibration report to an Excel file |

### Calibration Report

When you use `--output`, the Excel report contains three sheets:

| Sheet | Contents |
|-------|----------|
| Weights | Per-site model weights (site_id, model_name, weight, mape_pct, n_months) |
| Per-Model MAPE | Backtest accuracy for each model at each site |
| Interval Calibration | Residual distribution stats (std, p10, p90) per site and globally |

### Input Validation

All parameters are validated before any work begins. If you pass an invalid value (e.g., `--months 0`, `--horizon -1`, or a non-finite weight floor), you'll get a clear error message immediately rather than a confusing failure deep in the pipeline.

### When to Re-Calibrate

Re-run calibration when:
- You load a significant amount of new data
- Seasonal patterns change (e.g., a new site opens or a site changes behavior)
- You want to update weights after a few more months of history

## Exporting Raw Data

Export the database to CSV with optional filters:

```bash
# Everything
python cli.py export --output fuel_data.csv

# Date range
python cli.py export --output 2024.csv --start-date 2024-01-01 --end-date 2024-12-31

# Specific site
python cli.py export --output site_4551.csv --site-id 4551

# Specific grade
python cli.py export --output unl.csv --grade UNL

# Include estimated values (excluded by default)
python cli.py export --output with_estimated.csv --include-estimated
```

## Output Format

### Excel (.xlsx)

| Sheet | Contents |
|-------|----------|
| Forecasts | ENSEMBLE results only. Columns: `site_id, grade, target_month, forecast_volume, prior_year_volume, yoy_change_pct` |
| BU Total | One-row grand total with YoY % |
| Site Summary | Site totals from summing grades (only for `--by site_grade`) |
| Product Summary | Grade-level aggregation with YoY % (only for `--by site_grade`) |
| Model Detail | All models (ets, snaive, ENSEMBLE) with full diagnostic columns |
| Skipped | Sites skipped due to insufficient data |

### CSV (.csv)

Main file has ENSEMBLE results only. Companion files are created alongside:
- `<name>_bu_total.csv`
- `<name>_site_summary.csv` (only for `--by site_grade`)
- `<name>_product_summary.csv` (only for `--by site_grade`)
- `<name>_model_detail.csv`
- `<name>_skipped.csv`

## Backtesting

Evaluate forecast accuracy against historical actuals:

```bash
# Test against last 6 months (default)
python backtest.py

# Test against last 12 months, save results to Excel
python backtest.py --months 12 --output backtest_results.xlsx

# Custom horizon (e.g., 3 months ahead)
python backtest.py --horizon 3
```

Holds out recent months, generates forecasts using only prior data, and reports:
- MAPE (mean absolute percentage error) per site
- Median APE across all site-month rows
- Trimmed MAPE (per-site mean after dropping each site's single worst month)

Like calibration, backtest parameters are validated upfront — invalid values produce clear error messages.

## Examples

### Recommended workflow (first time)

```bash
# 1. Load your data
python cli.py load --directory ./data

# 2. Check data quality
python cli.py status --detailed

# 3. Run calibration to learn per-site weights
python cli.py calibrate --output calibration_report.xlsx

# 4. Generate forecasts (uses calibrated weights automatically)
python cli.py forecast 2026-04 --by site --output forecasts/2026-04_site.xlsx
```

### Weekly data update workflow

```bash
# 1. Load new export from PDI
python cli.py load --file data/FuelVolume_2026_H1.xlsx

# 2. Quick sanity check
python cli.py status

# 3. Generate forecast (calibrated weights from last run are still active)
python cli.py forecast 2026-04 --by site --output forecasts/2026-04_site.xlsx
```

### Re-calibrate after loading several months of new data

```bash
python cli.py calibrate --months 12 --output calibration_report.xlsx
```

### Detailed site-grade forecasts

```bash
python cli.py forecast 2026-03 --by site_grade --output forecasts/2026-03_detailed.xlsx
```

### Include sites with limited data

```bash
# Lower the threshold to 12 months
python cli.py forecast 2026-03 --by site --min-months 12 --output forecasts/lenient.xlsx

# Or include everything regardless
python cli.py forecast 2026-03 --by site --include-all --output forecasts/all.xlsx
```

## Running Tests

```bash
# Run the full test suite
uv run pytest

# Verbose output (see every test name)
uv run pytest tests/ -v

# Run just one file
uv run pytest tests/test_models.py

# Run tests matching a keyword
uv run pytest -k "test_ensemble"
```

Tests use temporary databases with synthetic data — they don't touch `fuel_sales.db` or any real data.

## Expected Runtimes

| Level | Approximate time |
|-------|-----------------|
| `grade` | Under 1 minute |
| `site` (~400 sites) | 5-10 minutes |
| `site_grade` (~1,200 combos) | 15-30 minutes |

## Best Practices

1. Load data weekly after PDI exports
2. Run `calibrate` after loading significant new data to keep weights fresh
3. Use ENSEMBLE for production decisions
4. Run `status --detailed` regularly to monitor data quality
5. Require 24+ months of data for reliable forecasts
6. Review the Skipped sheet to understand gaps
7. Keep past forecasts for accuracy tracking
8. Spot-check forecasts against recent actuals

## Troubleshooting

### "uv not found"
```powershell
pip install uv
```

### "Python not found"
Install Python 3.9+ from python.org, then retry.

### Insufficient data errors
```bash
# See which sites have issues
python cli.py status --detailed

# Lower the threshold
python cli.py forecast 2026-03 --by site --min-months 12

# Or include all sites
python cli.py forecast 2026-03 --by site --include-all
```

### "months must be >= 1" or similar validation errors
These mean a parameter is out of range. Check the `--months`, `--min-months`, `--horizon`, or `--weight-floor` values you passed. All must be positive (horizon can be 0).

### Permission errors
Run PowerShell as Administrator.

## File Locations

| Path | Purpose |
|------|---------|
| `fuel_sales.db` | SQLite database (auto-created on first load) |
| `data/` | Put PDI Excel/CSV exports here |
| `forecasts/` | Forecast output files |
| `calibrate.py` | Calibration module (learns per-site weights) |
| `backtest.py` | Backtesting module (evaluates forecast accuracy) |
| `tests/` | Automated test suite (run with `uv run pytest`) |
