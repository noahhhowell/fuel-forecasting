# Fuel Forecasting System

Forecast gas station fuel volumes using ETS (Holt-Winters) and Seasonal Naive models with an ensemble median. Supports Excel/CSV ingest, SQLite storage, and Excel/CSV forecast exports.

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

### Options

| Flag | What it does |
|------|--------------|
| `--by` | Aggregation level: `grade`, `site`, `site_grade` (default: `site`) |
| `--output` | Save to Excel (.xlsx) or CSV (.csv) |
| `--model ets` | Use only ETS model |
| `--model snaive` | Use only Seasonal Naive model |
| `--min-months 12` | Lower the data threshold (default: 24 months) |
| `--include-all` | Include sites with insufficient data |

### Models

- **ETS** (Holt-Winters): Exponential smoothing with trend and seasonality
- **Seasonal Naive**: Uses same-month-last-year values
- **ENSEMBLE**: Median of the above two. This is the recommended value for decisions.

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
| Forecasts | All model results. Columns: `site_id, grade, target_month, model, forecast_volume, prior_year_month, prior_year_volume, yoy_change_pct, ...` |
| Site Summary | Site-level totals reconciled from grade sums (only for `--by site_grade`) |
| Product Summary | Grade-level aggregation with YoY % |
| BU Summary | Overall business unit total with YoY % |
| Skipped | Sites skipped due to insufficient data |
| Summary | Statistics by model (count, total, average, min, max) |

### CSV (.csv)

Main forecasts go to the file you specify. Additional files are created alongside:
- `<name>_site_summary.csv`
- `<name>_product_summary.csv`
- `<name>_bu_summary.csv`
- `<name>_skipped.csv`
- `<name>_summary.csv`

## Examples

### Site-level with Excel export

```bash
python cli.py forecast 2026-03 --by site --output forecasts/2026-03_site.xlsx

# Progress: 80/400 sites
# Progress: 160/400 sites
# ...
```

### Detailed site-grade forecasts

```bash
python cli.py forecast 2026-03 --by site_grade --output forecasts/2026-03_detailed.xlsx

# 400 sites x 3 grades = ~1,200 combinations
```

### Include sites with limited data

```bash
# Lower the threshold to 12 months
python cli.py forecast 2026-03 --by site --min-months 12 --output forecasts/lenient.xlsx

# Or include everything regardless
python cli.py forecast 2026-03 --by site --include-all --output forecasts/all.xlsx
```

### Weekly data update workflow

```bash
# 1. Load new export from PDI
python cli.py load --file data/FuelVolume_2026_H1.xlsx

# 2. Quick sanity check
python cli.py status

# 3. Generate forecast
python cli.py forecast 2026-04 --by site --output forecasts/2026-04_site.xlsx
```

## Expected Runtimes

| Level | Approximate time |
|-------|-----------------|
| `grade` | Under 1 minute |
| `site` (~400 sites) | 5-10 minutes |
| `site_grade` (~1,200 combos) | 15-30 minutes |

## Best Practices

1. Load data weekly after PDI exports
2. Use ENSEMBLE for production decisions
3. Run `status --detailed` regularly to monitor data quality
4. Require 24+ months of data for reliable forecasts
5. Review the Skipped sheet to understand gaps
6. Keep past forecasts for accuracy tracking
7. Spot-check forecasts against recent actuals

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

### Permission errors
Run PowerShell as Administrator.

## File Locations

| Path | Purpose |
|------|---------|
| `fuel_sales.db` | SQLite database (auto-created on first load) |
| `data/` | Put PDI Excel/CSV exports here |
| `forecasts/` | Forecast output files |
