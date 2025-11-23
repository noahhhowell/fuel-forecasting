# Fuel Forecasting System

Forecast gas station fuel volumes with two stable models (ETS + Seasonal Naive) and an ensemble median. Supports Excel/CSV ingest, SQLite storage, and Excel/CSV forecast exports.

## Quick Start
```bash
# Install deps (recommended)
uv sync

# Load data (Excel skips first 5 rows by default)
python cli.py load --file data/sales_2024.xlsx
python cli.py load --file data/sales_2024.csv

# Check data quality
python cli.py status --detailed

# Forecast next month (aggregate)
python cli.py forecast 2025-07

# Forecast by site (Excel output with ordered columns)
python cli.py forecast 2025-07 --by site --output forecasts/2025-07_sites.xlsx

# Forecast by site-grade to CSV
python cli.py forecast 2025-07 --by site_grade --output forecasts/2025-07_detailed.csv
```

## Forecast Levels
| Level | Description | Typical Rows |
|-------|-------------|--------------|
| total | One forecast for all sites/grades | 3 (2 models + ensemble) |
| grade | One per fuel grade | 3 × #grades |
| site | One per site (all grades combined) | 3 × #sites |
| site_grade | One per site-grade combo | 3 × #combinations |

## Output Format
- **Excel**: Sheets = Forecasts, Skipped, Summary. Forecast columns are ordered `site_id, grade, target_month, model, forecast_volume` (extra columns follow).
- **CSV**: Main forecasts to your file, with `*_skipped.csv` and `*_summary.csv` alongside.
- Ensemble (median) is recommended for decisions.

## Common Options
- `--include-all` to bypass the default 24-month data requirement.
- `--min-months 12` to lower the threshold.
- `--model ets` or `--model snaive` to run a single model.
- `--by site_grade` for the most detailed forecasts (slowest).

## Data Loading
- Excel: headers expected on row 5 (0-indexed skiprows=4), flexible casing/spacing in column names.
- CSV: same columns in any case (`Brand, Site Id, Grade, Day, Volume, ...`).
- Deduplication is automatic (primary key: site_id, grade, day).

## Tips
- Start with aggregate or site-level before site-grade.
- Use `status --detailed` before forecasting to see data sufficiency.
- Load new data weekly; forecast monthly 1–2 months ahead.
- Keep forecasts for accuracy tracking; check the Skipped sheet to understand gaps.

More scenarios: [EXAMPLES.md](EXAMPLES.md). Cheatsheet: [QUICKREF.md](QUICKREF.md).
