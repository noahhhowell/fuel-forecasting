# Fuel Forecasting Examples

Focused examples for common tasks. Replace dates/paths with your own.

## First-Time Setup

```bash
# Load all historical data
python cli.py load --directory ./data

# Check what you have
python cli.py status --detailed

# Generate simple test forecast
python cli.py forecast 2026-01
```

## Quick Aggregate Forecast

```bash
# Just need total company-wide number
python cli.py forecast 2026-01

# Output will show:
# ETS:                     1,234,567.89 gallons
# SeasonalNaive:           1,238,901.23 gallons
# ──────────────────────────────────────────────
# RECOMMENDED (Ensemble):  1,236,734.56 gallons  ← Use this (median)
```

## Forecast by Fuel Grade

```bash
# Need to know UNL vs PRE vs DSL volumes
python cli.py forecast 2026-01 --by grade --output grades_jan_2026.xlsx

# Excel will have:
# Grade | Model         | Forecast
# UNL   | ETS           | 800,000
# UNL   | SeasonalNaive | 805,000
# UNL   | ENSEMBLE      | 802,500  ← Use this for UNL
# PRE   | ETS           | 300,000
# PRE   | SeasonalNaive | 302,000
# PRE   | ENSEMBLE      | 301,000  ← Use this for PRE
# ... (3 rows per grade)
```

## Forecast by Site

```bash
# Forecast for each site (all grades combined)
python cli.py forecast 2026-01 --by site --output sites_jan_2026.xlsx

# Takes ~5-10 minutes for 400 sites
# Progress tracker will show:
#   Progress: 80/400 sites
#   Progress: 160/400 sites
#   ...
```

## Detailed Site-Grade Forecasts

```bash
# Most detailed: separate forecast for each site-grade combo
python cli.py forecast 2026-01 --by site_grade --output detailed_jan_2026.xlsx

# With 400 sites × 3 grades = 1,200 combinations
# Takes ~15-30 minutes
# Excel will have 1,200 × 3 = 3,600 rows (3 per combination: ETS, SeasonalNaive, ENSEMBLE)
```

## Only High-Quality Sites

```bash
# Only forecast sites with 24+ months of data (default behavior)
python cli.py forecast 2026-01 --by site --output quality_sites.xlsx

# See what was skipped in the Excel "Skipped" sheet
```

## Include All Sites

```bash
# Forecast ALL sites, even those with limited data
python cli.py forecast 2026-01 --by site --include-all --output all_sites.xlsx

# Or lower the threshold
python cli.py forecast 2026-01 --by site --min-months 12 --output sites_12mo.xlsx
```

## Use Specific Model Only

```bash
# Only use ETS model
python cli.py forecast 2026-01 --model ets --output ets_only.xlsx

# Only use Seasonal Naive model
python cli.py forecast 2026-01 --model snaive --output snaive_only.xlsx

# Compare both models individually
python cli.py forecast 2026-01 --by site --model ets --output sites_ets.xlsx
python cli.py forecast 2026-01 --by site --model snaive --output sites_snaive.xlsx
# Then compare against ensemble forecast
python cli.py forecast 2026-01 --by site --output sites_ensemble.xlsx
```

## Weekly Data Updates

```bash
# Every week after PDI export
python cli.py load --file data/week_ending_2025_11_01.xlsx

# Check what's new
python cli.py status

# Shows:
# Records:
#   • Total: 428,456 (+2,609 from last week)
#   • Non-estimated: 428,456
```

## Export Data to CSV

```bash
# Export all data
python cli.py export --output fuel_data.csv

# Export specific date range
python cli.py export --output 2024_data.csv --start-date 2024-01-01 --end-date 2024-12-31

# Export specific site
python cli.py export --output site_4551.csv --site-id 4551

# Export specific grade
python cli.py export --output unl_only.csv --grade UNL

# Combine filters
python cli.py export --output site_4551_unl_2024.csv \
  --site-id 4551 --grade UNL --start-date 2024-01-01 --end-date 2024-12-31

# Include estimated values (excluded by default)
python cli.py export --output with_estimated.csv --include-estimated
```

## Data Quality Check

```bash
python cli.py status --detailed

# Shows which sites have insufficient data:
# Sites with <24 months: 15
# site_id | site      | months_of_data
# 4889    | New Store | 8
# 4890    | New Store | 12
# 4891    | Recent    | 18
# ...
```

## Multi-File Load

```bash
# Load all yearly exports at once
python cli.py load --directory ./data

# Finds all .xlsx files and loads them
# Automatically skips duplicates

# Output:
# Found 4 Excel files
#
# Load Summary:
# file              inserted  duplicates
# sales_2021.xlsx   106,234   0
# sales_2022.xlsx   106,521   145  ← 145 overlapping records skipped
# sales_2023.xlsx   106,789   89
# sales_2024.xlsx   107,234   0
#
# Total inserted: 426,778
## Handling New Sites

```bash
# New sites won't have enough data yet
python cli.py status --detailed

# Shows:
# Sites with <12 months: 5
# site_id | site      | months_of_data
# 4900    | Brand New | 3
# 4901    | Recent    | 8
# ...

# Option 1: Skip them (default)
python cli.py forecast 2026-01 --by site --output with_quality.xlsx

# Option 2: Include them anyway (use with caution)
python cli.py forecast 2026-01 --by site --include-all --output all_sites.xlsx
# Their forecasts will be less reliable but might be better than nothing
```

## Custom Thresholds

```bash
# Default requires 24 months, but you can adjust:

# More strict (36 months)
python cli.py forecast 2026-01 --by site --min-months 36 --output conservative.xlsx

# More lenient (12 months)
python cli.py forecast 2026-01 --by site --min-months 12 --output lenient.xlsx

# Very lenient (6 months) - use with extreme caution
python cli.py forecast 2026-01 --by site --min-months 6 --output risky.xlsx
```

## Recovery from Errors

```bash
# If a forecast fails partway through:

# 1. Check what succeeded
python cli.py status --detailed

# 2. Look at the Excel "Skipped" sheet to see what failed
# 3. Re-run with different settings
python cli.py forecast 2026-01 --by site --min-months 12 --output retry.xlsx
```

**Need something not shown here?** See README.md or `python cli.py --help`.
