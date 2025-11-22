# Fuel Forecasting Examples

Real-world examples for common forecasting tasks.

## Example 1: First Time Setup

```bash
# Load all historical data
python cli.py load --directory ./data

# Check what you have
python cli.py status --detailed

# Generate simple test forecast
python cli.py forecast 2026-01
```

## Example 2: Monthly Forecast Process

```bash
# Every month, you need forecasts 2 months ahead

# January: forecast for March
python cli.py forecast 2026-03 --by site_grade --output forecasts/mar_2026.xlsx

# February: forecast for April
python cli.py forecast 2026-04 --by site_grade --output forecasts/apr_2026.xlsx

# And so on...
```

## Example 3: Quick Aggregate Forecast

```bash
# Just need total company-wide number
python cli.py forecast 2026-01

# Output will show:
# ETS:                     1,234,567.89 gallons
# SeasonalNaive:           1,238,901.23 gallons
# ──────────────────────────────────────────────
# RECOMMENDED (Ensemble):  1,236,734.56 gallons  ← Use this (median)
```

## Example 4: Forecast by Fuel Grade

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

## Example 5: Individual Site Forecasts

```bash
# Forecast for each site (all grades combined)
python cli.py forecast 2026-01 --by site --output sites_jan_2026.xlsx

# Takes ~5-10 minutes for 400 sites
# Progress tracker will show:
#   Progress: 80/400 sites
#   Progress: 160/400 sites
#   ...
```

## Example 6: Detailed Site-Grade Forecasts

```bash
# Most detailed: separate forecast for each site-grade combo
python cli.py forecast 2026-01 --by site_grade --output detailed_jan_2026.xlsx

# With 400 sites × 3 grades = 1,200 combinations
# Takes ~15-30 minutes
# Excel will have 1,200 × 3 = 3,600 rows (3 per combination: ETS, SeasonalNaive, ENSEMBLE)
```

## Example 7: Only High-Quality Sites

```bash
# Only forecast sites with 24+ months of data (default behavior)
python cli.py forecast 2026-01 --by site --output quality_sites.xlsx

# See what was skipped in the Excel "Skipped" sheet
```

## Example 8: Include All Sites

```bash
# Forecast ALL sites, even those with limited data
python cli.py forecast 2026-01 --by site --include-all --output all_sites.xlsx

# Or lower the threshold
python cli.py forecast 2026-01 --by site --min-months 12 --output sites_12mo.xlsx
```

## Example 9: Use Specific Model Only

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

## Example 10: Weekly Data Updates

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

## Example 11: Export Data to CSV

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

## Example 12: Data Quality Check

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

## Example 13: Multi-File Load

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
# Total duplicates: 234
```

## Example 14: Production Workflow

```bash
# Monthly production workflow script

# Step 1: Load latest data (first Monday of month)
python cli.py load --file data/october_2025.xlsx

# Step 2: Generate forecast for 2 months ahead
python cli.py forecast 2026-01 --by site_grade --output production/jan_2026_forecast.xlsx

# Step 3: Verify data quality
python cli.py status --detailed > production/quality_report.txt

# Step 4: Archive last month's forecast
# (manually compare actuals vs predictions)
```

## Example 15: Handling New Sites

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

## Example 16: Custom Thresholds

```bash
# Default requires 24 months, but you can adjust:

# More strict (36 months)
python cli.py forecast 2026-01 --by site --min-months 36 --output conservative.xlsx

# More lenient (12 months)
python cli.py forecast 2026-01 --by site --min-months 12 --output lenient.xlsx

# Very lenient (6 months) - use with extreme caution
python cli.py forecast 2026-01 --by site --min-months 6 --output risky.xlsx
```

## Example 17: Comparing Methods

```bash
# Generate forecasts with different methods to compare

# Method 1: Aggregate then distribute
python cli.py forecast 2026-01 --output total_forecast.xlsx
# Then manually distribute to sites based on historical %

# Method 2: Individual sites
python cli.py forecast 2026-01 --by site --output site_forecasts.xlsx
# Sum these up for total

# Compare the two approaches - should be similar!
```

## Example 18: Recovery from Errors

```bash
# If a forecast fails partway through:

# 1. Check what succeeded
python cli.py status --detailed

# 2. Look at the Excel "Skipped" sheet to see what failed

# 3. Try those specific sites individually (not implemented in CLI, but possible in Python API)

# 4. Or just re-run with different settings
python cli.py forecast 2026-01 --by site --min-months 12 --output retry.xlsx
```

## Tips from the Examples

1. **Start simple**: Begin with aggregate forecasts before going detailed
2. **Validate first**: Compare forecasts to recent actuals before using them in production
3. **Use ensemble**: The ENSEMBLE median is more robust than individual models
4. **Check quality**: Use `--detailed` flag to spot data issues early
5. **Update weekly**: More data = better forecasts
6. **Archive results**: Keep old forecasts to track accuracy over time
7. **Monitor actuals**: After month-end, compare forecasts to actuals in Excel to gauge accuracy
8. **New sites**: Exclude until they have 24+ months of data
9. **Progress tracking**: For bulk forecasts, the progress counter is your friend
10. **Excel sheets**: Use the "Skipped" sheet to understand what didn't work
11. **Outlier handling**: Check logs for outlier capping - may indicate data quality issues
12. **Model differences**: ETS adapts to trends, Seasonal Naive is simpler but often competitive

## Python API Examples

If you need more control, use the Python API:

```python
from database import FuelDatabase
from forecaster import FuelForecaster

db = FuelDatabase('fuel_sales.db')
forecaster = FuelForecaster(db)

# Example: Forecast only high-volume sites
sites = db.get_site_data_quality()
high_volume = sites[sites['total_records'] > 1000]

forecasts = []
for _, site in high_volume.iterrows():
    try:
        forecast = forecaster.generate_forecast(
            target_month='2026-01',
            site_id=site['site_id']
        )
        forecasts.append(forecast)
    except Exception as e:
        print(f"Failed {site['site_id']}: {e}")

import pandas as pd
all_forecasts = pd.concat(forecasts)
all_forecasts.to_excel('custom_forecast.xlsx', index=False)

db.close()
```

---

**Need something not shown here?** Check `python cli.py --help` or the main README.md
