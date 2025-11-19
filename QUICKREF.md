# Quick Reference Card

## Installation (One-Time Setup)

```bash
# 1. Install dependencies
uv sync

# 2. Create directories
mkdir data forecasts
```

## Daily Commands

```bash
# Load new data
python cli.py load --file data/latest.xlsx

# Check status
python cli.py status

# Generate forecast (2 months ahead)
python cli.py forecast 2026-03 --by site_grade --output forecasts/mar_2026.xlsx
```

## Common Forecasts

```bash
# Total (aggregate)
python cli.py forecast 2026-01

# By site (400 sites)
python cli.py forecast 2026-01 --by site --output jan_sites.xlsx

# By site-grade (1200 combinations)
python cli.py forecast 2026-01 --by site_grade --output jan_detailed.xlsx
```

## Models

- **SARIMA**: Statistical, seasonal patterns
- **Exponential Smoothing**: Fast, smooth trends
- **Prophet**: Robust to missing data
- **XGBoost**: High accuracy, non-linear
- **ENSEMBLE**: ⭐ Recommended (averages all)

## Output Format

Excel with 3 sheets:
1. **Forecasts**: All models + ENSEMBLE
2. **Skipped**: Insufficient data
3. **Summary**: Statistics by model

## Troubleshooting

```bash
# Insufficient data error
python cli.py forecast 2026-01 --by site --min-months 12

# Include all sites
python cli.py forecast 2026-01 --by site --include-all

# Check which sites have issues
python cli.py status --detailed
```

## Model Evaluation

```bash
# Generate a trial forecast and save to Excel
python cli.py forecast 2026-01 --by site --output trial_forecast.xlsx

# After actuals arrive, compare them against the forecast in Excel
# Good: MAPE < 5%
# Acceptable: MAPE 5-10%
# Poor: MAPE > 10% (investigate data quality)
```

## File Locations

- **Database**: `fuel_sales.db` (auto-created)
- **Excel data**: Put in `data/` folder
- **Forecasts**: Saved to `forecasts/` folder

## Using with uv

All commands can be prefixed with `uv run`:
```bash
uv run python cli.py forecast 2026-01
```

Or activate the venv:
```bash
.\.venv\Scripts\activate
python cli.py forecast 2026-01
```

## Help

```bash
python cli.py --help
python cli.py forecast --help
```

## What Each Forecast Level Does

| Level | Example | Output Rows |
|-------|---------|-------------|
| `total` | All sites/grades combined | 5 |
| `grade` | UNL, PRE, DSL | 15 |
| `site` | Each site (all grades) | 2,000 |
| `site_grade` | Each site-grade combo | 6,000 |

## Expected Runtimes

- Total/Grade: < 1 minute
- By site (400): 5-10 minutes
- By site-grade (1200): 15-30 minutes

Progress tracker shows exactly where it is.

## Best Practices

1. ✅ Load data weekly
2. ✅ Use ENSEMBLE for production
3. ✅ Check `status --detailed` regularly
4. ✅ Require 24+ months of data
5. ✅ Review "Skipped" sheet in output
6. ✅ Keep forecasts for accuracy tracking
7. ✅ Spot-check forecasts against recent actuals

---

**Full docs**: See README.md and EXAMPLES.md
