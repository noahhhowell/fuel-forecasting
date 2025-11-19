# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fuel Forecasting System v2.1 - A professional forecasting system for gas station fuel volume predictions using 2 fast, stable models (ETS/Holt-Winters, Seasonal Naive). The system provides automatic ensemble predictions, smart deduplication, progress tracking, and data quality checks.

## Core Architecture

### Module Structure

The codebase is organized into 4 main Python modules with clear separation of concerns:

1. **database.py** - SQLite operations with automatic deduplication
   - `FuelDatabase` class manages all data persistence
   - Composite primary key (site_id, grade, day) prevents duplicates
   - Flexible column mapping handles various Excel formats
   - Default header row is 4 (0-indexed = row 5 in Excel)

2. **models.py** - Two forecasting models with unified interface
   - Abstract base class `ForecastModel` defines fit/predict contract
   - **ETS (Holt-Winters)**: Adaptive multiplicative/additive seasonality based on CV, damped trend, fast fitting
   - **Seasonal Naive**: Ultra-fast baseline using same-month-last-year logic, zero-tuning fallback
   - All models adapt to data length (disable seasonality if <24 months)
   - ETS includes validation bounds to prevent extreme forecasts

3. **forecaster.py** - High-level forecasting orchestration
   - `FuelForecaster` class provides user-facing API
   - Handles monthly data aggregation from daily sales
   - Progress tracking for bulk forecasts (reports every 20 sites or 50 site-grade combos)
   - Data sufficiency checks before forecasting
   - Ensemble prediction = robust median of all successful models (more stable than mean)
   - Multi-sheet Excel output: Forecasts, Skipped items, Summary

4. **cli.py** - Command-line interface with argparse
   - Four main commands: load, status, export, forecast
   - All user interactions flow through this entry point
   - Database path defaults to 'fuel_sales.db' in current directory

### Data Flow

```
Excel files → database.py (load_from_excel) → SQLite (fuel_sales.db)
                                                      ↓
                        forecaster.py (prepare_monthly_data) aggregates daily → monthly
                                                      ↓
                        models.py (fit) trains all available models
                                                      ↓
                        models.py (predict) generates forecasts
                                                      ↓
                        forecaster.py creates ensemble + exports to Excel
```

### Forecasting Levels

The `by` parameter in forecasts determines aggregation:
- **total**: One forecast for all sites/grades (5 rows: 4 models + ensemble)
- **grade**: One forecast per fuel type (5 × number of grades)
- **site**: One forecast per site, all grades combined (5 × number of sites)
- **site_grade**: Separate forecast for each site-grade combo (5 × number of combinations)

Most detailed forecasts (site_grade) can take 15-30 minutes for 400 sites with 3 grades = 1,200 combinations.

### Data Quality Requirements

- Default minimum: 24 months of data (controlled by `--min-months`)
- Recommended: 24 months for reliable forecasts with seasonal models
- Models adapt behavior based on available data:
  - <24 months: Disable seasonal components in ETS
  - <12 months: Seasonal Naive falls back to naive last-value
- `skip_insufficient=True` (default) skips items with insufficient data
- Skipped items are reported in Excel "Skipped" sheet with reasons

## Common Commands

### Setup and Data Loading
```bash
# Install dependencies
uv sync

# Load single file
python cli.py load --file sales_2024.xlsx

# Load all files in directory (automatically deduplicates)
python cli.py load --directory ./data

# Check database status
python cli.py status
python cli.py status --detailed  # Shows data quality per site
```

### Export to CSV
```bash
# Export all data
python cli.py export --output fuel_data.csv

# Export with filters
python cli.py export --output 2024.csv --start-date 2024-01-01 --end-date 2024-12-31
python cli.py export --output site_123.csv --site-id 123
python cli.py export --output unl.csv --grade UNL

# Include estimated values (excluded by default)
python cli.py export --output all_data.csv --include-estimated

# Combine multiple filters
python cli.py export --output filtered.csv --site-id 123 --grade UNL --start-date 2024-01-01
```

### Forecasting
```bash
# Aggregate forecast (fastest, 6 rows)
python cli.py forecast 2026-01

# By site (default behavior)
python cli.py forecast 2026-01 --by site --output jan_2026.xlsx

# By site-grade (most detailed, slowest)
python cli.py forecast 2026-01 --by site_grade --output jan_2026_detailed.xlsx

# Specific model only (no ensemble)
python cli.py forecast 2026-01 --model ets --output ets_only.xlsx
python cli.py forecast 2026-01 --model snaive --output snaive_only.xlsx

# Include all sites regardless of data quality
python cli.py forecast 2026-01 --by site --include-all

# Lower minimum data requirement
python cli.py forecast 2026-01 --by site --min-months 12
```

### Model Evaluation
```bash
# Generate a trial forecast and review results in Excel
python cli.py forecast 2026-01 --by site --output trial_forecast.xlsx

# After actuals arrive, compare them in Excel to estimate MAPE/MAE
```

### Testing
```bash
# Run installation test
python test_install.py
```

## Development Guidelines

### Model Behavior

All models must:
- Implement `fit(data: pd.DataFrame)` and `predict(periods: int) -> pd.DataFrame`
- Handle limited data gracefully (adapt parameters, don't just fail)
- Return predictions as DataFrame with columns: date, forecast, model
- Raise clear exceptions on failure (caught and logged by forecaster)

ETS specifically:
- Validates forecasts are within 3 standard deviations of training mean
- Clips extreme values at boundaries to prevent nonsensical predictions
- Auto-selects multiplicative vs additive seasonality based on coefficient of variation (CV > 0.3 → mul)
- Uses damped trend for stability

Seasonal Naive:
- Zero-tuning baseline: forecasts = same month last year
- Fastest model (~0 CPU cost), often competitive for stable seasonal patterns
- Included in ensemble to stabilize against model failures

### Database Operations

- **Never update database schema in production** without migration strategy
- Primary key (site_id, grade, day) is immutable
- Use `INSERT OR IGNORE` for automatic deduplication
- All dates stored as YYYY-MM-DD strings
- `is_estimated` column filters unreliable data (default: exclude estimated values)

### Error Handling Philosophy

- Individual site/model failures should NOT stop bulk operations
- Failed forecasts are logged and added to "Skipped" sheet
- Progress continues with remaining items
- Only raise exceptions at orchestration level if ALL items fail

### Progress Tracking

Bulk forecasts report progress:
- Sites: Every 20 sites or at completion
- Site-grade: Every 50 combinations or at completion
- Format: "Progress: X/Y sites"

### Excel Output Structure

All forecast exports have 3 sheets:
1. **Forecasts**: Main results with all models + ensemble
2. **Skipped**: Items that failed with reasons (site_id, reason)
3. **Summary**: Aggregated statistics by model (count, sum, mean, min, max)

## Python API Usage

For programmatic access beyond CLI:

```python
from database import FuelDatabase
from forecaster import FuelForecaster

# Initialize
db = FuelDatabase('fuel_sales.db')
forecaster = FuelForecaster(db, min_months_data=24)

# Generate forecast
forecast = forecaster.generate_forecast(
    target_month='2026-01',
    site_id='4551',  # Optional: None for all sites
    grade='UNL',     # Optional: None for all grades
    models_to_use=['ets']  # Optional: None for all models
)

# Bulk forecasting
bulk_forecast = forecaster.generate_bulk_forecasts(
    target_month='2026-01',
    by='site_grade',  # 'total', 'grade', 'site', or 'site_grade'
    output_path='forecast.xlsx',
    skip_insufficient=True
)

# Clean up
db.close()
```

## Important Implementation Details

### Model Availability
Models are conditionally imported at runtime. Check availability with:
```python
from models import get_available_models
available = get_available_models()  # Returns dict of available model classes
```

### Date Handling
- Target month format: 'YYYY-MM' string
- Internally converted to pandas datetime (first day of month)
- Forecasts are always for end of month volumes
- Uses `dateutil.relativedelta` for month arithmetic

### Ensemble Logic
- Ensemble = robust median of all successful model predictions
- Median is more robust to outlier models than arithmetic mean
- If a model fails, it's excluded from ensemble (doesn't cause ensemble failure)
- Ensemble added as separate row with model='ENSEMBLE'
- **Recommended for production use** - more robust than individual models

## Windows-Specific Notes

- Uses `uv` as package manager (preferred over pip)
- File paths may include OneDrive paths (handle spaces in paths)
- Database file (fuel_sales.db) is ~720MB with typical data
