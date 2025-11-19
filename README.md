# Fuel Forecasting System v2.0

Clean, professional forecasting system for gas station fuel volume with 4 state-of-the-art models.

## ✨ Features

- **4 Forecasting Models**: SARIMA, Exponential Smoothing, Prophet, XGBoost
- **Automatic Ensemble**: Combines all models for robust predictions
- **Smart Deduplication**: Never worry about loading duplicate data
- **Progress Tracking**: See exactly what's happening during bulk forecasts
- **Data Quality Checks**: Skip sites with insufficient data automatically
- **Flexible Output**: Forecast by total, grade, site, or site-grade combinations
- **Professional Excel Exports**: Multi-sheet workbooks with summaries

## 🚀 Quick Start

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install pandas numpy openpyxl statsmodels prophet xgboost python-dateutil
```

### 2. Load Your Data

```bash
# Load a single file
python cli.py load --file sales_2024.xlsx

# Load all files in a directory
python cli.py load --directory ./data
```

### 3. Check Status

```bash
python cli.py status --detailed
```

### 4. Generate Forecast

```bash
# Forecast for all sites/grades (aggregate)
python cli.py forecast 2026-01

# Forecast by site (all grades combined per site)
python cli.py forecast 2026-01 --by site --output jan_2026_by_site.xlsx

# Forecast by site-grade (separate forecast for each combination)
python cli.py forecast 2026-01 --by site_grade --output jan_2026_detailed.xlsx
```

## 📊 Forecasting Levels

| Level | What It Does | Use Case | Output Rows |
|-------|-------------|----------|-------------|
| `total` | One forecast for all sites/grades | Company-wide planning | 5 (4 models + ensemble) |
| `grade` | One forecast per fuel type | Grade-level planning | 5 × # grades |
| `site` | One forecast per site (all grades combined) | Site-level planning | 5 × # sites |
| `site_grade` | One forecast per site-grade combo | Detailed planning | 5 × # combinations |

## 🎯 Monthly Workflow

```bash
# 1. Load new data (weekly)
python cli.py load --file sales_latest.xlsx

# 2. Generate forecast (monthly, 2 months ahead)
python cli.py forecast 2026-03 --by site_grade --output mar_2026.xlsx
```

## 🔧 Common Commands

```bash
# Load data
python cli.py load --file sales_2024.xlsx
python cli.py load --directory ./data

# Check status
python cli.py status
python cli.py status --detailed

# Forecast (aggregate)
python cli.py forecast 2026-01

# Forecast by site-grade (detailed)
python cli.py forecast 2026-01 --by site_grade --output forecast.xlsx

# Use specific model
python cli.py forecast 2026-01 --model prophet
```

## 📈 Understanding Output

Forecast Excel file has 3 sheets:

1. **Forecasts**: All models + ENSEMBLE (recommended)
2. **Skipped**: Sites with insufficient data
3. **Summary**: Aggregated statistics

Use **ENSEMBLE** for production - it averages all models.

## 🐛 Troubleshooting

**"Insufficient data" error?**
```bash
# Option 1: Lower requirement
python cli.py forecast 2026-01 --by site --min-months 12

# Option 2: Include all sites
python cli.py forecast 2026-01 --by site --include-all

# Option 3: Check which sites have issues
python cli.py status --detailed
```

**Model not available?**
```bash
uv add prophet xgboost
```

**Slow with 400 sites?**
- Normal! Site-grade forecasts take 15-30 minutes
- Progress tracker shows exactly where it is

## 💡 Tips

1. Start simple: `python cli.py forecast 2026-01`
2. Use ENSEMBLE for production forecasts
3. Check data quality: `python cli.py status --detailed`
4. Keep database updated weekly

## 📋 Full Documentation

See [EXAMPLES.md](EXAMPLES.md) for detailed command examples and use cases.

---

**Questions?** Run `python cli.py --help`
