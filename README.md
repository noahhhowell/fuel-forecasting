# Fuel Forecasting System

Clean, professional forecasting system for gas station fuel volume predictions using fast, stable models.

## ✨ Features

- **2 Forecasting Models**: ETS (Holt-Winters) and Seasonal Naive
- **Automatic Ensemble**: Combines models using robust median for stable predictions
- **Smart Deduplication**: Never worry about loading duplicate data
- **Progress Tracking**: See exactly what's happening during bulk forecasts
- **Data Quality Checks**: Skip sites with insufficient data automatically
- **Outlier Handling**: Automatic detection and capping using MAD method
- **Flexible Output**: Forecast by total, grade, site, or site-grade combinations
- **Professional Excel Exports**: Multi-sheet workbooks with summaries and skipped items
- **CSV Export**: Export filtered data with flexible date/site/grade filters

## 🚀 Quick Start

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install pandas numpy openpyxl statsmodels python-dateutil
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
| `total` | One forecast for all sites/grades | Company-wide planning | 3 (2 models + ensemble) |
| `grade` | One forecast per fuel type | Grade-level planning | 3 × # grades |
| `site` | One forecast per site (all grades combined) | Site-level planning | 3 × # sites |
| `site_grade` | One forecast per site-grade combo | Detailed planning | 3 × # combinations |

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
python cli.py forecast 2026-01 --model ets --output ets_only.xlsx
python cli.py forecast 2026-01 --model snaive --output snaive_only.xlsx

# Export data to CSV
python cli.py export --output fuel_data.csv
python cli.py export --output 2024.csv --start-date 2024-01-01 --end-date 2024-12-31
```

## 📈 Understanding Output

Forecast Excel file has 3 sheets:

1. **Forecasts**: All models + ENSEMBLE (recommended)
2. **Skipped**: Sites with insufficient data and reasons
3. **Summary**: Aggregated statistics by model

Use **ENSEMBLE** for production - it uses the robust median of all models, providing more stable predictions than individual models.

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

**Slow with 400 sites?**
- Normal! Site-grade forecasts take 15-30 minutes for 400 sites × 3 grades
- Progress tracker shows exactly where it is (updates every 20 sites or 50 combinations)
- The ETS model is fast but still needs time for 1,200+ fits

## 💡 Tips

1. **Start simple**: Begin with aggregate forecasts (`python cli.py forecast 2026-01`)
2. **Use ENSEMBLE**: The ensemble median is more robust than individual models
3. **Check data quality**: Run `python cli.py status --detailed` before bulk forecasts
4. **Update weekly**: Load new data weekly for better forecast accuracy
5. **24-month minimum**: Sites with less than 24 months of data produce unreliable forecasts
6. **Monitor outliers**: The system auto-caps outliers using MAD method - check logs for details
7. **Export for analysis**: Use `python cli.py export` to analyze historical data in Excel/Python

## 📋 Full Documentation

See [EXAMPLES.md](EXAMPLES.md) for detailed command examples and use cases.

---

**Questions?** Run `python cli.py --help`
