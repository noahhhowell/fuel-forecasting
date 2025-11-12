# Setup Guide - Windows with uv

This guide will help you set up the Fuel Forecasting System on your Windows laptop using uv.

## Prerequisites

- Windows 10/11
- Python 3.9 or higher
- Internet connection

## Step 1: Install uv (if not already installed)

Open PowerShell or Command Prompt and run:

```powershell
pip install uv
```

Or download from: https://github.com/astral-sh/uv

## Step 2: Setup Project

1. **Create project directory**:
```powershell
mkdir C:\fuel-forecasting
cd C:\fuel-forecasting
```

2. **Copy all Python files** to this directory:
   - `pyproject.toml`
   - `database.py`
   - `models.py`
   - `forecaster.py`
   - `cli.py`
   - `README.md`

3. **Install dependencies**:
```powershell
uv sync
```

This will:
- Create a virtual environment at `.venv`
- Install all required packages
- Download Python if needed

## Step 3: Create Data Directories

```powershell
mkdir data
mkdir forecasts
```

## Step 4: Test Installation

```powershell
uv run python cli.py --help
```

You should see the help text with all available commands.

## Step 5: Load Your First Data File

1. **Export data from PDI** and save to the `data` folder

2. **Load it**:
```powershell
uv run python cli.py load --file data\sales_2024.xlsx
```

3. **Check status**:
```powershell
uv run python cli.py status
```

## Step 6: Generate Your First Forecast

```powershell
uv run python cli.py forecast 2026-01 --output forecasts\jan_2026.xlsx
```

## Running Commands

Two ways to run commands:

### Option 1: With uv (recommended)
```powershell
uv run python cli.py <command>
```

### Option 2: Activate venv manually
```powershell
# Activate
.\.venv\Scripts\activate

# Now you can run directly
python cli.py <command>

# Deactivate when done
deactivate
```

## Common Issues

### "uv not found"
Install uv: `pip install uv`

### "Python not found"
Install Python 3.9+ from python.org

### "No module named 'prophet'"
Run: `uv sync` to install all dependencies

### Permission errors
Run PowerShell as Administrator

## Next Steps

1. Read [README.md](README.md) for full documentation
2. Load all your historical data
3. Run a backtest to evaluate models
4. Generate your first production forecast

## Daily Usage

Once set up, your typical workflow is:

```powershell
cd C:\fuel-forecasting

# Load new data
uv run python cli.py load --file data\latest.xlsx

# Generate forecast
uv run python cli.py forecast 2026-01 --by site_grade --output forecasts\jan_2026.xlsx
```

## Updating Dependencies

If you need to add or update packages:

```powershell
# Add a package
uv add package-name

# Update all packages
uv sync --upgrade
```

## Support

For issues, check:
1. README.md troubleshooting section
2. Run with `--help` flag
3. Check the output Excel "Skipped" sheet

---

**Ready to start!** Run: `uv run python cli.py status`
