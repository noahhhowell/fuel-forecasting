# GEMINI.md: Fuel Forecasting System

This file provides a comprehensive overview of the Fuel Forecasting System for Gemini.

## Project Overview

This is a Python-based command-line application designed for forecasting fuel sales volume. It provides a clean, professional interface for loading sales data, generating forecasts with multiple models, and evaluating model performance.

**Core Technologies:**

*   **CLI:** `argparse`
*   **Database:** `sqlite3`
*   **Data Manipulation:** `pandas`, `numpy`
*   **Forecasting Models:**
    *   `statsmodels` (for ETS/Holt-Winters)
    *   A custom `SeasonalNaiveModel`
*   **Excel I/O:** `openpyxl`

**Architecture:**

The application is structured into four main modules:

1.  `cli.py`: The main entry point and user interface. It handles command-line argument parsing and orchestrates the overall workflow.
2.  `database.py`: A `FuelDatabase` class that manages all interactions with the SQLite database. This includes data loading, deduplication, and querying.
3.  `forecaster.py`: A `FuelForecaster` class that contains the core forecasting logic. It prepares data, trains the models, and generates predictions.
4.  `models.py`: Defines the forecasting models. It includes a base `ForecastModel` class and implementations for Exponential Smoothing (ETS) and a Seasonal Naive model.

## Building and Running

### 1. Install Dependencies

The project uses `uv` for dependency management.

```bash
uv sync
```

Alternatively, you can use `pip`:

```bash
pip install pandas numpy openpyxl statsmodels python-dateutil
```

### 2. Running the Application

The application is run via `cli.py`. Here are the main commands:

*   **Load Data:** Load sales data from Excel files into the SQLite database.

    ```bash
    # Load a single file
    python cli.py load --file path/to/your/sales_data.xlsx

    # Load all files in a directory
    python cli.py load --directory ./data
    ```

*   **Check Status:** Display a summary of the data in the database.

    ```bash
    python cli.py status
    python cli.py status --detailed
    ```

*   **Generate Forecasts:** Create forecasts for a future month.

    ```bash
    # Forecast for all sites (aggregated)
    python cli.py forecast YYYY-MM

    # Forecast by site and grade (most detailed)
    python cli.py forecast YYYY-MM --by site_grade --output forecast.xlsx
    ```

## Development Conventions

*   **Code Style:** The code follows standard Python conventions (PEP 8).
*   **Typing:** The code uses type hints for clarity and static analysis.
*   **Modularity:** The code is well-structured into distinct modules with clear responsibilities.
*   **Error Handling:** The application includes error handling to gracefully manage issues like missing files or insufficient data.
*   **Logging:** The `logging` module is used to provide informative output during execution.
