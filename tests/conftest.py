"""
Shared test fixtures.

pytest automatically discovers this file and makes its fixtures available
to every test file in the directory. A "fixture" is just reusable setup
code — e.g., a temporary database pre-loaded with sample data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import FuelDatabase
from forecaster import FuelForecaster


def _build_sample_sales(
    site_ids=("100", "200"),
    grades=("UNL", "DSL"),
    start="2022-01-01",
    months=36,
):
    """
    Generate realistic-looking daily fuel sales records.

    Returns a DataFrame that looks like what the database would store:
    one row per site/grade/day with a plausible volume.
    """
    rows = []
    dates = pd.date_range(start, periods=months * 30, freq="D")
    # Keep only ~30 days per month (trim to month boundaries)
    dates = dates[dates < pd.Timestamp(start) + pd.DateOffset(months=months)]

    rng = np.random.default_rng(42)  # fixed seed for reproducible tests

    for site_id in site_ids:
        base_volume = rng.uniform(300, 800)
        for grade in grades:
            grade_factor = 1.0 if grade == "UNL" else 0.6
            for day in dates:
                # Add seasonality: higher in summer, lower in winter
                month_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (day.month - 1) / 12)
                volume = base_volume * grade_factor * month_factor
                volume += rng.normal(0, base_volume * 0.05)  # noise
                volume = max(0.1, volume)  # no zeros

                rows.append(
                    {
                        "site_id": site_id,
                        "grade": grade,
                        "day": day.strftime("%Y-%m-%d"),
                        "volume": round(volume, 2),
                        "is_estimated": 0,
                        "brand": "TestBrand",
                        "site": f"Site {site_id}",
                        "address": "123 Test St",
                        "city": "TestCity",
                        "state": "TX",
                        "owner": "TestOwner",
                        "b_unit": "BU1",
                        "stock": None,
                        "delivered": None,
                        "total_sales": None,
                        "target": None,
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_sales_df():
    """Raw sample sales DataFrame (not in a database yet)."""
    return _build_sample_sales()


@pytest.fixture
def db(tmp_path):
    """
    An in-memory-style temp database pre-loaded with 36 months of sample data.

    tmp_path is a pytest built-in fixture that gives each test its own
    temporary directory — the database file is created there and cleaned
    up automatically after the test.
    """
    db_path = str(tmp_path / "test_fuel_sales.db")
    database = FuelDatabase(db_path)

    # Insert sample data directly via SQL (bypass Excel loading logic)
    df = _build_sample_sales()
    records = df[
        [
            "site_id", "grade", "day", "brand", "site", "address",
            "city", "state", "owner", "b_unit", "stock", "delivered",
            "volume", "is_estimated", "total_sales", "target",
        ]
    ].itertuples(index=False, name=None)

    sql = """
        INSERT OR IGNORE INTO sales (
            site_id, grade, day, brand, site, address, city, state,
            owner, b_unit, stock, delivered, volume, is_estimated,
            total_sales, target
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with database.conn:
        database.conn.executemany(sql, records)

    yield database
    database.close()


@pytest.fixture
def forecaster(db):
    """A FuelForecaster wired to the test database."""
    return FuelForecaster(db, min_months_data=24)
