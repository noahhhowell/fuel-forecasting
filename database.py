"""
Database Module - SQLite operations with automatic deduplication
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class FuelDatabase:
    """Manages fuel sales data in SQLite"""

    def __init__(self, db_path: str = "fuel_sales.db", header_row: int = 4):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
            header_row: Row number where headers start (0-indexed, default 4 = row 5)
        """
        self.db_path = Path(db_path)
        self.header_row = header_row
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        """Create tables and indexes"""
        
        # Main sales table with composite primary key to prevent duplicates
        create_sales = """
        CREATE TABLE IF NOT EXISTS sales (
            site_id TEXT NOT NULL,
            grade TEXT NOT NULL,
            day DATE NOT NULL,
            brand TEXT,
            site TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            owner TEXT,
            b_unit TEXT,
            stock REAL,
            delivered REAL,
            volume REAL,
            is_estimated BOOLEAN,
            total_sales REAL,
            target REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (site_id, grade, day)
        )
        """
        
        # Indexes for query performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_day ON sales(day)",
            "CREATE INDEX IF NOT EXISTS idx_site_grade ON sales(site_id, grade)",
            "CREATE INDEX IF NOT EXISTS idx_site ON sales(site_id)",
            "CREATE INDEX IF NOT EXISTS idx_grade ON sales(grade)",
        ]
        
        # Metadata tracking
        create_metadata = """
        CREATE TABLE IF NOT EXISTS load_metadata (
            load_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            rows_loaded INTEGER,
            rows_duplicates INTEGER,
            load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        create_calibration_runs = """
        CREATE TABLE IF NOT EXISTS calibration_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            backtest_months INTEGER,
            horizon INTEGER,
            min_months INTEGER,
            sites_calibrated INTEGER,
            overall_mape REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        create_calibration_weights = """
        CREATE TABLE IF NOT EXISTS calibration_weights (
            site_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            weight REAL NOT NULL,
            mape_pct REAL,
            n_months INTEGER,
            run_id INTEGER NOT NULL REFERENCES calibration_runs(run_id),
            PRIMARY KEY (site_id, model_name)
        )
        """

        create_interval_calibration = """
        CREATE TABLE IF NOT EXISTS interval_calibration (
            segment TEXT NOT NULL PRIMARY KEY,
            residual_std REAL NOT NULL,
            residual_p10 REAL NOT NULL,
            residual_p90 REAL NOT NULL,
            n_observations INTEGER,
            run_id INTEGER NOT NULL REFERENCES calibration_runs(run_id)
        )
        """

        with self.conn:
            self.conn.execute(create_sales)
            for idx in indexes:
                self.conn.execute(idx)
            self.conn.execute(create_metadata)
            self.conn.execute(create_calibration_runs)
            self.conn.execute(create_calibration_weights)
            self.conn.execute(create_interval_calibration)

        logger.info(f"Database initialized: {self.db_path}")

    def load_from_excel(self, file_path: str) -> dict:
        """Load data from Excel file with automatic deduplication"""
        logger.info(f"Loading: {file_path}")
        df = pd.read_excel(file_path, skiprows=self.header_row)
        logger.info(f"  Read {len(df):,} rows from Excel")
        return self._load_dataframe(df, file_path)

    def load_from_csv(self, file_path: str) -> dict:
        """Load data from CSV file with automatic deduplication"""
        logger.info(f"Loading: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"  Read {len(df):,} rows from CSV")
        return self._load_dataframe(df, file_path)

    def _load_dataframe(self, df: pd.DataFrame, file_path: str) -> dict:
        """Normalize columns, validate, and insert rows with deduplication"""
        total_rows = len(df)
        df.columns = df.columns.str.strip()
        column_map = self._build_column_mapping(df.columns)
        df = df.rename(columns=column_map)

        required = ["site_id", "grade", "day", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Normalize key fields to avoid duplicate keys caused by mixed types/whitespace
        df["site_id"] = self._normalize_text_column(df["site_id"])
        df["grade"] = self._normalize_text_column(df["grade"], uppercase=True)
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        required_valid = (
            df["site_id"].notna()
            & df["grade"].notna()
            & df["day"].notna()
            & df["volume"].notna()
        )
        invalid = int((~required_valid).sum())
        if invalid:
            logger.warning(f"  Skipping {invalid:,} invalid row(s) with missing required fields")
        df = df.loc[required_valid].copy()
        if df.empty:
            raise ValueError("No valid rows after cleaning required fields")

        df["day"] = df["day"].dt.strftime("%Y-%m-%d")

        if "is_estimated" in df.columns:
            df["is_estimated"] = df["is_estimated"].apply(self._normalize_bool)
        else:
            df["is_estimated"] = False

        db_cols = [
            "site_id", "grade", "day", "brand", "site", "address", 
            "city", "state", "owner", "b_unit", "stock", "delivered",
            "volume", "is_estimated", "total_sales", "target"
        ]
        for col in db_cols:
            if col not in df.columns:
                df[col] = None

        count_before = self._get_count()
        records = df[db_cols].itertuples(index=False, name=None)

        sql = """
            INSERT OR IGNORE INTO sales (
                site_id, grade, day, brand, site, address, city, state,
                owner, b_unit, stock, delivered, volume, is_estimated,
                total_sales, target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.conn:
            self.conn.executemany(sql, records)

        count_after = self._get_count()
        inserted = count_after - count_before
        duplicates = len(df) - inserted

        self.conn.execute(
            "INSERT INTO load_metadata (file_name, rows_loaded, rows_duplicates) VALUES (?, ?, ?)",
            (Path(file_path).name, inserted, duplicates)
        )
        self.conn.commit()

        logger.info(f"  Inserted: {inserted:,} | Duplicates skipped: {duplicates:,}")

        return {
            "file": Path(file_path).name,
            "total_rows": total_rows,
            "valid_rows": len(df),
            "inserted": inserted,
            "duplicates": duplicates,
            "invalid": invalid,
        }

    def _build_column_mapping(self, columns) -> dict:
        """Build flexible column name mapping"""
        # Pattern rules: (check_function, target_column_name)
        # Order matters - more specific patterns first
        patterns = [
            (lambda c: "site" in c and "id" in c, "site_id"),
            (lambda c: "b" in c and "unit" in c, "b_unit"),
            (lambda c: "estimated" in c, "is_estimated"),
            (lambda c: "total" in c and "sales" in c, "total_sales"),
            (lambda c: c == "site", "site"),  # Exact match after site_id check
            (lambda c: c == "grade", "grade"),
            (lambda c: c == "day", "day"),
            (lambda c: c == "brand", "brand"),
            (lambda c: c == "address", "address"),
            (lambda c: c == "city", "city"),
            (lambda c: c == "state", "state"),
            (lambda c: c == "owner", "owner"),
            (lambda c: c == "stock", "stock"),
            (lambda c: c == "delivered", "delivered"),
            (lambda c: c == "volume", "volume"),
            (lambda c: c == "target", "target"),
        ]

        mapping = {}
        for col in columns:
            col_lower = col.lower().replace(" ", "_").replace("/", "_")
            for check, target in patterns:
                if check(col_lower):
                    mapping[col] = target
                    break

        return mapping

    def _normalize_bool(self, value) -> bool:
        """Convert common truthy/falsey representations to bool"""
        if value is None or pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y"}

    @staticmethod
    def _normalize_text_column(series: pd.Series, uppercase: bool = False) -> pd.Series:
        """Normalize text identifiers and convert blank-like values to NA."""
        normalized = series.astype("string").str.strip()
        blank_like = normalized.str.lower().isin({"", "nan", "none", "<na>"})
        normalized = normalized.mask(blank_like, pd.NA)
        if uppercase:
            normalized = normalized.str.upper()
        return normalized

    def _get_count(self) -> int:
        """Get total record count"""
        return self.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]

    def load_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Load multiple files (Excel or CSV)"""
        results = []
        for file_path in file_paths:
            loader = self.load_from_excel
            suffix = Path(file_path).suffix.lower()
            if suffix == ".csv":
                loader = self.load_from_csv
            try:
                stats = loader(file_path)
                results.append(stats)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                results.append({
                    "file": Path(file_path).name,
                    "total_rows": 0,
                    "valid_rows": 0,
                    "inserted": 0,
                    "duplicates": 0,
                    "invalid": 0,
                    "error": str(e)
                })
        return pd.DataFrame(results)

    def get_sales_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        site_ids: Optional[List[str]] = None,
        grades: Optional[List[str]] = None,
        exclude_estimated: bool = True,
    ) -> pd.DataFrame:
        """
        Query sales data with filters
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            site_ids: List of site IDs
            grades: List of fuel grades
            exclude_estimated: Filter out estimated values
            
        Returns:
            DataFrame with filtered sales data
        """
        query = "SELECT * FROM sales WHERE 1=1"
        params = []
        
        if exclude_estimated:
            query += " AND COALESCE(is_estimated, 0) = 0"
        
        if start_date:
            query += " AND day >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND day <= ?"
            params.append(end_date)
        
        if site_ids:
            placeholders = ",".join("?" * len(site_ids))
            query += f" AND site_id IN ({placeholders})"
            params.extend(site_ids)
        
        if grades:
            placeholders = ",".join("?" * len(grades))
            query += f" AND grade IN ({placeholders})"
            params.extend(grades)
        
        query += " ORDER BY day, site_id, grade"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        df["day"] = pd.to_datetime(df["day"])
        
        return df

    def get_summary_stats(self) -> dict:
        """Get database summary statistics"""
        stats = {}
        
        stats["total_records"] = self._get_count()
        
        stats["non_estimated_records"] = self.conn.execute(
            "SELECT COUNT(*) FROM sales WHERE COALESCE(is_estimated, 0) = 0"
        ).fetchone()[0]
        
        date_range = pd.read_sql_query(
            "SELECT MIN(day) as min_date, MAX(day) as max_date FROM sales",
            self.conn
        ).iloc[0]
        min_date = date_range["min_date"]
        max_date = date_range["max_date"]
        if pd.isna(min_date) or pd.isna(max_date):
            stats["date_range"] = "N/A"
        else:
            stats["date_range"] = f"{min_date} to {max_date}"
        
        stats["unique_sites"] = self.conn.execute(
            "SELECT COUNT(DISTINCT site_id) FROM sales"
        ).fetchone()[0]
        
        grades = pd.read_sql_query(
            "SELECT DISTINCT grade FROM sales ORDER BY grade",
            self.conn
        )
        stats["fuel_grades"] = [
            grade
            for grade in grades["grade"].tolist()
            if pd.notna(grade) and str(grade).strip() != ""
        ]
        
        return stats

    def get_site_data_quality(self) -> pd.DataFrame:
        """
        Check data quality per site
        
        Returns:
            DataFrame with months of data per site
        """
        query = """
        SELECT 
            site_id,
            site,
            COUNT(DISTINCT strftime('%Y-%m', day)) as months_of_data,
            MIN(day) as first_date,
            MAX(day) as last_date,
            COUNT(*) as total_records
        FROM sales
        WHERE COALESCE(is_estimated, 0) = 0
        GROUP BY site_id, site
        ORDER BY months_of_data DESC
        """
        return pd.read_sql_query(query, self.conn)

    def get_distinct_sites(self) -> pd.DataFrame:
        """
        Get list of distinct sites

        Returns:
            DataFrame with site_id and site columns
        """
        query = """
        SELECT site_id, MAX(site) as site
        FROM sales
        GROUP BY site_id
        ORDER BY site_id
        """
        return pd.read_sql_query(query, self.conn)

    def get_distinct_site_grades(self) -> pd.DataFrame:
        """
        Get list of distinct site-grade combinations

        Returns:
            DataFrame with site_id, site, and grade columns
        """
        query = """
        SELECT site_id, MAX(site) as site, grade
        FROM sales
        GROUP BY site_id, grade
        ORDER BY site_id, grade
        """
        return pd.read_sql_query(query, self.conn)

    # -- calibration methods ---------------------------------------------------

    def save_calibration_run(self, params: Dict) -> int:
        """Insert calibration run metadata, return run_id."""
        sql = """
        INSERT INTO calibration_runs
            (backtest_months, horizon, min_months, sites_calibrated, overall_mape)
        VALUES (?, ?, ?, ?, ?)
        """
        with self.conn:
            cursor = self.conn.execute(sql, (
                params.get("backtest_months"),
                params.get("horizon"),
                params.get("min_months"),
                params.get("sites_calibrated"),
                params.get("overall_mape"),
            ))
        return cursor.lastrowid

    def save_site_weights(self, run_id: int, weights: List[Dict]) -> int:
        """Bulk upsert site-level model weights. Returns rows written."""
        sql = """
        INSERT OR REPLACE INTO calibration_weights
            (site_id, model_name, weight, mape_pct, n_months, run_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        rows = [
            (w["site_id"], w["model_name"], w["weight"],
             w.get("mape_pct"), w.get("n_months"), run_id)
            for w in weights
        ]
        with self.conn:
            self.conn.executemany(sql, rows)
        return len(rows)

    def save_interval_calibration(self, run_id: int, segments: List[Dict]):
        """Store residual distribution stats per segment."""
        sql = """
        INSERT OR REPLACE INTO interval_calibration
            (segment, residual_std, residual_p10, residual_p90, n_observations, run_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        rows = [
            (s["segment"], s["residual_std"], s["residual_p10"],
             s["residual_p90"], s.get("n_observations"), run_id)
            for s in segments
        ]
        with self.conn:
            self.conn.executemany(sql, rows)

    def _get_latest_calibration_run_id(self) -> Optional[int]:
        """Return the latest calibration run_id, or None if no runs exist."""
        row = self.conn.execute(
            "SELECT run_id FROM calibration_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def get_site_weights_bulk(self) -> Dict[str, Dict[str, float]]:
        """Return latest-run site weights as {site_id: {model_name: weight}}."""
        latest_run_id = self._get_latest_calibration_run_id()
        if latest_run_id is None:
            return {}

        rows = self.conn.execute(
            "SELECT site_id, model_name, weight "
            "FROM calibration_weights "
            "WHERE run_id = ?",
            (latest_run_id,),
        ).fetchall()
        result: Dict[str, Dict[str, float]] = {}
        for site_id, model_name, weight in rows:
            result.setdefault(str(site_id), {})[str(model_name)] = float(weight)
        return result

    def get_interval_factors(self, segment: str) -> Optional[Dict]:
        """Lookup residual distribution for a segment."""
        latest_run_id = self._get_latest_calibration_run_id()
        if latest_run_id is None:
            return None

        row = self.conn.execute(
            "SELECT residual_std, residual_p10, residual_p90, n_observations "
            "FROM interval_calibration "
            "WHERE segment = ? AND run_id = ?",
            (segment, latest_run_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "residual_std": row[0],
            "residual_p10": row[1],
            "residual_p90": row[2],
            "n_observations": row[3],
        }

    def get_latest_calibration_run(self) -> Optional[Dict]:
        """Return most recent calibration run metadata."""
        row = self.conn.execute(
            "SELECT run_id, backtest_months, horizon, min_months, "
            "sites_calibrated, overall_mape, created_at "
            "FROM calibration_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "backtest_months": row[1],
            "horizon": row[2],
            "min_months": row[3],
            "sites_calibrated": row[4],
            "overall_mape": row[5],
            "created_at": row[6],
        }

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database closed")
