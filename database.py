"""
Database Module - SQLite operations with automatic deduplication
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FuelDatabase:
    """Manages fuel sales data in SQLite"""

    SALES_DB_COLUMNS = [
        "site_id",
        "grade",
        "day",
        "brand",
        "site",
        "address",
        "city",
        "state",
        "owner",
        "b_unit",
        "stock",
        "delivered",
        "volume",
        "is_estimated",
        "total_sales",
        "target",
    ]
    SALES_KEY_COLUMNS = ["site_id", "grade", "day"]
    SALES_UPDATE_COLUMNS = [
        "brand",
        "site",
        "address",
        "city",
        "state",
        "owner",
        "b_unit",
        "stock",
        "delivered",
        "volume",
        "is_estimated",
        "total_sales",
        "target",
    ]

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
            rows_updated INTEGER DEFAULT 0,
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
            grade TEXT NOT NULL DEFAULT 'ALL',
            model_name TEXT NOT NULL,
            weight REAL NOT NULL,
            mape_pct REAL,
            n_months INTEGER,
            run_id INTEGER NOT NULL REFERENCES calibration_runs(run_id),
            PRIMARY KEY (run_id, site_id, grade, model_name)
        )
        """

        create_interval_calibration = """
        CREATE TABLE IF NOT EXISTS interval_calibration (
            segment TEXT NOT NULL,
            residual_std REAL NOT NULL,
            residual_p10 REAL NOT NULL,
            residual_p90 REAL NOT NULL,
            n_observations INTEGER,
            run_id INTEGER NOT NULL REFERENCES calibration_runs(run_id),
            PRIMARY KEY (run_id, segment)
        )
        """

        # Triggers to enforce normalized grades on every insert/update,
        # regardless of how data enters the table (Python load_file,
        # manual SQL, etc.).
        create_grade_trigger_insert = """
        CREATE TRIGGER IF NOT EXISTS trg_uppercase_grade_insert
        BEFORE INSERT ON sales
        BEGIN
            SELECT RAISE(ABORT, 'grade must be trimmed uppercase')
            WHERE LENGTH(TRIM(NEW.grade)) = 0
               OR NEW.grade != TRIM(NEW.grade)
               OR NEW.grade != UPPER(NEW.grade);
        END
        """

        create_grade_trigger_update = """
        CREATE TRIGGER IF NOT EXISTS trg_uppercase_grade_update
        BEFORE UPDATE OF grade ON sales
        BEGIN
            SELECT RAISE(ABORT, 'grade must be trimmed uppercase')
            WHERE LENGTH(TRIM(NEW.grade)) = 0
               OR NEW.grade != TRIM(NEW.grade)
               OR NEW.grade != UPPER(NEW.grade);
        END
        """

        with self.conn:
            self.conn.execute(create_sales)
            for idx in indexes:
                self.conn.execute(idx)
            self.conn.execute(create_metadata)
            self._ensure_column("load_metadata", "rows_updated", "INTEGER DEFAULT 0")
            self.conn.execute(create_calibration_runs)
            self._ensure_calibration_weights_schema(create_calibration_weights)
            self._ensure_interval_calibration_schema(create_interval_calibration)
            self.conn.execute(create_grade_trigger_insert)
            self.conn.execute(create_grade_trigger_update)

        logger.info(f"Database initialized: {self.db_path}")

    def load_from_excel(self, file_path: str, replace: bool = False) -> dict:
        """Load data from Excel file with automatic deduplication"""
        logger.info(f"Loading: {file_path}")
        df = pd.read_excel(file_path, skiprows=self.header_row)
        logger.info(f"  Read {len(df):,} rows from Excel")
        return self._load_dataframe(df, file_path, replace=replace)

    def load_from_csv(self, file_path: str, replace: bool = False) -> dict:
        """Load data from CSV file with automatic deduplication"""
        logger.info(f"Loading: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"  Read {len(df):,} rows from CSV")
        return self._load_dataframe(df, file_path, replace=replace)

    def _load_dataframe(
        self,
        df: pd.DataFrame,
        file_path: str,
        replace: bool = False,
    ) -> dict:
        """Normalize columns, validate, and insert rows with deduplication."""
        load_df, total_rows, valid_rows, invalid, input_duplicates = self._prepare_sales_dataframe(df)
        backup_path = None
        metadata_file_name = Path(file_path).name

        if replace:
            backup_path = self.backup_database()
            metadata_file_name = f"REPLACE::{metadata_file_name}"

        with self.conn:
            if replace:
                self._clear_sales_and_calibration()
            inserted, updated, unchanged = self._upsert_sales_rows(
                load_df,
                manage_transaction=False,
            )
            duplicates = unchanged + input_duplicates
            self.conn.execute(
                """
                INSERT INTO load_metadata (file_name, rows_loaded, rows_duplicates, rows_updated)
                VALUES (?, ?, ?, ?)
                """,
                (metadata_file_name, inserted, duplicates, updated),
            )

        logger.info(
            "  Inserted: %s | Updated existing: %s | Duplicates skipped: %s",
            f"{inserted:,}",
            f"{updated:,}",
            f"{duplicates:,}",
        )

        result = {
            "file": Path(file_path).name,
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "inserted": inserted,
            "updated": updated,
            "duplicates": duplicates,
            "invalid": invalid,
        }
        if backup_path is not None:
            result["backup"] = str(backup_path)
        return result

    def _prepare_sales_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, int, int, int, int]:
        """Normalize and validate a raw sales dataframe before writing it."""
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

        valid_rows = len(df)
        df["day"] = df["day"].dt.strftime("%Y-%m-%d")

        if "is_estimated" in df.columns:
            df["is_estimated"] = df["is_estimated"].apply(self._normalize_bool)
        else:
            df["is_estimated"] = False

        for col in self.SALES_DB_COLUMNS:
            if col not in df.columns:
                df[col] = None

        load_df = df[self.SALES_DB_COLUMNS].copy()
        deduped_df = load_df.drop_duplicates(subset=self.SALES_KEY_COLUMNS, keep="last")
        input_duplicates = len(load_df) - len(deduped_df)
        return deduped_df, total_rows, valid_rows, invalid, input_duplicates

    def backup_database(self) -> Path:
        """Create a timestamped backup of the current database file."""
        self.conn.commit()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.with_name(
            f"{self.db_path.stem}_backup_{timestamp}{self.db_path.suffix}"
        )
        counter = 1
        while backup_path.exists():
            backup_path = self.db_path.with_name(
                f"{self.db_path.stem}_backup_{timestamp}_{counter}{self.db_path.suffix}"
            )
            counter += 1

        backup_conn = sqlite3.connect(backup_path)
        try:
            self.conn.backup(backup_conn)
        finally:
            backup_conn.close()
        logger.info("Database backup created: %s", backup_path)
        return backup_path

    def _clear_sales_and_calibration(self) -> None:
        """Delete all sales rows plus derived calibration artifacts."""
        self.conn.execute("DELETE FROM calibration_weights")
        self.conn.execute("DELETE FROM interval_calibration")
        self.conn.execute("DELETE FROM calibration_runs")
        self.conn.execute("DELETE FROM sales")

    def _ensure_column(self, table_name: str, column_name: str, definition: str) -> None:
        """Add a missing column to an existing table."""
        existing_columns = {
            row[1] for row in self.conn.execute(f"PRAGMA table_info({table_name})")
        }
        if column_name not in existing_columns:
            self.conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )

    def _table_exists(self, table_name: str) -> bool:
        """Return True when a SQLite table exists."""
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _table_pk_columns(self, table_name: str) -> List[str]:
        """Return primary-key columns in declared key order."""
        rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        pk_rows = sorted((row for row in rows if row[5]), key=lambda row: row[5])
        return [row[1] for row in pk_rows]

    def _ensure_calibration_weights_schema(self, create_sql: str) -> None:
        """Migrate calibration weights to run- and grade-scoped keys."""
        table = "calibration_weights"
        desired_pk = ["run_id", "site_id", "grade", "model_name"]
        if not self._table_exists(table):
            self.conn.execute(create_sql)
            return

        columns = {
            row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")
        }
        if "grade" in columns and self._table_pk_columns(table) == desired_pk:
            return

        legacy = f"{table}_legacy_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
        self.conn.execute(create_sql)

        legacy_columns = {
            row[1] for row in self.conn.execute(f"PRAGMA table_info({legacy})")
        }
        grade_expr = "grade" if "grade" in legacy_columns else "'ALL'"
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO {table}
                (site_id, grade, model_name, weight, mape_pct, n_months, run_id)
            SELECT
                site_id,
                COALESCE(NULLIF(TRIM({grade_expr}), ''), 'ALL') AS grade,
                model_name,
                weight,
                mape_pct,
                n_months,
                run_id
            FROM {legacy}
            """
        )
        self.conn.execute(f"DROP TABLE {legacy}")

    def _ensure_interval_calibration_schema(self, create_sql: str) -> None:
        """Migrate interval calibration to preserve rows from every run."""
        table = "interval_calibration"
        desired_pk = ["run_id", "segment"]
        if not self._table_exists(table):
            self.conn.execute(create_sql)
            return

        if self._table_pk_columns(table) == desired_pk:
            return

        legacy = f"{table}_legacy_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
        self.conn.execute(create_sql)
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO {table}
                (segment, residual_std, residual_p10, residual_p90, n_observations, run_id)
            SELECT
                segment,
                residual_std,
                residual_p10,
                residual_p90,
                n_observations,
                run_id
            FROM {legacy}
            """
        )
        self.conn.execute(f"DROP TABLE {legacy}")

    def _upsert_sales_rows(
        self,
        df: pd.DataFrame,
        manage_transaction: bool = True,
    ) -> tuple:
        """Insert new rows and refresh existing keys when source data changes."""
        if manage_transaction:
            with self.conn:
                return self._upsert_sales_rows(df, manage_transaction=False)

        stage_table = "temp.staging_sales_load"
        column_list = ", ".join(self.SALES_DB_COLUMNS)
        placeholders = ", ".join("?" for _ in self.SALES_DB_COLUMNS)
        join_on_keys = " AND ".join(
            f"sales.{col} = staging.{col}" for col in self.SALES_KEY_COLUMNS
        )
        changed_predicate = " OR ".join(
            f"sales.{col} IS NOT staging.{col}" for col in self.SALES_UPDATE_COLUMNS
        )
        update_assignments = ", ".join(
            f"{col} = excluded.{col}" for col in self.SALES_UPDATE_COLUMNS
        )
        stage_columns_sql = """
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
            target REAL
        """

        insert_stage_sql = f"""
            INSERT INTO {stage_table} ({column_list})
            VALUES ({placeholders})
        """
        upsert_sql = f"""
            INSERT INTO sales ({column_list})
            VALUES ({placeholders})
            ON CONFLICT(site_id, grade, day) DO UPDATE SET
                {update_assignments}
        """

        self.conn.execute(f"DROP TABLE IF EXISTS {stage_table}")
        try:
            self.conn.execute(f"CREATE TEMP TABLE {stage_table} ({stage_columns_sql})")
            self.conn.executemany(
                insert_stage_sql,
                df[self.SALES_DB_COLUMNS].itertuples(index=False, name=None),
            )

            inserted = self.conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {stage_table} AS staging
                LEFT JOIN sales ON {join_on_keys}
                WHERE sales.site_id IS NULL
                """
            ).fetchone()[0]
            updated = self.conn.execute(
                f"""
                SELECT COUNT(*)
                FROM {stage_table} AS staging
                JOIN sales ON {join_on_keys}
                WHERE {changed_predicate}
                """
            ).fetchone()[0]

            self.conn.executemany(
                upsert_sql,
                df[self.SALES_DB_COLUMNS].itertuples(index=False, name=None),
            )
        finally:
            self.conn.execute(f"DROP TABLE IF EXISTS {stage_table}")

        unchanged = len(df) - inserted - updated
        return inserted, updated, unchanged

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
        """Bulk upsert model weights. Returns rows written."""
        sql = """
        INSERT OR REPLACE INTO calibration_weights
            (site_id, grade, model_name, weight, mape_pct, n_months, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (w["site_id"], w.get("grade") or "ALL", w["model_name"], w["weight"],
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

    def get_site_weights_bulk(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Return latest-run weights as {(site_id, grade): {model_name: weight}}."""
        latest_run_id = self._get_latest_calibration_run_id()
        if latest_run_id is None:
            return {}

        rows = self.conn.execute(
            "SELECT site_id, grade, model_name, weight "
            "FROM calibration_weights "
            "WHERE run_id = ?",
            (latest_run_id,),
        ).fetchall()
        result: Dict[Tuple[str, str], Dict[str, float]] = {}
        for site_id, grade, model_name, weight in rows:
            key = (str(site_id), str(grade or "ALL"))
            result.setdefault(key, {})[str(model_name)] = float(weight)
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

    def get_grade_cohort_stats(self, end_date: Optional[str] = None) -> dict:
        """Per-grade cohort max/min monthly volume from mature sites (>=12 months).

        Returns:
            {grade: {"cohort_monthly_max": float, "cohort_monthly_min": float}}
        """
        date_filter = "AND day <= ?" if end_date else ""
        query = """
        WITH monthly AS (
            SELECT site_id, grade,
                   strftime('%Y-%m', day) AS ym,
                   SUM(volume) AS monthly_vol
            FROM sales
            WHERE COALESCE(is_estimated, 0) = 0
              {date_filter}
            GROUP BY site_id, grade, ym
        ),
        mature_sites AS (
            SELECT site_id, grade
            FROM monthly
            GROUP BY site_id, grade
            HAVING COUNT(DISTINCT ym) >= 12
        ),
        mature_monthly AS (
            SELECT m.grade, m.monthly_vol
            FROM monthly m
            INNER JOIN mature_sites ms
                ON m.site_id = ms.site_id AND m.grade = ms.grade
            WHERE m.monthly_vol > 0
        )
        SELECT grade,
               MAX(monthly_vol) AS cohort_monthly_max,
               MIN(monthly_vol) AS cohort_monthly_min
        FROM mature_monthly
        GROUP BY grade
        """.format(date_filter=date_filter)
        params = (end_date,) if end_date else ()
        rows = self.conn.execute(query, params).fetchall()
        result = {}
        for grade, cmax, cmin in rows:
            result[grade] = {
                "cohort_monthly_max": float(cmax) if cmax else 0.0,
                "cohort_monthly_min": float(cmin) if cmin else 0.0,
            }
        return result

    def get_site_monthly_stats(self, end_date: Optional[str] = None) -> dict:
        """Per (site_id, grade) monthly volume stats.

        Returns:
            {(site_id, grade): {"months_count": int, "site_monthly_max": float, "site_monthly_min": float}}
        """
        date_filter = "AND day <= ?" if end_date else ""
        query = """
        WITH monthly AS (
            SELECT site_id, grade,
                   strftime('%Y-%m', day) AS ym,
                   SUM(volume) AS monthly_vol
            FROM sales
            WHERE COALESCE(is_estimated, 0) = 0
              {date_filter}
            GROUP BY site_id, grade, ym
        )
        SELECT site_id, grade,
               COUNT(DISTINCT ym) AS months_count,
               MAX(monthly_vol) AS site_monthly_max,
               MIN(CASE WHEN monthly_vol > 0 THEN monthly_vol END) AS site_monthly_min
        FROM monthly
        GROUP BY site_id, grade
        """.format(date_filter=date_filter)
        params = (end_date,) if end_date else ()
        rows = self.conn.execute(query, params).fetchall()
        result = {}
        for site_id, grade, mcount, smax, smin in rows:
            result[(str(site_id), str(grade))] = {
                "months_count": int(mcount),
                "site_monthly_max": float(smax) if smax else 0.0,
                "site_monthly_min": float(smin) if smin else 0.0,
            }
        return result

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database closed")
