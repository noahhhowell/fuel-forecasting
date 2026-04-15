#!/usr/bin/env python3
"""
Transform Access-exported fuel history into the normalized CSV shape expected by
the standard database loader.
"""

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PRODUCT_TO_GRADE: Dict[str, str] = {
    "100 Octane Racing Fuel": "RACING FUEL",
    "Blend Value 89 Octane": "UNL MID",
    "Blend Value 91 Octane": "UNL PREM",
    "Blend Value 92 Octane": "UNL PREM",
    "Diesel": "DSL",
    "Diesel Dyed": "DYED DSL",
    "Diesel Exhaust Fluid": "DEF",
    "Dyed Kerosene": "DYED KERO",
    "Kerosene": "KERO",
    "PRE Unleaded": "UNL PREM",
    "UNL Non Ethanol": "UNL NO-ETH",
    "UNL Plus": "UNL MID",
    "UNL Plus Non Ethanol": "UNL NO-ETH PREM",
    "UNL Regular": "UNL",
}

ACCESS_REQUIRED_COLUMNS = ["site_id", "product", "volume", "date"]
OUTPUT_COLUMNS = ["site_id", "grade", "day", "volume", "is_estimated"]
AGGREGATE_COLUMNS = ["site_id", "grade", "day", "volume"]
SQLITE_BATCH_SIZE = 200_000


def _normalize_column_name(name: str) -> str:
    return str(name).lstrip("\ufeff").strip()


def _normalize_lookup_key(value: str) -> str:
    return str(value).strip().casefold()


PRODUCT_TO_GRADE_LOOKUP = {
    _normalize_lookup_key(product): grade
    for product, grade in PRODUCT_TO_GRADE.items()
}


def _normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    blank_like = normalized.str.lower().isin({"", "nan", "none", "<na>"})
    return normalized.mask(blank_like, pd.NA)


def transform_access_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Normalize one Access-format dataframe chunk into loader-compatible rows."""
    total_rows = len(df)
    df = df.copy()
    df.columns = [_normalize_column_name(column) for column in df.columns]

    missing = [col for col in ACCESS_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["site_id"] = _normalize_text(df["site_id"])
    product = _normalize_text(df["product"])
    product_lookup = product.dropna().map(_normalize_lookup_key)
    unknown_products = sorted(
        set(product.loc[product_lookup[~product_lookup.isin(PRODUCT_TO_GRADE_LOOKUP)].index])
    )
    if unknown_products:
        raise ValueError(f"Unmapped products: {unknown_products}")

    df["grade"] = product.map(
        lambda value: PRODUCT_TO_GRADE_LOOKUP.get(_normalize_lookup_key(value))
        if pd.notna(value)
        else pd.NA
    )
    df["day"] = pd.to_datetime(df["date"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    required_valid = (
        df["site_id"].notna()
        & df["grade"].notna()
        & df["day"].notna()
        & df["volume"].notna()
    )
    invalid_rows = int((~required_valid).sum())
    df = df.loc[required_valid, ["site_id", "grade", "day", "volume"]].copy()
    if df.empty:
        raise ValueError("No valid rows after cleaning required fields")

    df["day"] = df["day"].dt.strftime("%Y-%m-%d")
    grouped = (
        df.groupby(["site_id", "grade", "day"], as_index=False, sort=True)["volume"]
        .sum()
    )
    grouped["is_estimated"] = False
    return grouped[OUTPUT_COLUMNS], {
        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
        "rows_after_grouping": len(grouped),
    }


def transform_access_csv(
    input_path: str,
    output_path: str,
    chunk_size: int = 250_000,
) -> Dict[str, int]:
    """Transform an Access-format CSV into a normalized loader-compatible CSV."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f"{output_path_obj.stem}_aggregate_",
        suffix=".sqlite",
        dir=output_path_obj.parent,
        delete=False,
    )
    temp_db_path = Path(temp_handle.name)
    temp_handle.close()

    total_rows = 0
    invalid_rows = 0

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        usecols=lambda column: _normalize_column_name(column) in ACCESS_REQUIRED_COLUMNS,
        encoding="utf-8-sig",
        dtype={"site_id": "string", "product": "string", "date": "string"},
    )
    try:
        conn = sqlite3.connect(temp_db_path)
        try:
            conn.execute(
                """
                CREATE TABLE aggregated_sales (
                    site_id TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    day TEXT NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (site_id, grade, day)
                )
                """
            )

            for chunk in reader:
                normalized_chunk, stats = transform_access_dataframe(chunk)
                rows = normalized_chunk[AGGREGATE_COLUMNS].itertuples(index=False, name=None)
                conn.executemany(
                    """
                    INSERT INTO aggregated_sales (site_id, grade, day, volume)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(site_id, grade, day) DO UPDATE SET
                        volume = aggregated_sales.volume + excluded.volume
                    """,
                    rows,
                )
                total_rows += stats["total_rows"]
                invalid_rows += stats["invalid_rows"]

            written_rows = conn.execute(
                "SELECT COUNT(*) FROM aggregated_sales"
            ).fetchone()[0]
            if written_rows == 0:
                raise ValueError("No rows were read from the Access CSV")

            wrote_any = False
            for chunk in pd.read_sql_query(
                """
                SELECT site_id, grade, day, volume
                FROM aggregated_sales
                ORDER BY day, site_id, grade
                """,
                conn,
                chunksize=SQLITE_BATCH_SIZE,
            ):
                chunk["is_estimated"] = False
                chunk = chunk[OUTPUT_COLUMNS]
                chunk.to_csv(
                    output_path_obj,
                    mode="a" if wrote_any else "w",
                    index=False,
                    header=not wrote_any,
                )
                wrote_any = True
        finally:
            conn.close()
    finally:
        if temp_db_path.exists():
            os.remove(temp_db_path)

    return {
        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
        "written_rows": written_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform Access fuel history CSV into loader-compatible CSV",
    )
    parser.add_argument("--input", required=True, help="Source Access-format CSV file")
    parser.add_argument("--output", required=True, help="Normalized output CSV file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows to process per chunk (default: 250000)",
    )
    args = parser.parse_args()

    stats = transform_access_csv(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )

    print(f"Input: {Path(args.input).name}")
    print(f"Output: {Path(args.output).name}")
    print(f"Rows read: {stats['total_rows']:,}")
    print(f"Invalid rows skipped: {stats['invalid_rows']:,}")
    print(f"Rows written: {stats['written_rows']:,}")


if __name__ == "__main__":
    main()
