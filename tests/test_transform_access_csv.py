import pandas as pd
import pytest

from scripts.transform_access_csv import (
    OUTPUT_COLUMNS,
    PRODUCT_TO_GRADE,
    transform_access_csv,
    transform_access_dataframe,
)


class TestTransformAccessCsv:
    """Tests for the Access CSV normalization script."""

    def test_transform_access_dataframe_maps_known_products(self):
        rows = [
            {
                "site_id": "100",
                "product": product,
                "volume": float(index + 1),
                "date": f"2024-01-{index + 1:02d}",
            }
            for index, product in enumerate(PRODUCT_TO_GRADE)
        ]

        transformed, _ = transform_access_dataframe(pd.DataFrame(rows))

        expected = {
            (row["site_id"], PRODUCT_TO_GRADE[row["product"]], row["date"])
            for row in rows
        }
        actual = set(
            transformed[["site_id", "grade", "day"]].itertuples(index=False, name=None)
        )
        assert actual == expected

    def test_transform_access_dataframe_aggregates_grade_collisions(self):
        df = pd.DataFrame(
            [
                {
                    "site_id": "100",
                    "product": "Blend Value 89 Octane",
                    "volume": 100.0,
                    "date": "2024-01-01",
                },
                {
                    "site_id": "100",
                    "product": "UNL Plus",
                    "volume": 25.0,
                    "date": "2024-01-01",
                },
            ]
        )

        transformed, _ = transform_access_dataframe(df)

        assert len(transformed) == 1
        row = transformed.iloc[0]
        assert row["site_id"] == "100"
        assert row["grade"] == "UNL MID"
        assert row["day"] == "2024-01-01"
        assert row["volume"] == 125.0
        assert bool(row["is_estimated"]) is False

    def test_transform_access_dataframe_rejects_unmapped_products(self):
        df = pd.DataFrame(
            [
                {
                    "site_id": "100",
                    "product": "Mystery Fuel",
                    "volume": 100.0,
                    "date": "2024-01-01",
                }
            ]
        )

        with pytest.raises(ValueError, match="Unmapped products"):
            transform_access_dataframe(df)

    def test_transform_access_dataframe_normalizes_product_case_and_whitespace(self):
        df = pd.DataFrame(
            [
                {
                    "site_id": "100",
                    "product": " unl regular ",
                    "volume": 10.0,
                    "date": "2024-01-01",
                }
            ]
        )

        transformed, _ = transform_access_dataframe(df)

        assert len(transformed) == 1
        assert transformed.iloc[0]["grade"] == "UNL"

    def test_transform_access_csv_writes_loader_compatible_columns(self, tmp_path):
        input_path = tmp_path / "access.csv"
        output_path = tmp_path / "normalized.csv"
        input_path.write_text(
            "site_id,product,volume,date,source_file,source_sheet\n"
            "100,UNL Regular,10,2024-01-01,access.xlsx,Sheet1\n"
            "100,Blend Value 89 Octane,5,2024-01-01,access.xlsx,Sheet1\n"
        )

        stats = transform_access_csv(str(input_path), str(output_path), chunk_size=1)

        written = pd.read_csv(output_path)
        assert list(written.columns) == OUTPUT_COLUMNS
        assert stats["total_rows"] == 2
        assert stats["written_rows"] == 2
