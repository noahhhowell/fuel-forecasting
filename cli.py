#!/usr/bin/env python3
"""
Fuel Forecasting CLI - Simple command-line interface
"""

import argparse
import sys
import logging
from pathlib import Path
from database import FuelDatabase
from forecaster import FuelForecaster

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_command(args):
    """Load data from Excel files"""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    db = FuelDatabase(args.database, header_row=args.header_row)

    try:
        if args.file:
            # Single file
            suffix = Path(args.file).suffix.lower()
            if suffix == ".csv":
                stats = db.load_from_csv(args.file)
            else:
                stats = db.load_from_excel(args.file)
            print(f"\n✓ Loaded: {stats['file']}")
            print(f"  • Inserted: {stats['inserted']:,} records")
            print(f"  • Duplicates skipped: {stats['duplicates']:,}")

        elif args.directory:
            # All Excel files in directory
            directory = Path(args.directory)
            excel_files = (
                list(directory.glob("*.xlsx"))
                + list(directory.glob("*.xls"))
                + list(directory.glob("*.csv"))
            )

            if not excel_files:
                print(f"\n✗ No Excel/CSV files found in {directory}")
                return

            print(f"\nFound {len(excel_files)} Excel/CSV files\n")
            results = db.load_multiple_files([str(f) for f in excel_files])

            print("\nLoad Summary:")
            print(results[["file", "inserted", "duplicates"]].to_string(index=False))
            print(f"\nTotal inserted: {results['inserted'].sum():,}")
            print(f"Total duplicates: {results['duplicates'].sum():,}")

        print("\n✓ Complete!")

    finally:
        db.close()


def status_command(args):
    """Show database status"""
    print("\n" + "=" * 60)
    print("DATABASE STATUS")
    print("=" * 60)

    db = FuelDatabase(args.database)

    try:
        # Summary stats
        stats = db.get_summary_stats()
        print("\nRecords:")
        print(f"  • Total: {stats['total_records']:,}")
        print(f"  • Non-estimated: {stats['non_estimated_records']:,}")
        print(f"\nDate Range: {stats['date_range']}")
        print(f"Unique Sites: {stats['unique_sites']}")
        print(f"Fuel Grades: {', '.join(stats['fuel_grades'])}")

        # Data quality check
        if args.detailed:
            print("\n" + "-" * 60)
            print("DATA QUALITY BY SITE")
            print("-" * 60)
            quality = db.get_site_data_quality()

            # Show sites with insufficient data
            insufficient = quality[quality["months_of_data"] < 24]
            if not insufficient.empty:
                print(f"\nSites with <24 months: {len(insufficient)}")
                print(
                    insufficient[["site_id", "site", "months_of_data"]]
                    .head(10)
                    .to_string(index=False)
                )

            # Show summary
            print("\nData Quality Summary:")
            print(
                f"  • Sites with 24+ months: {len(quality[quality['months_of_data'] >= 24])}"
            )
            print(
                f"  • Sites with 12-23 months: {len(quality[(quality['months_of_data'] >= 12) & (quality['months_of_data'] < 24)])}"
            )
            print(
                f"  • Sites with <12 months: {len(quality[quality['months_of_data'] < 12])}"
            )

    finally:
        db.close()


def export_command(args):
    """Export database to CSV"""
    print("\n" + "=" * 60)
    print("EXPORTING DATABASE TO CSV")
    print("=" * 60)

    db = FuelDatabase(args.database)

    try:
        # Get filtered data
        print("\nFilters:")
        if args.start_date:
            print(f"  • Start date: {args.start_date}")
        if args.end_date:
            print(f"  • End date: {args.end_date}")
        if args.site_id:
            print(f"  • Site ID: {args.site_id}")
        if args.grade:
            print(f"  • Grade: {args.grade}")
        if args.include_estimated:
            print(f"  • Including estimated values: Yes")
        else:
            print(f"  • Excluding estimated values (default)")

        print("\nQuerying database...")

        # Parse site_ids and grades as lists if provided
        site_ids = [args.site_id] if args.site_id else None
        grades = [args.grade] if args.grade else None

        data = db.get_sales_data(
            start_date=args.start_date,
            end_date=args.end_date,
            site_ids=site_ids,
            grades=grades,
            exclude_estimated=not args.include_estimated,
        )

        if data.empty:
            print("\n✗ No data found matching the specified filters")
            return

        # Export to CSV
        output_file = args.output
        data.to_csv(output_file, index=False)

        # Show summary
        print(f"\n✓ Exported {len(data):,} records to: {output_file}")
        print(f"\nData Summary:")
        print(f"  • Date range: {data['day'].min()} to {data['day'].max()}")
        print(f"  • Unique sites: {data['site_id'].nunique()}")
        print(f"  • Unique grades: {data['grade'].nunique()}")
        print(f"  • File size: ~{len(data) * 50 / 1024 / 1024:.1f} MB (estimated)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

    finally:
        db.close()


def forecast_command(args):
    """Generate forecasts"""
    print("\n" + "=" * 60)
    print(f"GENERATING FORECAST FOR {args.month}")
    print("=" * 60)

    db = FuelDatabase(args.database)
    forecaster = FuelForecaster(db, min_months_data=args.min_months)

    try:
        # Generate forecast
        forecast = forecaster.generate_bulk_forecasts(
            target_month=args.month,
            by=args.by,
            models_to_use=[args.model] if args.model else None,
            output_path=args.output,
            skip_insufficient=not args.include_all,
        )

        # Display summary
        print("\n" + "=" * 60)
        print("FORECAST SUMMARY")
        print("=" * 60)

        if args.by in ["site", "site_grade"]:
            # Show counts
            unique_sites = forecast["site_id"].nunique()
            print(f"\nForecasts generated for {unique_sites} sites")

            # Show top/bottom forecasts
            ensemble = forecast[forecast["model"] == "ENSEMBLE"]
            if not ensemble.empty:
                print("\nTop 5 highest forecasts:")
                top5 = ensemble.nlargest(5, "forecast_volume")[
                    ["site_id", "site_name", "grade", "forecast_volume"]
                ]
                for _, row in top5.iterrows():
                    grade_str = f" - {row['grade']}" if row["grade"] != "ALL" else ""
                    print(
                        f"  {row['site_id']}{grade_str}: {row['forecast_volume']:,.0f} gallons"
                    )

        else:
            # Show all models
            print("\nForecast by Model:")
            for _, row in forecast.iterrows():
                print(f"  {row['model']:20s}: {row['forecast_volume']:>15,.2f} gallons")

            # Highlight ensemble
            ensemble = forecast[forecast["model"] == "ENSEMBLE"]
            if not ensemble.empty:
                print("\n" + "-" * 60)
                print(
                    f"RECOMMENDED (Ensemble): {ensemble['forecast_volume'].iloc[0]:,.2f} gallons"
                )
                print("-" * 60)

        if args.output:
            print(f"\n✓ Exported to: {args.output}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

    finally:
        db.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Fuel Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Load data:
    python cli.py load --file sales_2024.xlsx
    python cli.py load --directory ./data

  Check status:
    python cli.py status
    python cli.py status --detailed

  Export to CSV:
    python cli.py export --output fuel_data.csv
    python cli.py export --output 2024.csv --start-date 2024-01-01
    python cli.py export --output site_123.csv --site-id 123
    python cli.py export --output unl.csv --grade UNL

  Generate forecast:
    python cli.py forecast 2026-01                         # Forecasts by site (default)
    python cli.py forecast 2026-01 --output jan_2026.xlsx  # With Excel export
    python cli.py forecast 2026-01 --by total              # Total forecast only
    python cli.py forecast 2026-01 --by site_grade         # By site and grade
        """,
    )

    parser.add_argument(
        "--database",
        default="fuel_sales.db",
        help="Database file path (default: fuel_sales.db)",
    )

    parser.add_argument(
        "--header-row",
        type=int,
        default=4,
        help="Header row (0-indexed, default: 4 for row 5)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load data from Excel")
    load_group = load_parser.add_mutually_exclusive_group(required=True)
    load_group.add_argument("--file", help="Single Excel file")
    load_group.add_argument("--directory", help="Directory with Excel files")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show database status")
    status_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed quality report"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to CSV")
    export_parser.add_argument(
        "--output",
        default="fuel_sales_export.csv",
        help="Output CSV file (default: fuel_sales_export.csv)",
    )
    export_parser.add_argument(
        "--start-date", help="Start date filter (YYYY-MM-DD)"
    )
    export_parser.add_argument(
        "--end-date", help="End date filter (YYYY-MM-DD)"
    )
    export_parser.add_argument(
        "--site-id", help="Filter by specific site ID"
    )
    export_parser.add_argument(
        "--grade", help="Filter by specific fuel grade"
    )
    export_parser.add_argument(
        "--include-estimated",
        action="store_true",
        help="Include estimated values (default: excluded)",
    )

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Generate forecast")
    forecast_parser.add_argument("month", help="Target month (YYYY-MM)")
    forecast_parser.add_argument(
        "--by",
        choices=["total", "grade", "site", "site_grade"],
        default="site",
        help="Aggregation level (default: site)",
    )
    forecast_parser.add_argument(
        "--model",
        choices=["ets", "snaive"],
        help="Use specific model only",
    )
    forecast_parser.add_argument(
        "--min-months",
        type=int,
        default=24,
        help="Minimum months of data required (default: 24)",
    )
    forecast_parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include sites with insufficient data",
    )
    forecast_parser.add_argument("--output", help="Output Excel file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "load":
            load_command(args)
        elif args.command == "status":
            status_command(args)
        elif args.command == "export":
            export_command(args)
        elif args.command == "forecast":
            forecast_command(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
