# Repository Guidelines

## Project Structure & Module Organization
- `cli.py` is the entrypoint for all commands (load/status/export/forecast/calibrate) and wires together database and forecasting logic.
- `database.py` handles SQLite reads/writes and CSV/Excel ingest; `models.py` defines ETS and seasonal naive models; `forecaster.py` runs model selection/ensemble.
- `calibrate.py` learns per-site optimal ensemble weights and prediction intervals via backtesting. It stores results in the database scoped to a calibration run ID. Only the latest run is used for forecasting.
- `backtest.py` evaluates forecast accuracy by holding out recent months and comparing predictions to actuals.
- All user-facing documentation is in `README.md`. Sample data belongs in `data/` and outputs in `forecasts/` (both gitignored).
- `tests/` contains the pytest suite: `conftest.py` (shared fixtures), `test_database.py`, `test_models.py`, `test_forecaster.py`, `test_backtest.py`, `test_calibrate.py`.

## Build, Test, and Development Commands
- Install deps: `uv sync` (creates `.venv` and installs from `uv.lock`).
- Help and options: `uv run python cli.py --help` or `uv run python cli.py forecast --help`.
- Load data: `uv run python cli.py load --file data/latest.xlsx` (CSV also supported).
- Health check: `uv run python cli.py status --detailed` to see coverage and gaps.
- Forecast: `uv run python cli.py forecast 2026-01 --output forecasts/jan_2026.xlsx`.
- Calibrate: `uv run python cli.py calibrate` to learn per-site weights. Add `--output cal.xlsx` for a report.
- Backtest: `uv run python backtest.py --months 12` to evaluate accuracy over the last 12 months.
- Run tests: `uv run pytest` (full suite, ~20s). Use `uv run pytest tests/ -v` for verbose or `uv run pytest -k "keyword"` to filter.

## Key Design Decisions
- **Adaptive weights**: By default, `forecast` uses calibrated per-site weights if a calibration run exists in the database. Pass `--no-calibration` to force fixed 70/30 weights. Calibration weights are read via `get_site_weights_bulk()` which only returns rows from the latest calibration run ID, preventing stale data from older runs.
- **Input validation**: Both `calibrate.py` and `backtest.py` validate parameters (months, min_months, horizon, weight_floor) at the top of the call stack. Invalid values raise `ValueError` with a clear message before any database or model work begins.
- **Shrinkage**: Sites with fewer than 6 backtest observations have their weights blended toward the global average to avoid overfitting on small samples.

## Coding Style & Naming Conventions
- Python 3.9+, 4-space indentation, keep functions and variables snake_case; classes PascalCase; constants UPPER_SNAKE.
- Prefer type hints for public functions (`FuelDatabase`, `FuelForecaster`, model classes) and keep docstrings concise about inputs/outputs.
- Keep column naming consistent (`site_id`, `grade`, `day`, `volume`, `target_month`); preserve ordered columns shown in README.
- Use logging (module-level logger) instead of print for diagnostics; avoid broad exceptions; raise `ValueError`/`RuntimeError` with context.

## Testing Guidelines
- Test suite lives in `tests/` with one file per module: `test_database.py`, `test_models.py`, `test_forecaster.py`, `test_backtest.py`, `test_calibrate.py`.
- Shared fixtures in `tests/conftest.py` provide a temporary SQLite database pre-loaded with 36 months of synthetic data. Tests never touch `fuel_sales.db`.
- When adding features, add corresponding tests. Cover: expected output shape, edge cases (empty data, zeros, missing months), and error paths (`pytest.raises`).
- Validation tests exist for all parameter boundaries in both calibration and backtest modules.
- For manual smoke testing: load a small CSV into `fuel_sales.db`, run `status --detailed`, then `forecast <month>` and inspect the Excel/CSV outputs (`Skipped` and `Summary` tabs for anomalies).

## Commit & Pull Request Guidelines
- Follow existing history: short, imperative summaries ("Add backtesting changes", "Update CSV export"); keep scope-focused and avoid long prefixes.
- In PRs, include: what changed, sample commands used, expected outputs/paths, and any data or flags required (`--include-all`, `--min-months`).
- Link related issue/ticket if available; attach screenshots of Excel outputs when UI/format changes matter; update README.md when user-facing behavior or file formats shift.

## Security & Configuration Tips
- Keep large inputs/outputs in `data/` and `forecasts/` out of version control (already in `.gitignore`); avoid committing `.venv` or SQLite DBs.
- Treat spreadsheets as sensitive operational data; scrub before sharing and prefer sample/mock files in docs and tests.
- When regenerating environments, prefer `uv sync --frozen` to stay pinned to `uv.lock` unless intentionally upgrading.
