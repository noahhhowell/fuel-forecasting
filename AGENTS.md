# Repository Guidelines

## Project Structure & Module Organization
- `cli.py` is the entrypoint for all commands (load/status/export/forecast) and wires together database and forecasting logic.
- `database.py` handles SQLite reads/writes and CSV/Excel ingest; `models.py` defines ETS and seasonal naive models; `forecaster.py` runs model selection/ensemble.
- All user-facing documentation is in `README.md`. Sample data belongs in `data/` and outputs in `forecasts/` (both gitignored).
- `tests/` contains the pytest suite: `conftest.py` (shared fixtures), `test_database.py`, `test_models.py`, `test_forecaster.py`.

## Build, Test, and Development Commands
- Install deps: `uv sync` (creates `.venv` and installs from `uv.lock`).
- Help and options: `uv run python cli.py --help` or `uv run python cli.py forecast --help`.
- Load data: `uv run python cli.py load --file data/latest.xlsx` (CSV also supported).
- Health check: `uv run python cli.py status --detailed` to see coverage and gaps.
- Forecast: `uv run python cli.py forecast 2026-01 --by site_grade --output forecasts/jan_2026.xlsx` (swap `--by` level as needed).
- Run tests: `uv run pytest` (53 tests, ~10s). Use `uv run pytest tests/ -v` for verbose or `uv run pytest -k "keyword"` to filter.

## Coding Style & Naming Conventions
- Python 3.9+, 4-space indentation, keep functions and variables snake_case; classes PascalCase; constants UPPER_SNAKE.
- Prefer type hints for public functions (`FuelDatabase`, `FuelForecaster`, model classes) and keep docstrings concise about inputs/outputs.
- Keep column naming consistent (`site_id`, `grade`, `day`, `volume`, `target_month`); preserve ordered columns shown in README.
- Use logging (module-level logger) instead of print for diagnostics; avoid broad exceptions; raise `ValueError`/`RuntimeError` with context.

## Testing Guidelines
- Test suite lives in `tests/` with one file per module: `test_database.py`, `test_models.py`, `test_forecaster.py`.
- Shared fixtures in `tests/conftest.py` provide a temporary SQLite database pre-loaded with 36 months of synthetic data. Tests never touch `fuel_sales.db`.
- When adding features, add corresponding tests. Cover: expected output shape, edge cases (empty data, zeros, missing months), and error paths (`pytest.raises`).
- For manual smoke testing: load a small CSV into `fuel_sales.db`, run `status --detailed`, then `forecast <month>` and inspect the Excel/CSV outputs (`Skipped` and `Summary` tabs for anomalies).

## Commit & Pull Request Guidelines
- Follow existing history: short, imperative summaries ("Add backtesting changes", "Update CSV export"); keep scope-focused and avoid long prefixes.
- In PRs, include: what changed, sample commands used, expected outputs/paths, and any data or flags required (`--include-all`, `--min-months`).
- Link related issue/ticket if available; attach screenshots of Excel outputs when UI/format changes matter; update README.md when user-facing behavior or file formats shift.

## Security & Configuration Tips
- Keep large inputs/outputs in `data/` and `forecasts/` out of version control (already in `.gitignore`); avoid committing `.venv` or SQLite DBs.
- Treat spreadsheets as sensitive operational data; scrub before sharing and prefer sample/mock files in docs and tests.
- When regenerating environments, prefer `uv sync --frozen` to stay pinned to `uv.lock` unless intentionally upgrading.
