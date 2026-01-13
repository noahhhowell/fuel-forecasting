# Repository Guidelines

## Project Structure & Module Organization
- `cli.py` is the entrypoint for all commands (load/status/forecast) and wires together database and forecasting logic.
- `database.py` handles SQLite reads/writes and CSV/Excel ingest; `models.py` defines ETS and seasonal naive models; `forecaster.py` runs model selection/ensemble.
- Docs live in `README.md`, `EXAMPLES.md`, `QUICKREF.md`, and `BACKTESTING_GUIDE.md` for workflows; sample data belongs in `data/` and outputs in `forecasts/` (ignored from git).
- No dedicated test suite directory yet; add `tests/` with `test_*.py` when introducing automated checks.

## Build, Test, and Development Commands
- Install deps: `uv sync` (creates `.venv` and installs from `uv.lock`).
- Help and options: `uv run python cli.py --help` or `uv run python cli.py forecast --help`.
- Load data: `uv run python cli.py load --file data/latest.xlsx` (CSV also supported).
- Health check: `uv run python cli.py status --detailed` to see coverage and gaps.
- Forecast: `uv run python cli.py forecast 2026-01 --by site_grade --output forecasts/jan_2026.xlsx` (swap `--by` level as needed).
- If you add pytest tests, run `uv run pytest` from repo root.

## Coding Style & Naming Conventions
- Python 3.9+, 4-space indentation, keep functions and variables snake_case; classes PascalCase; constants UPPER_SNAKE.
- Prefer type hints for public functions (`FuelDatabase`, `FuelForecaster`, model classes) and keep docstrings concise about inputs/outputs.
- Keep column naming consistent (`site_id`, `grade`, `day`, `volume`, `target_month`); preserve ordered columns shown in README/QUICKREF.
- Use logging (module-level logger) instead of print for diagnostics; avoid broad exceptions; raise `ValueError`/`RuntimeError` with context.

## Testing Guidelines
- Current repo lacks automated tests; add `tests/` with `test_*.py` mirroring modules when extending logic.
- Start with smoke runs: load a small CSV into `fuel_sales.db`, run `status --detailed`, then `forecast <month>` and inspect the Excel/CSV outputs (`Skipped` and `Summary` tabs for anomalies).
- For new features, include fixtures for small sample datasets and assert row counts/column order/ensemble selection.

## Commit & Pull Request Guidelines
- Follow existing history: short, imperative summaries ("Add backtesting changes", "Update CSV export"); keep scope-focused and avoid long prefixes.
- In PRs, include: what changed, sample commands used, expected outputs/paths, and any data or flags required (`--include-all`, `--min-months`).
- Link related issue/ticket if available; attach screenshots of Excel outputs when UI/format changes matter; update docs (README/QUICKREF/BACKTESTING_GUIDE) when user-facing behavior or file formats shift.

## Security & Configuration Tips
- Keep large inputs/outputs in `data/` and `forecasts/` out of version control (already in `.gitignore`); avoid committing `.venv` or SQLite DBs.
- Treat spreadsheets as sensitive operational data; scrub before sharing and prefer sample/mock files in docs and tests.
- When regenerating environments, prefer `uv sync --frozen` to stay pinned to `uv.lock` unless intentionally upgrading.
