# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Pure Python CLI project — ML-Assisted Crypto Trading Research Pipeline. No databases, Docker, or web servers required. All storage is file-based (Parquet, JSON, CSV, pickle).

### Running the project

Standard commands are documented in `CLAUDE.md` and `README.md`. Key commands:

- **Tests:** `pytest tests/ -v` (91 tests, all use synthetic data — no network needed)
- **Pipeline:** `python -m src.cli download` → `build` → `train` → `backtest`
- **Paper trading:** `python -m src.cli paper --all` (requires WebSocket access to `stream.binance.com`)

### Non-obvious caveats

- **Binance API geo-restriction:** The Binance REST API (`api.binance.com`) returns HTTP 451 from US-based cloud environments. The `download` command will fail. To test the full pipeline without real data, generate synthetic 1m parquet files in `data/` matching the schema: columns `open_time` (datetime64 UTC), `open`, `high`, `low`, `close`, `volume` (float64), `close_time` (datetime64 UTC). At least 12 months of 1m data is needed for walk-forward training to produce valid folds.
- **No linter configured:** The project has no flake8, ruff, pylint, or pyproject.toml. Use `python3 -m py_compile` for basic syntax checks.
- **PATH for pip-installed scripts:** When using user installs, add `$HOME/.local/bin` to `PATH` to access `pytest` and other scripts.
- **Walk-forward fold minimum:** Training requires `train_window_days + test_window_days` (default 270+60=330) days of labeled data to generate at least one fold. Data spanning ~18 months produces 4 folds.
- **Stale build warning:** After any change to `signals.py`, `labeling.py`, `features.py`, or `resample.py`, re-run `python -m src.cli build` before `train`/`backtest`.
