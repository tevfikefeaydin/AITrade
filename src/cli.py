"""
CLI Entry Point for ML-Assisted Crypto Trading Research Pipeline.

Commands:
- download: Download 1m klines from Binance
- build: Build features from downloaded data
- train: Train ML models with walk-forward validation
- backtest: Run realistic backtest with costs

Usage:
    python -m src.cli download [--start 2024-01-01] [--end 2025-12-31]
    python -m src.cli build
    python -m src.cli train --out models/ --train_window_days 540 --test_window_days 60
    python -m src.cli backtest --models_dir models/ --fee_bps 10 --slippage_bps 2 --prob_threshold 0.55 --pt 0.008 --sl 0.006 --max_hold 12
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config
from .utils import setup_logging, ensure_directories

logger = logging.getLogger(__name__)


def cmd_download(args):
    """Download 1m klines from Binance for all symbols (with incremental support)."""
    from .data_binance import download_all_symbols

    ensure_directories()

    if args.force:
        logger.info(f"Force re-downloading all data from {args.start} to {args.end}")
    else:
        logger.info(f"Checking/downloading data from {args.start} to {args.end}")
        logger.info("(Use --force to re-download all data)")

    results = download_all_symbols(
        interval="1m",
        start=args.start,
        end=args.end,
        force=args.force,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for symbol, result in results.items():
        stats = result["stats"]
        print(f"\n{symbol}:")
        print(f"  Existing rows:  {stats['existing_rows']:>12,}")
        print(f"  New rows:       {stats['new_rows']:>12,}")
        print(f"  Total rows:     {stats['total_rows']:>12,}")
        if stats["gaps_filled"]:
            print(f"  Gaps filled:    {', '.join(stats['gaps_filled'])}")
        else:
            print(f"  Status:         Data was already up to date")

    print("\n" + "=" * 60)
    logger.info("Download complete!")


def cmd_build(args):
    """Build features from downloaded data."""
    from .data_binance import load_klines
    from .resample import build_multi_timeframe_data
    from .features import build_features
    from .signals import generate_candidates, add_signal_columns
    from .labeling import label_candidates
    from .utils import save_parquet

    ensure_directories()

    for symbol in config.SYMBOLS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Building features for {symbol}")
        logger.info(f"{'='*50}")

        # Load 1m data
        df_1m = load_klines(symbol, "1m")
        if df_1m is None:
            logger.error(f"No data found for {symbol}. Run 'download' first.")
            continue

        # Build multi-timeframe data
        df_1m, df_1h, df_4h, df_intrabar = build_multi_timeframe_data(df_1m)

        # Build features
        df_features = build_features(df_1m, df_1h, df_4h)

        # Add signal columns
        df_features = add_signal_columns(df_features)

        # Generate candidates
        df_candidates = generate_candidates(df_features)

        if len(df_candidates) > 0:
            # Label candidates using 1m data
            df_labeled = label_candidates(df_candidates, df_1m, df_features=df_features)

            # Save labeled candidates
            labeled_path = config.DATA_DIR / f"{symbol}_labeled.parquet"
            save_parquet(df_labeled, labeled_path)
            logger.info(f"Saved {len(df_labeled)} labeled candidates to {labeled_path}")

        # Save features
        features_path = config.get_symbol_features_path(symbol)
        save_parquet(df_features, features_path)
        logger.info(f"Saved {len(df_features)} feature rows to {features_path}")

        # Save 1h data for backtest
        hourly_path = config.DATA_DIR / f"{symbol}_1h.parquet"
        save_parquet(df_1h, hourly_path)

    logger.info("\nBuild complete!")


def cmd_train(args):
    """Train ML models with walk-forward validation."""
    from .train import train_walk_forward

    ensure_directories()

    for symbol in config.SYMBOLS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*50}")

        # Load features
        features_path = config.get_symbol_features_path(symbol)
        if not features_path.exists():
            logger.error(f"Features not found for {symbol}. Run 'build' first.")
            continue

        df_features = pd.read_parquet(features_path)

        # Load labeled candidates
        labeled_path = config.DATA_DIR / f"{symbol}_labeled.parquet"
        if not labeled_path.exists():
            logger.error(f"Labeled data not found for {symbol}. Run 'build' first.")
            continue

        df_labeled = pd.read_parquet(labeled_path)

        # Train with walk-forward
        model, fold_results, aggregate_metrics = train_walk_forward(
            symbol=symbol,
            df_features=df_features,
            df_labeled=df_labeled,
            train_window_days=args.train_window_days,
            test_window_days=args.test_window_days,
            save_model=True,
        )

        if model is not None:
            logger.info(f"Training complete for {symbol}")
            logger.info(f"  Mean AUC: {aggregate_metrics.get('mean_auc', 0):.3f}")
            logger.info(f"  Mean LogLoss: {aggregate_metrics.get('mean_logloss', 0):.3f}")
        else:
            logger.warning(f"Training failed for {symbol}")

    logger.info("\nTraining complete!")


def cmd_backtest(args):
    """Run realistic backtest with costs."""
    from .train import load_model
    from .backtest import run_backtest, run_baseline_all_candidates, save_backtest_results
    from .metrics import compute_baseline_buy_hold, create_summary, create_leaderboard

    ensure_directories()

    summaries = []

    for symbol in config.SYMBOLS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running backtest for {symbol}")
        logger.info(f"{'='*50}")

        # Load model
        model = load_model(symbol)
        if model is None:
            logger.error(f"Model not found for {symbol}. Run 'train' first.")
            continue

        # Load data
        features_path = config.get_symbol_features_path(symbol)
        labeled_path = config.DATA_DIR / f"{symbol}_labeled.parquet"
        hourly_path = config.DATA_DIR / f"{symbol}_1h.parquet"
        minute_path = config.get_symbol_data_path(symbol, "1m")

        if not all(p.exists() for p in [features_path, labeled_path, hourly_path, minute_path]):
            logger.error(f"Required data not found for {symbol}. Run 'build' first.")
            continue

        df_features = pd.read_parquet(features_path)
        df_labeled = pd.read_parquet(labeled_path)
        df_1h = pd.read_parquet(hourly_path)
        df_1m = pd.read_parquet(minute_path)

        # Load OOS predictions if available (for honest backtesting)
        oos_path = config.get_symbol_oos_path(symbol)
        oos_predictions = None
        if oos_path.exists():
            oos_predictions = pd.read_parquet(oos_path)
            logger.info(f"Loaded {len(oos_predictions)} OOS predictions for {symbol}")
        else:
            logger.warning(
                f"No OOS predictions found for {symbol}. "
                f"Backtest will use final model (IN-SAMPLE — results may be inflated)."
            )

        # Run ML-filtered backtest
        trades_df, equity_df, strategy_metrics = run_backtest(
            symbol=symbol,
            df_features=df_features,
            df_labeled=df_labeled,
            model=model,
            df_1m=df_1m,
            df_1h=df_1h,
            prob_threshold=args.prob_threshold,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            pt=args.pt,
            sl=args.sl,
            max_hold=args.max_hold,
            oos_predictions=oos_predictions,
        )

        # Run baseline (all candidates)
        _, _, all_cand_metrics = run_baseline_all_candidates(
            symbol=symbol,
            df_features=df_features,
            df_labeled=df_labeled,
            df_1m=df_1m,
            df_1h=df_1h,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            pt=args.pt,
            sl=args.sl,
            max_hold=args.max_hold,
        )

        # Buy-and-hold baseline
        bh_metrics = compute_baseline_buy_hold(df_1h)

        # Create summary
        summary = create_summary(
            symbol=symbol,
            strategy_metrics=strategy_metrics,
            baseline_metrics=bh_metrics,
            all_candidates_metrics=all_cand_metrics,
        )
        summaries.append(summary)

        # Save results
        save_backtest_results(symbol, trades_df, equity_df, summary)

        # Print summary
        logger.info(f"\nResults for {symbol}:")
        logger.info(f"  Strategy Return: {strategy_metrics.get('total_return', 0):.2%}")
        logger.info(f"  Strategy Sharpe: {strategy_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Buy-Hold Return: {bh_metrics.get('bh_total_return', 0):.2%}")
        logger.info(f"  All Cand Return: {all_cand_metrics.get('total_return', 0):.2%}")

    # Create leaderboard
    if summaries:
        leaderboard = create_leaderboard(summaries)
        leaderboard_path = config.OUTPUTS_DIR / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        logger.info(f"\nLeaderboard saved to {leaderboard_path}")
        print("\n" + leaderboard.to_string())

    logger.info("\nBacktest complete!")


def cmd_paper(args):
    """Run paper trading bot with real-time data."""
    import asyncio
    from .live.paper_trader import PaperTrader, MultiPaperTrader

    ensure_directories()

    kwargs = {
        "prob_threshold": args.prob_threshold,
        "pt": args.pt,
        "sl": args.sl,
        "max_hold_hours": args.max_hold,
        "adx_min_threshold": args.adx_min_threshold,
        "soft_guard": args.soft_guard,
        "guard_threshold_bonus": args.guard_threshold_bonus,
        "guard_cooldown_minutes": args.guard_cooldown_minutes,
    }

    if args.all:
        # Run all symbols from config
        logger.info(f"Starting paper trading for ALL symbols: {config.SYMBOLS}")
        trader = MultiPaperTrader(symbols=config.SYMBOLS, **kwargs)
    else:
        # Single symbol
        logger.info(f"Starting paper trading for {args.symbol}")
        trader = PaperTrader(symbol=args.symbol, **kwargs)

    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\nPaper trader stopped by user.")
        if hasattr(trader, 'print_combined_summary'):
            trader.print_combined_summary()
        else:
            trader.positions.print_summary()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML-Assisted Crypto Trading Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli download                    # Smart download (only missing data)
  python -m src.cli download --force            # Force re-download all data
  python -m src.cli download --start 2023-01-01 # Expand date range
  python -m src.cli build
  python -m src.cli train --train_window_days 540 --test_window_days 60
  python -m src.cli backtest --prob_threshold 0.55 --fee_bps 10

Note: This is an educational project. No trading strategy guarantees profits.

Paper Trading:
  python -m src.cli paper                        # Start paper trading (BTCUSDT)
  python -m src.cli paper --symbol ETHUSDT       # Paper trade ETH
  python -m src.cli paper --all                  # Paper trade ALL symbols (BTC + ETH)
        """,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download 1m klines from Binance",
    )
    download_parser.add_argument(
        "--start",
        type=str,
        default=config.DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {config.DEFAULT_START})",
    )
    download_parser.add_argument(
        "--end",
        type=str,
        default=config.DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {config.DEFAULT_END})",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download all data (ignore existing files)",
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build features from downloaded data",
    )
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train ML models with walk-forward validation",
    )
    train_parser.add_argument(
        "--out",
        type=str,
        default="models/",
        help="Output directory for models",
    )
    train_parser.add_argument(
        "--train_window_days",
        type=int,
        default=config.DEFAULT_TRAIN_WINDOW_DAYS,
        help=f"Training window in days (default: {config.DEFAULT_TRAIN_WINDOW_DAYS})",
    )
    train_parser.add_argument(
        "--test_window_days",
        type=int,
        default=config.DEFAULT_TEST_WINDOW_DAYS,
        help=f"Test window in days (default: {config.DEFAULT_TEST_WINDOW_DAYS})",
    )

    # Backtest command
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run realistic backtest with costs",
    )
    backtest_parser.add_argument(
        "--models_dir",
        type=str,
        default="models/",
        help="Directory containing trained models",
    )
    backtest_parser.add_argument(
        "--fee_bps",
        type=float,
        default=config.DEFAULT_FEE_BPS,
        help=f"Trading fee in basis points (default: {config.DEFAULT_FEE_BPS})",
    )
    backtest_parser.add_argument(
        "--slippage_bps",
        type=float,
        default=config.DEFAULT_SLIPPAGE_BPS,
        help=f"Slippage in basis points (default: {config.DEFAULT_SLIPPAGE_BPS})",
    )
    backtest_parser.add_argument(
        "--prob_threshold",
        type=float,
        default=config.DEFAULT_PROB_THRESHOLD,
        help=f"Minimum probability to take trade (default: {config.DEFAULT_PROB_THRESHOLD})",
    )
    backtest_parser.add_argument(
        "--pt",
        type=float,
        default=config.DEFAULT_PT,
        help=f"Take-profit percentage (default: {config.DEFAULT_PT})",
    )
    backtest_parser.add_argument(
        "--sl",
        type=float,
        default=config.DEFAULT_SL,
        help=f"Stop-loss percentage (default: {config.DEFAULT_SL})",
    )
    backtest_parser.add_argument(
        "--max_hold",
        type=int,
        default=config.DEFAULT_MAX_HOLD,
        help=f"Max holding period in hours (default: {config.DEFAULT_MAX_HOLD})",
    )

    # Paper trading command
    paper_parser = subparsers.add_parser(
        "paper",
        help="Run paper trading bot with real-time data",
    )
    paper_parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    paper_parser.add_argument(
        "--prob_threshold",
        type=float,
        default=config.DEFAULT_PROB_THRESHOLD,
        help=f"Minimum probability to take trade (default: {config.DEFAULT_PROB_THRESHOLD})",
    )
    paper_parser.add_argument(
        "--pt",
        type=float,
        default=config.DEFAULT_PT,
        help=f"Take-profit percentage (default: {config.DEFAULT_PT})",
    )
    paper_parser.add_argument(
        "--sl",
        type=float,
        default=config.DEFAULT_SL,
        help=f"Stop-loss percentage (default: {config.DEFAULT_SL})",
    )
    paper_parser.add_argument(
        "--max_hold",
        type=int,
        default=config.DEFAULT_MAX_HOLD,
        help=f"Max holding period in hours (default: {config.DEFAULT_MAX_HOLD})",
    )
    paper_parser.add_argument(
        "--all",
        action="store_true",
        help="Run paper trading for ALL configured symbols (BTC + ETH)",
    )
    paper_parser.add_argument(
        "--adx_min_threshold",
        type=float,
        default=config.ADX_MIN_THRESHOLD,
        help=f"Minimum ADX for entry filter (default: {config.ADX_MIN_THRESHOLD})",
    )
    paper_parser.add_argument(
        "--soft_guard",
        action=argparse.BooleanOptionalAction,
        default=config.SOFT_GUARD_ENABLED,
        help=f"Enable soft guardrail mode (default: {config.SOFT_GUARD_ENABLED})",
    )
    paper_parser.add_argument(
        "--guard_threshold_bonus",
        type=float,
        default=config.SOFT_GUARD_THRESHOLD_BONUS,
        help=f"Extra threshold in guard mode (default: {config.SOFT_GUARD_THRESHOLD_BONUS})",
    )
    paper_parser.add_argument(
        "--guard_cooldown_minutes",
        type=int,
        default=config.SOFT_GUARD_COOLDOWN_MINUTES,
        help=f"Entry cooldown in guard mode (default: {config.SOFT_GUARD_COOLDOWN_MINUTES})",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    commands = {
        "download": cmd_download,
        "build": cmd_build,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "paper": cmd_paper,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
