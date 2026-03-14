"""
Walk-Forward Training for ML-Assisted Crypto Trading Research Pipeline.

Implements walk-forward (rolling) cross-validation for training per-symbol models:
- No shuffle: respects temporal order
- Rolling window: train on N days, test on M days, step forward
- Per-fold metrics: AUC, logloss, and backtest metrics
- Final model trained on all data

Model: LightGBM classifier (with sklearn fallback)
"""

import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

from . import config
from .features import get_feature_columns

logger = logging.getLogger(__name__)


def get_model():
    """
    Get the ML model to use for training.

    Tries LightGBM first, falls back to sklearn HistGradientBoostingClassifier.

    Returns:
        Model instance
    """
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        logger.info("Using LightGBM classifier")
        return model
    except ImportError:
        logger.warning("LightGBM not available, falling back to sklearn")
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        return model


def walk_forward_split(
    df: pd.DataFrame,
    train_window_days: int,
    test_window_days: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.

    Args:
        df: DataFrame with open_time column (sorted)
        train_window_days: Number of days for training window
        test_window_days: Number of days for test window

    Returns:
        List of (train_df, test_df) tuples
    """
    if "open_time" not in df.columns:
        raise ValueError("DataFrame must have 'open_time' column")

    df = df.sort_values("open_time").reset_index(drop=True)

    train_delta = pd.Timedelta(days=train_window_days)
    test_delta = pd.Timedelta(days=test_window_days)

    start_time = df["open_time"].min()
    end_time = df["open_time"].max()

    splits = []
    current_train_start = start_time

    while True:
        train_end = current_train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta

        if test_end > end_time:
            break

        # Get train and test data
        train_mask = (df["open_time"] >= current_train_start) & (df["open_time"] < train_end)
        test_mask = (df["open_time"] >= test_start) & (df["open_time"] < test_end)

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            splits.append((train_df, test_df))

        # Move forward by test_window_days
        current_train_start = current_train_start + test_delta

    logger.info(f"Generated {len(splits)} walk-forward folds")
    return splits


def train_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "label",
) -> Tuple[object, Dict[str, float]]:
    """
    Train model on a single fold and compute metrics.

    Args:
        train_df: Training data with features and label
        test_df: Test data with features and label
        feature_cols: List of feature column names
        target_col: Name of target column

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Prepare features
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Handle missing features
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Binarize fractional labels (from timeout labeling) for classifier
    # Timeouts with exit return in upper half of barrier range → 1 (success)
    # Timeouts with exit return in lower half → 0 (failure)
    # This captures the key insight: positive-return timeouts are not failures
    y_train = (y_train >= 0.5).astype(int)
    y_test_binary = (y_test >= 0.5).astype(int)

    # Train model
    model = get_model()

    # Early stopping for LightGBM (skip if test set is single-class)
    has_two_classes = len(np.unique(y_test_binary)) > 1
    try:
        import lightgbm as lgb
        is_lgbm = isinstance(model, lgb.LGBMClassifier)
    except ImportError:
        is_lgbm = False

    if is_lgbm and has_two_classes:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test_binary)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)

    best_iter = getattr(model, "best_iteration_", model.n_estimators if hasattr(model, "n_estimators") else 500)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics (using binary labels for AUC/logloss)
    metrics = {}

    if has_two_classes:
        metrics["auc"] = roc_auc_score(y_test_binary, y_pred_proba)
    else:
        metrics["auc"] = 0.5

    if has_two_classes:
        metrics["logloss"] = log_loss(y_test_binary, y_pred_proba)
    else:
        metrics["logloss"] = 0.0

    # Accuracy at threshold 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics["accuracy"] = (y_pred == y_test_binary).mean()

    # Positive rate in predictions
    metrics["pred_positive_rate"] = y_pred.mean()
    metrics["actual_positive_rate"] = y_test_binary.mean()

    metrics["best_iteration"] = best_iter

    return model, metrics


def _run_folds(splits, feature_cols, symbol, pass_label=""):
    """Run walk-forward folds and collect results, OOS predictions, importances."""
    fold_results = []
    oos_records = []
    fold_importances = []
    best_iterations = []

    desc = f"Training {symbol}" + (f" ({pass_label})" if pass_label else "")
    for i, (train_df, test_df) in enumerate(tqdm(splits, desc=desc)):
        model, metrics = train_fold(train_df, test_df, feature_cols)

        # Collect OOS predictions from this fold's test set
        X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)
        oos_proba = model.predict_proba(X_test)[:, 1]
        for j, (_, row) in enumerate(test_df.iterrows()):
            record = {
                "open_time": row["open_time"],
                "oos_probability": oos_proba[j],
                "fold": i,
            }
            # Include signal_type_encoded for composite key (multi-signal support)
            if "signal_type_encoded" in row.index:
                record["signal_type_encoded"] = row["signal_type_encoded"]
            oos_records.append(record)

        # Collect feature importance
        if hasattr(model, "feature_importances_"):
            fold_importances.append(model.feature_importances_)

        best_iterations.append(metrics["best_iteration"])

        fold_result = {
            "fold": i,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_start": train_df["open_time"].min(),
            "train_end": train_df["open_time"].max(),
            "test_start": test_df["open_time"].min(),
            "test_end": test_df["open_time"].max(),
            **metrics,
        }
        fold_results.append(fold_result)

        logger.debug(
            f"Fold {i}: AUC={metrics['auc']:.3f}, "
            f"LogLoss={metrics['logloss']:.3f}, "
            f"best_iter={metrics['best_iteration']}"
        )

    return fold_results, oos_records, fold_importances, best_iterations


def train_walk_forward(
    symbol: str,
    df_features: pd.DataFrame,
    df_labeled: pd.DataFrame,
    train_window_days: int = None,
    test_window_days: int = None,
    save_model: bool = True,
) -> Tuple[object, List[Dict], Dict[str, float]]:
    """
    Train model using walk-forward validation.

    Args:
        symbol: Symbol being trained
        df_features: DataFrame with features
        df_labeled: DataFrame with labels
        train_window_days: Training window size
        test_window_days: Test window size
        save_model: Whether to save the final model

    Returns:
        Tuple of (final_model, fold_results, aggregate_metrics)
    """
    train_window_days = train_window_days or config.DEFAULT_TRAIN_WINDOW_DAYS
    test_window_days = test_window_days or config.DEFAULT_TEST_WINDOW_DAYS

    logger.info(f"Starting walk-forward training for {symbol}")
    logger.info(f"Train window: {train_window_days} days, Test window: {test_window_days} days")

    # Merge features with labels (include signal_type_encoded for multi-signal support)
    label_cols = ["entry_time", "label"]
    if "signal_type_encoded" in df_labeled.columns:
        label_cols.append("signal_type_encoded")

    df_merged = df_features.merge(
        df_labeled[label_cols],
        left_on="open_time",
        right_on="entry_time",
        how="inner",
    )

    # Ensure signal_type_encoded exists (default to 0=breakout for backward compat)
    if "signal_type_encoded" not in df_merged.columns:
        df_merged["signal_type_encoded"] = 0

    if len(df_merged) == 0:
        logger.warning(f"No labeled samples for {symbol}")
        return None, [], {}

    logger.info(f"Merged data: {len(df_merged)} samples, {df_merged['label'].mean():.1%} positive")

    # Get feature columns
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df_merged.columns]
    logger.info(f"Using {len(available_cols)} features")

    # Generate walk-forward splits
    splits = walk_forward_split(df_merged, train_window_days, test_window_days)

    if len(splits) == 0:
        logger.warning("No valid walk-forward splits generated")
        return None, [], {}

    # ── PASS 1: Train all folds with full feature set ──────────────────
    fold_results, oos_records, fold_importances, best_iterations = _run_folds(
        splits, available_cols, symbol, pass_label="Pass 1"
    )

    # ── Feature pruning ─────────────────────────────────────────────────
    pruned_cols = available_cols
    if fold_importances:
        avg_importance = np.mean(fold_importances, axis=0)
        total = avg_importance.sum()
        if total > 0:
            keep_mask = avg_importance >= (total * config.FEATURE_MIN_IMPORTANCE_PCT / 100)
            pruned_cols = [col for col, keep in zip(available_cols, keep_mask) if keep]
            dropped = [col for col, keep in zip(available_cols, keep_mask) if not keep]
            if dropped:
                logger.info(
                    f"Pruned {len(dropped)} low-importance features: {dropped}"
                )

    # ── PASS 2: Retrain with pruned features if any were dropped ─────
    if len(pruned_cols) < len(available_cols):
        logger.info(f"Retraining with {len(pruned_cols)} features (was {len(available_cols)})...")
        fold_results, oos_records, fold_importances, best_iterations = _run_folds(
            splits, pruned_cols, symbol, pass_label="Pass 2"
        )

    # ── Aggregate metrics across folds ───────────────────────────────
    aggregate_metrics = {
        "mean_auc": np.mean([r["auc"] for r in fold_results]),
        "std_auc": np.std([r["auc"] for r in fold_results]),
        "mean_logloss": np.mean([r["logloss"] for r in fold_results]),
        "std_logloss": np.std([r["logloss"] for r in fold_results]),
        "mean_accuracy": np.mean([r["accuracy"] for r in fold_results]),
        "n_folds": len(fold_results),
        "n_features": len(pruned_cols),
    }

    if best_iterations:
        aggregate_metrics["mean_best_iteration"] = int(np.mean(best_iterations))

    logger.info(
        f"Walk-forward results: AUC={aggregate_metrics['mean_auc']:.3f} "
        f"(+/- {aggregate_metrics['std_auc']:.3f}), "
        f"features={len(pruned_cols)}, "
        f"avg_iter={aggregate_metrics.get('mean_best_iteration', 'N/A')}"
    )

    # ── Train final model on all data ────────────────────────────────
    logger.info("Training final model on all data...")
    X_all = df_merged[pruned_cols].values
    y_all = (df_merged["label"].values >= 0.5).astype(int)  # Binarize fractional labels
    X_all = np.nan_to_num(X_all, nan=0.0)

    final_model = get_model()
    # Use average best iteration from folds (minimum 10)
    if best_iterations:
        avg_best_iter = max(10, int(np.mean(best_iterations)))
        final_model.n_estimators = avg_best_iter
        logger.info(f"Final model n_estimators set to {avg_best_iter} (fold average)")
    final_model.fit(X_all, y_all)

    # Save feature columns with model for prediction
    final_model.feature_cols_ = pruned_cols

    if save_model:
        model_path = config.get_symbol_model_path(symbol)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_path)
        logger.info(f"Saved model to {model_path}")

        # Save OOS predictions for honest backtesting
        if oos_records:
            df_oos = pd.DataFrame(oos_records)
            # Ensure consistent types for composite key merge in backtest
            if "signal_type_encoded" in df_oos.columns:
                df_oos["signal_type_encoded"] = df_oos["signal_type_encoded"].astype(int)
            # Deduplicate: keep last fold's prediction for each (open_time, signal_type)
            dedup_cols = ["open_time"]
            if "signal_type_encoded" in df_oos.columns:
                dedup_cols.append("signal_type_encoded")
            n_before = len(df_oos)
            df_oos = df_oos.sort_values("fold").drop_duplicates(
                subset=dedup_cols, keep="last"
            )
            if len(df_oos) < n_before:
                logger.warning(
                    "Deduplicated OOS predictions: %d -> %d rows",
                    n_before, len(df_oos),
                )
            df_oos = df_oos.drop(columns=["fold"])
            oos_path = config.get_symbol_oos_path(symbol)
            oos_path.parent.mkdir(parents=True, exist_ok=True)
            df_oos.to_parquet(oos_path, index=False)
            logger.info(f"Saved {len(df_oos)} OOS predictions to {oos_path}")

    return final_model, fold_results, aggregate_metrics


def load_model(symbol: str) -> Optional[object]:
    """
    Load a trained model for a symbol.

    Args:
        symbol: Symbol to load model for

    Returns:
        Loaded model or None if not found
    """
    model_path = config.get_symbol_model_path(symbol)

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def predict_proba(
    model: object,
    df_features: pd.DataFrame,
) -> np.ndarray:
    """
    Get probability predictions from a trained model.

    Args:
        model: Trained model with feature_cols_ attribute
        df_features: DataFrame with features

    Returns:
        Array of probability predictions
    """
    feature_cols = getattr(model, "feature_cols_", get_feature_columns())
    available_cols = [c for c in feature_cols if c in df_features.columns]

    X = df_features[available_cols].values
    X = np.nan_to_num(X, nan=0.0)

    return model.predict_proba(X)[:, 1]


def get_feature_importance(model: object) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model

    Returns:
        DataFrame with feature names and importance scores
    """
    feature_cols = getattr(model, "feature_cols_", get_feature_columns())

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = [0] * len(feature_cols)

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    })

    return df.sort_values("importance", ascending=False).reset_index(drop=True)
