"""
Walk-Forward Training for ML-Assisted Crypto Trading Research Pipeline.

Implements walk-forward (rolling) cross-validation for training per-symbol models:
- No shuffle: respects temporal order
- Rolling window: train on N days, test on M days, step forward
- Purge gap between train and test to prevent leakage
- Expanding window option (train_start always at data start)
- Per-fold metrics: AUC, logloss, and backtest metrics
- Final model trained on all data
- Ensemble support: LightGBM + XGBoost averaging

Model: LightGBM classifier (with sklearn fallback), optional XGBoost ensemble
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


# =============================================================================
# EnsembleModel: averages LightGBM + XGBoost predictions
# =============================================================================

class EnsembleModel:
    """
    Ensemble that averages predictions from multiple sub-models.

    Stores sub-models as a list, exposes predict_proba() that averages
    the positive-class probabilities. Serializable with joblib.
    """

    def __init__(self, models: list):
        """
        Args:
            models: List of fitted models, each with predict_proba().
        """
        self.models = models
        self.feature_cols_ = []  # Set externally after construction

    def predict_proba(self, X):
        """Average predict_proba across all sub-models."""
        probas = []
        for m in self.models:
            probas.append(m.predict_proba(X))
        avg = np.mean(probas, axis=0)
        return avg

    @property
    def feature_importances_(self):
        """Average feature importances across sub-models (if available)."""
        imps = []
        for m in self.models:
            if hasattr(m, "feature_importances_"):
                imps.append(m.feature_importances_)
        if not imps:
            return None
        return np.mean(imps, axis=0)


# =============================================================================
# Model factory functions
# =============================================================================

def get_model(params_override=None):
    """
    Get the LightGBM model to use for training.

    Tries LightGBM first, falls back to sklearn HistGradientBoostingClassifier.

    Args:
        params_override: Optional dict to override default LGBM_PARAMS.

    Returns:
        Model instance
    """
    try:
        import lightgbm as lgb
        params = dict(config.LGBM_PARAMS)
        if params_override:
            params.update(params_override)
        model = lgb.LGBMClassifier(**params)
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


def get_xgb_model(params_override=None):
    """
    Get an XGBoost model for ensemble training.

    Args:
        params_override: Optional dict to override default XGBM_PARAMS.

    Returns:
        XGBClassifier instance

    Raises:
        ImportError: If xgboost is not installed.
    """
    import xgboost as xgb
    params = dict(config.XGBM_PARAMS)
    if params_override:
        params.update(params_override)
    model = xgb.XGBClassifier(**params)
    logger.info("Using XGBoost classifier (ensemble member)")
    return model


def walk_forward_split(
    df: pd.DataFrame,
    train_window_days: int,
    test_window_days: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.

    Supports purge gap (config.PURGE_GAP_HOURS) between train and test,
    and expanding window mode (config.USE_EXPANDING_WINDOW).

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
    purge_delta = pd.Timedelta(hours=config.PURGE_GAP_HOURS)

    data_start = df["open_time"].min()
    end_time = df["open_time"].max()

    splits = []
    current_train_start = data_start

    while True:
        train_end = current_train_start + train_delta
        # Purge gap: test starts after train_end + purge_delta
        test_start = train_end + purge_delta
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

        if config.USE_EXPANDING_WINDOW:
            # Expanding: train_start stays at data_start, train_end grows
            train_delta = train_delta + test_delta
        else:
            # Sliding: train_start moves forward
            current_train_start = current_train_start + test_delta

    mode_label = "expanding" if config.USE_EXPANDING_WINDOW else "sliding"
    logger.info(
        f"Generated {len(splits)} walk-forward folds "
        f"(mode={mode_label}, purge={config.PURGE_GAP_HOURS}h)"
    )
    return splits


def _fit_lgbm(model, X_train, y_train, X_test, y_test_binary, has_two_classes, sample_weights=None):
    """Fit a LightGBM model with optional early stopping and sample weights."""
    try:
        import lightgbm as lgb
        is_lgbm = isinstance(model, lgb.LGBMClassifier)
    except ImportError:
        is_lgbm = False

    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    if is_lgbm and has_two_classes:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test_binary)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
            **fit_kwargs,
        )
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    return model


def _fit_xgb(model, X_train, y_train, X_test, y_test_binary, has_two_classes, sample_weights=None):
    """Fit an XGBoost model with optional early stopping and sample weights."""
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    if has_two_classes:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test_binary)],
            verbose=False,
            **fit_kwargs,
        )
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    return model


def train_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "label",
    lgbm_override: Optional[Dict] = None,
    xgb_override: Optional[Dict] = None,
) -> Tuple[object, Dict[str, float]]:
    """
    Train model on a single fold and compute metrics.

    When config.USE_ENSEMBLE is True, trains both LightGBM and XGBoost,
    returns an EnsembleModel that averages their predictions.

    Args:
        train_df: Training data with features and label
        test_df: Test data with features and label
        feature_cols: List of feature column names
        target_col: Name of target column
        lgbm_override: Optional dict to override LightGBM params
        xgb_override: Optional dict to override XGBoost params

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
    # Timeouts with exit return in upper half of barrier range -> 1 (success)
    # Timeouts with exit return in lower half -> 0 (failure)
    # This captures the key insight: positive-return timeouts are not failures
    raw_labels = y_train.copy()
    y_train = (y_train >= 0.5).astype(int)
    y_test_binary = (y_test >= 0.5).astype(int)

    # Compute sample weights: confident labels (near 0 or 1) get higher weight,
    # ambiguous labels (near 0.5) get lower weight (min 0.3)
    sample_weights = np.clip(np.abs(2 * raw_labels - 1), 0.3, 1.0)

    has_two_classes = len(np.unique(y_test_binary)) > 1

    # Train LightGBM
    lgbm_model = get_model(params_override=lgbm_override)
    lgbm_model = _fit_lgbm(
        lgbm_model, X_train, y_train, X_test, y_test_binary,
        has_two_classes, sample_weights=sample_weights,
    )

    best_iter = getattr(
        lgbm_model, "best_iteration_",
        lgbm_model.n_estimators if hasattr(lgbm_model, "n_estimators") else 500,
    )

    # Ensemble: also train XGBoost
    if config.USE_ENSEMBLE:
        try:
            xgb_model = get_xgb_model(params_override=xgb_override)
            xgb_model = _fit_xgb(
                xgb_model, X_train, y_train, X_test, y_test_binary,
                has_two_classes, sample_weights=sample_weights,
            )
            model = EnsembleModel([lgbm_model, xgb_model])
        except ImportError:
            logger.warning("XGBoost not available, falling back to LightGBM only")
            model = lgbm_model
    else:
        model = lgbm_model

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


def _run_folds(
    splits,
    feature_cols,
    symbol,
    pass_label="",
    lgbm_override=None,
    xgb_override=None,
):
    """Run walk-forward folds and collect results, OOS predictions, importances."""
    fold_results = []
    oos_records = []
    fold_importances = []
    best_iterations = []

    desc = f"Training {symbol}" + (f" ({pass_label})" if pass_label else "")
    for i, (train_df, test_df) in enumerate(tqdm(splits, desc=desc)):
        model, metrics = train_fold(
            train_df, test_df, feature_cols,
            lgbm_override=lgbm_override,
            xgb_override=xgb_override,
        )

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

        # Collect feature importance (works for both single model and ensemble)
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            fold_importances.append(fi)

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
    lgbm_override: Optional[Dict] = None,
    xgb_override: Optional[Dict] = None,
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
        lgbm_override: Optional dict to override LightGBM params per fold
        xgb_override: Optional dict to override XGBoost params per fold

    Returns:
        Tuple of (final_model, fold_results, aggregate_metrics)
    """
    train_window_days = train_window_days or config.DEFAULT_TRAIN_WINDOW_DAYS
    test_window_days = test_window_days or config.DEFAULT_TEST_WINDOW_DAYS

    logger.info(f"Starting walk-forward training for {symbol}")
    logger.info(f"Train window: {train_window_days} days, Test window: {test_window_days} days")
    if config.USE_ENSEMBLE:
        logger.info("Ensemble mode: LightGBM + XGBoost")

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

    # -- PASS 1: Train all folds with full feature set --------------------
    fold_results, oos_records, fold_importances, best_iterations = _run_folds(
        splits, available_cols, symbol, pass_label="Pass 1",
        lgbm_override=lgbm_override, xgb_override=xgb_override,
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

    # -- PASS 2: Retrain with pruned features if any were dropped ---------
    if len(pruned_cols) < len(available_cols):
        logger.info(f"Retraining with {len(pruned_cols)} features (was {len(available_cols)})...")
        fold_results, oos_records, fold_importances, best_iterations = _run_folds(
            splits, pruned_cols, symbol, pass_label="Pass 2",
            lgbm_override=lgbm_override, xgb_override=xgb_override,
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

    # -- Train final model on all data ------------------------------------
    logger.info("Training final model on all data...")
    X_all = df_merged[pruned_cols].values
    y_all = (df_merged["label"].values >= 0.5).astype(int)  # Binarize fractional labels
    X_all = np.nan_to_num(X_all, nan=0.0)

    lgbm_final = get_model(params_override=lgbm_override)
    # Use average best iteration from folds (minimum 10)
    avg_best_iter = None
    if best_iterations:
        avg_best_iter = max(10, int(np.mean(best_iterations)))
        lgbm_final.n_estimators = avg_best_iter
        logger.info(f"Final LightGBM n_estimators set to {avg_best_iter} (fold average)")
    lgbm_final.fit(X_all, y_all)

    if config.USE_ENSEMBLE:
        try:
            xgb_final = get_xgb_model(params_override=xgb_override)
            if avg_best_iter is not None:
                xgb_final.n_estimators = avg_best_iter
            xgb_final.fit(X_all, y_all)
            final_model = EnsembleModel([lgbm_final, xgb_final])
            logger.info("Final model: EnsembleModel (LightGBM + XGBoost)")
        except ImportError:
            logger.warning("XGBoost not available for final model, using LightGBM only")
            final_model = lgbm_final
    else:
        final_model = lgbm_final

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

    Works transparently with both single models and EnsembleModel.

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

    For EnsembleModel, returns averaged importances across sub-models.

    Args:
        model: Trained model

    Returns:
        DataFrame with feature names and importance scores
    """
    feature_cols = getattr(model, "feature_cols_", get_feature_columns())

    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        importance = [0] * len(feature_cols)

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    })

    return df.sort_values("importance", ascending=False).reset_index(drop=True)
