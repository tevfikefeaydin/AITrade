"""
Optuna hyperparameter optimization for LightGBM + XGBoost.

Runs walk-forward cross-validation with Optuna-suggested hyperparameters
and saves the best parameters to models/{symbol}_best_params.json.
"""

import json
import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .features import get_feature_columns
from .train import walk_forward_split, train_fold

logger = logging.getLogger(__name__)


def create_objective(
    symbol: str,
    df_merged: pd.DataFrame,
    feature_cols: List[str],
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
):
    """
    Create Optuna objective that returns mean OOS AUC.

    Args:
        symbol: Symbol name (for logging).
        df_merged: Merged features + labels DataFrame.
        feature_cols: Feature column names.
        splits: Walk-forward train/test splits.

    Returns:
        Callable objective for Optuna study.
    """
    def objective(trial):
        # Suggest LightGBM hyperparameters
        lgbm_params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        }

        # Run walk-forward folds with these params (no model saving)
        fold_aucs = []
        for i, (train_df, test_df) in enumerate(splits):
            try:
                model, metrics = train_fold(
                    train_df, test_df, feature_cols,
                    lgbm_override=lgbm_params,
                )
                fold_aucs.append(metrics["auc"])
            except Exception as e:
                logger.debug(f"Trial {trial.number}, fold {i} failed: {e}")
                fold_aucs.append(0.5)  # Penalize failed folds

        mean_auc = np.mean(fold_aucs)
        logger.debug(
            f"Trial {trial.number}: mean_auc={mean_auc:.4f} "
            f"(params: num_leaves={lgbm_params['num_leaves']}, "
            f"lr={lgbm_params['learning_rate']:.4f})"
        )
        return mean_auc

    return objective


def run_optimization(
    symbol: str,
    df_features: pd.DataFrame,
    df_labeled: pd.DataFrame,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
):
    """
    Run Optuna study and save best params.

    Args:
        symbol: Symbol to optimize for.
        df_features: DataFrame with features.
        df_labeled: DataFrame with labels.
        n_trials: Number of Optuna trials (default: config.OPTUNA_N_TRIALS).
        timeout: Timeout in seconds (default: config.OPTUNA_TIMEOUT_SECONDS).

    Returns:
        optuna.Study instance with results.
    """
    import optuna

    n_trials = n_trials or config.OPTUNA_N_TRIALS
    timeout = timeout or config.OPTUNA_TIMEOUT_SECONDS

    logger.info(f"Starting Optuna optimization for {symbol}")
    logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")

    # Merge features with labels
    label_cols = ["entry_time", "label"]
    if "signal_type_encoded" in df_labeled.columns:
        label_cols.append("signal_type_encoded")

    df_merged = df_features.merge(
        df_labeled[label_cols],
        left_on="open_time",
        right_on="entry_time",
        how="inner",
    )

    if "signal_type_encoded" not in df_merged.columns:
        df_merged["signal_type_encoded"] = 0

    if len(df_merged) == 0:
        logger.warning(f"No labeled samples for {symbol}")
        return None

    # Get feature columns
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df_merged.columns]

    # Build walk-forward splits
    splits = walk_forward_split(
        df_merged,
        config.DEFAULT_TRAIN_WINDOW_DAYS,
        config.DEFAULT_TEST_WINDOW_DAYS,
    )

    if len(splits) == 0:
        logger.warning("No valid splits for optimization")
        return None

    logger.info(f"Optimizing over {len(splits)} walk-forward folds, {len(available_cols)} features")

    # Create and run study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{symbol}_hpo",
    )

    objective = create_objective(symbol, df_merged, available_cols, splits)

    # Suppress Optuna's trial-level logging (we log ourselves)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Log results
    logger.info(f"Optimization complete for {symbol}")
    logger.info(f"  Best AUC: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")
    logger.info(f"  Trials completed: {len(study.trials)}")

    # Save best params
    best_params_path = config.get_symbol_best_params_path(symbol)
    best_params_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "lgbm_params": study.best_params,
        "best_auc": study.best_value,
        "n_trials": len(study.trials),
        "symbol": symbol,
    }

    with open(best_params_path, "w") as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Saved best params to {best_params_path}")

    return study
