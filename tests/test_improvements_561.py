"""
Tests for Improvements 5, 6, and 1:
- Improvement 5: Model Ensemble (EnsembleModel)
- Improvement 6: Walk-Forward Improvements (purge gap, expanding window)
- Improvement 1: Optuna Hyperparameter Optimization
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src import config
from src.train import (
    EnsembleModel,
    walk_forward_split,
    train_fold,
)


# =============================================================================
# Improvement 6: Walk-Forward Improvements
# =============================================================================

def test_purge_gap_separates_train_test():
    """Verify purge gap creates separation between train end and test start."""
    n_rows = 500
    start = datetime(2024, 1, 1)
    df = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(n_rows)],
        "label": np.random.randint(0, 2, n_rows),
    })

    # Use a 12h purge gap
    with patch.object(config, "PURGE_GAP_HOURS", 12):
        with patch.object(config, "USE_EXPANDING_WINDOW", False):
            splits = walk_forward_split(df, train_window_days=5, test_window_days=2)

    assert len(splits) > 0, "Should generate at least one split"

    for train_df, test_df in splits:
        train_end = train_df["open_time"].max()
        test_start = test_df["open_time"].min()
        gap = test_start - train_end
        # The gap between the last train sample and first test sample
        # should be >= purge_gap_hours (12h). Since data is hourly,
        # the actual gap is at least 12h.
        assert gap >= timedelta(hours=12), (
            f"Purge gap violated: train_end={train_end}, test_start={test_start}, "
            f"gap={gap}"
        )


def test_expanding_window_grows_train_set():
    """Verify expanding window mode grows training set each fold."""
    n_rows = 1000
    start = datetime(2024, 1, 1)
    df = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(n_rows)],
        "label": np.random.randint(0, 2, n_rows),
    })

    with patch.object(config, "PURGE_GAP_HOURS", 0):
        with patch.object(config, "USE_EXPANDING_WINDOW", True):
            splits = walk_forward_split(df, train_window_days=10, test_window_days=5)

    assert len(splits) >= 2, "Need at least 2 splits to verify growth"

    # Each subsequent fold should have a larger training set
    train_sizes = [len(train_df) for train_df, _ in splits]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] > train_sizes[i - 1], (
            f"Expanding window: fold {i} train_size={train_sizes[i]} "
            f"should be > fold {i-1} train_size={train_sizes[i-1]}"
        )


def test_purge_gap_zero_matches_original_behavior():
    """With purge_gap=0 and sliding window, behavior matches original."""
    n_rows = 500
    start = datetime(2024, 1, 1)
    df = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(n_rows)],
        "label": np.random.randint(0, 2, n_rows),
    })

    with patch.object(config, "PURGE_GAP_HOURS", 0):
        with patch.object(config, "USE_EXPANDING_WINDOW", False):
            splits = walk_forward_split(df, train_window_days=5, test_window_days=2)

    assert len(splits) > 0
    # With 0 purge gap, test_start should be right after train_end
    for train_df, test_df in splits:
        train_end = train_df["open_time"].max()
        test_start = test_df["open_time"].min()
        gap = test_start - train_end
        # Gap should be exactly 1 hour (the next data point)
        assert gap <= timedelta(hours=2), (
            f"With purge=0, gap should be minimal but got {gap}"
        )


# =============================================================================
# Improvement 5: Model Ensemble
# =============================================================================

class MockModel:
    """Simple mock model for testing EnsembleModel."""

    def __init__(self, prob):
        self.prob = prob
        self.feature_importances_ = np.array([0.5, 0.3, 0.2])

    def predict_proba(self, X):
        n = len(X)
        pos = np.full(n, self.prob)
        neg = 1 - pos
        return np.column_stack([neg, pos])


def test_ensemble_model_averages_probabilities():
    """EnsembleModel.predict_proba averages sub-model predictions."""
    model_a = MockModel(prob=0.8)
    model_b = MockModel(prob=0.4)

    ensemble = EnsembleModel([model_a, model_b])

    X = np.zeros((5, 3))
    proba = ensemble.predict_proba(X)

    # Average of 0.8 and 0.4 = 0.6 for positive class
    expected_pos = 0.6
    np.testing.assert_allclose(proba[:, 1], expected_pos, atol=1e-10)

    # Negative class should be 1 - 0.6 = 0.4
    expected_neg = 0.4
    np.testing.assert_allclose(proba[:, 0], expected_neg, atol=1e-10)


def test_ensemble_model_feature_importances():
    """EnsembleModel.feature_importances_ averages sub-model importances."""
    model_a = MockModel(prob=0.5)
    model_a.feature_importances_ = np.array([10.0, 20.0, 30.0])
    model_b = MockModel(prob=0.5)
    model_b.feature_importances_ = np.array([30.0, 10.0, 20.0])

    ensemble = EnsembleModel([model_a, model_b])

    expected = np.array([20.0, 15.0, 25.0])
    np.testing.assert_allclose(ensemble.feature_importances_, expected)


def test_ensemble_model_serializable():
    """EnsembleModel can be serialized and deserialized with joblib."""
    import joblib
    import tempfile
    import os

    model_a = MockModel(prob=0.7)
    model_b = MockModel(prob=0.3)
    ensemble = EnsembleModel([model_a, model_b])
    ensemble.feature_cols_ = ["feat_a", "feat_b", "feat_c"]

    X = np.zeros((3, 3))
    original_proba = ensemble.predict_proba(X)

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        joblib.dump(ensemble, path)
        loaded = joblib.load(path)

        assert isinstance(loaded, EnsembleModel)
        assert loaded.feature_cols_ == ["feat_a", "feat_b", "feat_c"]
        loaded_proba = loaded.predict_proba(X)
        np.testing.assert_allclose(loaded_proba, original_proba)
    finally:
        os.unlink(path)


def test_ensemble_model_single_model():
    """EnsembleModel with a single model returns that model's predictions."""
    model = MockModel(prob=0.75)
    ensemble = EnsembleModel([model])

    X = np.zeros((4, 3))
    proba = ensemble.predict_proba(X)

    np.testing.assert_allclose(proba[:, 1], 0.75, atol=1e-10)


# =============================================================================
# Improvement 1: Optuna Smoke Test
# =============================================================================

def test_optuna_smoke():
    """Run 2 Optuna trials on tiny data, verify study returns."""
    pytest.importorskip("optuna")
    from src.optimize import run_optimization
    from src.features import get_feature_columns

    # Create synthetic dataset with enough rows for walk-forward splits
    # 3 days train + 2 days test = 5 days minimum per fold
    # Use 15 days of hourly data = 360 rows, enough for 2 folds
    np.random.seed(42)
    n = 360
    start = datetime(2024, 1, 1)
    feature_cols = get_feature_columns()

    df_features = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(n)],
    })
    for col in feature_cols:
        df_features[col] = np.random.randn(n)

    df_labeled = pd.DataFrame({
        "entry_time": df_features["open_time"].values,
        "label": np.random.choice([0.0, 1.0], size=n),
        "signal_type_encoded": np.zeros(n, dtype=int),
    })

    # Run with very small trial count, short windows so folds generate
    with patch.object(config, "DEFAULT_TRAIN_WINDOW_DAYS", 3):
        with patch.object(config, "DEFAULT_TEST_WINDOW_DAYS", 2):
            with patch.object(config, "PURGE_GAP_HOURS", 0):
                with patch.object(config, "USE_ENSEMBLE", False):
                    study = run_optimization(
                        symbol="TEST",
                        df_features=df_features,
                        df_labeled=df_labeled,
                        n_trials=2,
                        timeout=120,
                    )

    assert study is not None, "Study should be returned"
    assert len(study.trials) == 2, f"Expected 2 trials, got {len(study.trials)}"
    assert study.best_value > 0, "Best AUC should be positive"
    assert "num_leaves" in study.best_params
    assert "learning_rate" in study.best_params
