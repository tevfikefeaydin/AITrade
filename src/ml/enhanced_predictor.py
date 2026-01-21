"""
EnhancedPredictor - Gelişmiş ML tahmin sınıfı.

Ensemble metodları ve çoklu model desteği.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from loguru import logger

from src.ml.predictor import MLPredictor


class EnhancedPredictor:
    """Ensemble tabanlı gelişmiş tahmin sınıfı."""

    def __init__(self):
        """EnhancedPredictor başlat."""
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.base_predictor = MLPredictor()

    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
    ) -> None:
        """
        Ensemble'a model ekle.

        Args:
            name: Model adı
            model: Scikit-learn uyumlu model
            weight: Model ağırlığı
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Model eklendi: {name} (ağırlık: {weight})")

    async def predict(
        self,
        df: pd.DataFrame,
        method: str = "weighted_average",
    ) -> float:
        """
        Ensemble tahmin yap.

        Args:
            df: OHLCV DataFrame
            method: Birleştirme metodu (weighted_average, voting, stacking)

        Returns:
            0-1 arası güven skoru
        """
        if not self.models:
            # Modeller yoksa base predictor kullan
            return await self.base_predictor.predict(df)

        features_df = self.base_predictor.prepare_features(df)
        if features_df.empty:
            return 0.5

        X = features_df[self.base_predictor.feature_columns].iloc[-1:].values

        predictions = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    predictions[name] = proba[1]
                else:
                    pred = model.predict(X)[0]
                    predictions[name] = float(pred)
            except Exception as e:
                logger.warning(f"{name} tahmin hatası: {e}")

        if not predictions:
            return 0.5

        if method == "weighted_average":
            return self._weighted_average(predictions)
        elif method == "voting":
            return self._voting(predictions)
        else:
            return self._weighted_average(predictions)

    def _weighted_average(self, predictions: Dict[str, float]) -> float:
        """Ağırlıklı ortalama."""
        total_weight = 0
        weighted_sum = 0

        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0)
            weighted_sum += pred * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    def _voting(self, predictions: Dict[str, float]) -> float:
        """Oylama."""
        votes_up = sum(1 for p in predictions.values() if p > 0.5)
        votes_down = len(predictions) - votes_up

        if votes_up > votes_down:
            return 0.5 + (votes_up / len(predictions)) * 0.5
        elif votes_down > votes_up:
            return 0.5 - (votes_down / len(predictions)) * 0.5
        else:
            return 0.5

    def train_ensemble(
        self,
        df: pd.DataFrame,
        models_config: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Ensemble modellerini eğit.

        Args:
            df: OHLCV DataFrame
            models_config: Model konfigürasyonları

        Returns:
            Eğitim sonuçları
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
        except ImportError:
            logger.error("Gerekli kütüphaneler yüklü değil")
            return {"error": "Kütüphaneler eksik"}

        # Varsayılan modeller
        if models_config is None:
            models_config = [
                {"name": "xgb", "model": XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="logloss"), "weight": 1.5},
                {"name": "rf", "model": RandomForestClassifier(n_estimators=100, max_depth=5), "weight": 1.0},
                {"name": "gb", "model": GradientBoostingClassifier(n_estimators=100, max_depth=5), "weight": 1.2},
            ]

        # Özellik hazırla
        features_df = self.base_predictor.prepare_features(df)
        features_df["target"] = (
            features_df["close"].shift(-1) > features_df["close"]
        ).astype(int)
        features_df = features_df.dropna()

        self.base_predictor.feature_columns = [
            c for c in features_df.columns
            if c not in ["target", "open", "high", "low", "close", "volume"]
        ]

        X = features_df[self.base_predictor.feature_columns].values
        y = features_df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        results = {}
        for config in models_config:
            name = config["name"]
            model = config["model"]
            weight = config.get("weight", 1.0)

            logger.info(f"{name} eğitiliyor...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self.add_model(name, model, weight)
            results[name] = {"accuracy": accuracy}

            logger.info(f"{name} accuracy: {accuracy:.4f}")

        return results

    def save(self, path: Path) -> None:
        """Ensemble'ı kaydet."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "models": self.models,
                    "weights": self.weights,
                    "features": self.base_predictor.feature_columns,
                },
                f,
            )
        logger.info(f"Ensemble kaydedildi: {path}")

    def load(self, path: Path) -> None:
        """Ensemble'ı yükle."""
        if not path.exists():
            logger.warning(f"Dosya bulunamadı: {path}")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.models = data.get("models", {})
            self.weights = data.get("weights", {})
            self.base_predictor.feature_columns = data.get("features", [])

        logger.info(f"Ensemble yüklendi: {path}")
