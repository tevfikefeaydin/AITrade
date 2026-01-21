"""
MLPredictor - Temel ML tahmin sınıfı.

XGBoost ile fiyat yönü tahmini yapar.
"""

from typing import Optional, Tuple
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from loguru import logger

from src.analysis.indicators.simple import SimpleIndicators
from src.data.processor import DataProcessor


class MLPredictor:
    """XGBoost tabanlı fiyat tahmin sınıfı."""

    MODEL_PATH = Path("models/xgb_predictor.pkl")

    def __init__(self, model_path: Optional[Path] = None):
        """
        MLPredictor başlat.

        Args:
            model_path: Model dosya yolu (opsiyonel)
        """
        self.model_path = model_path or self.MODEL_PATH
        self.model = None
        self.feature_columns = []
        self._load_model()

    def _load_model(self) -> None:
        """Eğitilmiş modeli yükle."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
                    self.feature_columns = data.get("features", [])
                logger.info(f"Model yüklendi: {self.model_path}")
            except Exception as e:
                logger.warning(f"Model yüklenemedi: {e}")
                self.model = None
        else:
            logger.warning(f"Model dosyası bulunamadı: {self.model_path}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ML için özellik çıkarımı yap.

        Args:
            df: OHLCV DataFrame

        Returns:
            Özellikler eklenmiş DataFrame
        """
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Fiyat özellikleri
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        # Teknik indikatörler
        df["rsi"] = SimpleIndicators.rsi(close)
        df["ema_9"] = SimpleIndicators.ema(close, 9)
        df["ema_21"] = SimpleIndicators.ema(close, 21)
        df["sma_50"] = SimpleIndicators.sma(close, 50)

        # MACD
        macd, signal, hist = SimpleIndicators.macd(close)
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist

        # Bollinger Bands
        upper, middle, lower = SimpleIndicators.bollinger_bands(close)
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_position"] = (close - lower) / (upper - lower)

        # ATR
        df["atr"] = SimpleIndicators.atr(high, low, close)
        df["atr_percent"] = (df["atr"] / close) * 100

        # Volume özellikleri
        df["volume_ma"] = SimpleIndicators.sma(volume, 20)
        df["volume_ratio"] = volume / df["volume_ma"]

        # Mum özellikleri
        df["body_size"] = abs(close - df["open"]) / close
        df["upper_wick"] = (high - df[["open", "close"]].max(axis=1)) / close
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - low) / close

        # Lag özellikleri
        for lag in [1, 2, 3, 5]:
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)
            df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)

        # Trend özellikleri
        df["above_ema_9"] = (close > df["ema_9"]).astype(int)
        df["above_ema_21"] = (close > df["ema_21"]).astype(int)
        df["above_sma_50"] = (close > df["sma_50"]).astype(int)

        # NaN temizle
        df = df.dropna()

        return df

    async def predict(self, df: pd.DataFrame) -> float:
        """
        Fiyat yönü tahmini yap.

        Args:
            df: OHLCV DataFrame

        Returns:
            0-1 arası güven skoru (1 = kesin yukarı)
        """
        if self.model is None:
            logger.warning("Model yüklü değil, varsayılan skor döndürülüyor")
            return 0.5

        try:
            features_df = self.prepare_features(df)

            if features_df.empty:
                return 0.5

            # Son satırı al
            X = features_df[self.feature_columns].iloc[-1:].values

            # Tahmin
            proba = self.model.predict_proba(X)[0]
            confidence = proba[1]  # Yukarı yön olasılığı

            return float(confidence)

        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return 0.5

    def train(
        self,
        df: pd.DataFrame,
        target_periods: int = 1,
        test_size: float = 0.2,
    ) -> dict:
        """
        Model eğit.

        Args:
            df: OHLCV DataFrame
            target_periods: Hedef periyot (kaç mum sonrası)
            test_size: Test seti oranı

        Returns:
            Eğitim metrikleri
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
        except ImportError:
            logger.error("xgboost veya sklearn yüklü değil")
            return {"error": "Gerekli kütüphaneler yüklü değil"}

        logger.info("Model eğitimi başlıyor...")

        # Özellik çıkarımı
        features_df = self.prepare_features(df)

        # Target: Bir sonraki mum yukarı mı?
        features_df["target"] = (
            features_df["close"].shift(-target_periods) > features_df["close"]
        ).astype(int)
        features_df = features_df.dropna()

        # Feature kolonları
        exclude_cols = ["target", "open", "high", "low", "close", "volume"]
        self.feature_columns = [
            c for c in features_df.columns if c not in exclude_cols
        ]

        X = features_df[self.feature_columns].values
        y = features_df["target"].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Model eğit
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model.fit(X_train, y_train)

        # Değerlendirme
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Model eğitildi. Accuracy: {accuracy:.4f}")

        # Modeli kaydet
        self._save_model()

        return {
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": self.feature_columns,
        }

    def _save_model(self) -> None:
        """Modeli kaydet."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "features": self.feature_columns,
                },
                f,
            )
        logger.info(f"Model kaydedildi: {self.model_path}")
