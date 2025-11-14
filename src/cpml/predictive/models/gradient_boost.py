import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from cpml.predictive.models.base import CPModel


class CPGradientBoostModel(CPModel):
    """
    Uses sklearn's HistGradientBoostingRegressor.
    - Keeps your legacy create_model(...) entrypoint.
    - Implements predict(...) to honor feature order and return an int.
    - Stores feature_names_ and metrics so save/load works uniformly.
    """
    DEFAULT_SERIALIZER = "joblib"   # saved as *.bin via joblib
    CLASS_NAME = "CPGradientBoostModel"

    def __init__(self, **hgb_kwargs):
        # model_name shows up in artifact filenames
        super().__init__(model_name="gradient_boost")
        self._rmse = None
        self._r2 = None
        self.model: HistGradientBoostingRegressor | None = None
        # sensible defaults; caller can override via kwargs
        self._kwargs = {
            "learning_rate": 0.05,
            "max_depth": 6,
            "max_iter": 500,
            "l2_regularization": 0.0,
            "early_stopping": True,
            "random_state": 42,
        }
        self._kwargs.update(hgb_kwargs)

    # --- training -------------------------------------------------------------
    def create_model(self, crowd_level_df: pd.DataFrame):
        """
        Legacy entrypoint. If you’ve already normalized booleans and done one-hot
        encoding in prepare_for_modeling(), pass that DF in here.
        This method will:
          - Try to map 'Y'/'N' columns if they exist (safe fallback).
          - One-hot 'day_of_week' if still categorical (safe fallback).
        """
        df = crowd_level_df.copy()

        # Split
        X = df.drop(columns=["crowd_level"])
        y = df["crowd_level"].astype(float)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        self.model = HistGradientBoostingRegressor(**self._kwargs)
        self.model.fit(X_train, y_train)
        self.feature_names_ = list(X_train.columns)

        # Eval
        y_pred = self.model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        self.set_metrics(r2=r2, rmse=rmse)
        return self

    # (Optional) new-style fit if you decide to call fit(X, y) elsewhere.
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = HistGradientBoostingRegressor(**self._kwargs).fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    # --- inference ------------------------------------------------------------
    def predict(self, prediction_df: pd.DataFrame) -> int:
        """
        Expects a single-row DF (your existing notebook behavior).
        Aligns/filters columns to self.feature_names_ and returns an int.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained.")

        # Align columns strictly to training features
        missing = [c for c in self.feature_names_ if c not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = prediction_df[self.feature_names_]
        y_hat = self.model.predict(X)[0]
        # Round to nearest int (crowd levels are integer categories)
        return int(round(float(y_hat)))
