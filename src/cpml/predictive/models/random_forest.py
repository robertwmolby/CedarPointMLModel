import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from cpml.predictive.models.base import CPModel


class CPRandomForestModel(CPModel):
    """
    RandomForestRegressor with:
      - legacy create_model(...)
      - predict(single-row DF) -> int
      - feature_names_ + metrics for meta
    """
    CLASS_NAME = "CPRandomForestModel"
    DEFAULT_SERIALIZER = "joblib"

    def __init__(self, **rf_kwargs):
        super().__init__(model_name="random_forest")
        self._rmse = None
        self._r2 = None
        self.model: RandomForestRegressor | None = None
        self._kwargs = {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,
        }
        self._kwargs.update(rf_kwargs)

    # training (legacy entrypoint)
    def create_model(self, crowd_level_df: pd.DataFrame):
        df = crowd_level_df.copy()

        X = df.drop(columns=["crowd_level"])
        y = df["crowd_level"].astype(float)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(**self._kwargs).fit(X_train, y_train)
        self.feature_names_ = list(X_train.columns)

        y_pred = self.model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))
        self.set_metrics(rmse=rmse, r2=r2)
        return self

    # optional sklearn-style fit
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = RandomForestRegressor(**self._kwargs).fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    # inference
    def predict(self, prediction_df: pd.DataFrame) -> int:
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        missing = [c for c in self.feature_names_ if c not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X = prediction_df[self.feature_names_]
        y_hat = float(self.model.predict(X)[0])
        return int(round(y_hat))
