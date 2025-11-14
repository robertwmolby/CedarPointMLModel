import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from lightgbm import LGBMRegressor
from cpml.predictive.models.base import CPModel


class CPLightGBModel(CPModel):
    """
    LightGBM regressor with:
      - legacy create_model(...) entrypoint
      - working predict(...) (single-row DF -> int)
      - feature_names_ + metrics for save/load
    """
    CLASS_NAME = "CPLightGBModel"
    DEFAULT_SERIALIZER = "lightgbm_txt"  # saves Booster to *.txt

    def __init__(self, **lgbm_kwargs):
        super().__init__(model_name="lightgbm")
        self._rmse = None
        self._r2 = None
        self.model: LGBMRegressor | None = None
        self._kwargs = {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.8,          # bagging_fraction
            "colsample_bytree": 0.8,   # feature_fraction
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "metric": "rmse",
        }
        self._kwargs.update(lgbm_kwargs)

    # --- training -------------------------------------------------------------
    def create_model(self, crowd_level_df: pd.DataFrame):
        X = crowd_level_df.drop(columns=["crowd_level"])
        y = crowd_level_df["crowd_level"].astype(float)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = LGBMRegressor(**self._kwargs).fit(X_train, y_train)
        self.feature_names_ = list(X_train.columns)

        y_pred = self.model.predict(X_val)
        r2 = float(r2_score(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        self.set_metrics(r2=r2, rmse=rmse)
        return self

    # Optional sklearn-style fit for when you move off create_model(...)
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = LGBMRegressor(**self._kwargs).fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    # --- inference ------------------------------------------------------------
    def predict(self, prediction_df: pd.DataFrame) -> int:
        if self.model is None:
            raise RuntimeError("Model is not trained.")

        missing = [c for c in self.feature_names_ if c not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = prediction_df[self.feature_names_]
        y_hat = float(self.model.predict(X)[0])
        return int(round(y_hat))
