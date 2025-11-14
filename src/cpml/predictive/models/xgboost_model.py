import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

from cpml.predictive.models.base import CPModel


class CPXGBoostModel(CPModel):
    """
    XGBoost regressor:
      - TimeSeriesSplit CV to pick best n_estimators
      - Final train on train+valid, test on held-out tail
      - feature_names_ + metrics for consistent save/load
      - predict(single-row DF) -> int
    """
    CLASS_NAME = "CPXGBoostModel"
    DEFAULT_SERIALIZER = "xgboost_json"   # saves *.json via Booster/Regressor.save_model()

    def __init__(
        self,
        *,
        learning_rate: float = 0.03,
        max_depth: int = 8,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        cv_splits: int = 5,
        max_rounds: int = 4000,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        **extra_kwargs
    ):
        super().__init__(model_name="xgboost")
        self._rmse = None
        self._r2 = None
        self.model: Optional[XGBRegressor] = None

        # Base params shared across CV & final fit
        self._params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "tree_method": tree_method,
            "device": "cuda",
            "random_state": random_state,
            **extra_kwargs,
        }
        self._cv_splits = cv_splits
        self._max_rounds = max_rounds
        self._esr = early_stopping_rounds

    # ---- training (legacy entrypoint) ----------------------------------------
    def create_model(self, crowd_level_df: pd.DataFrame):
        df = crowd_level_df.copy()

        X = df.drop(columns=["crowd_level"])
        y = df["crowd_level"].astype(float)

        # Hold out tail as final test (time order preserved)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )

        # TimeSeriesSplit CV to choose best #trees
        tscv = TimeSeriesSplit(n_splits=self._cv_splits)
        dtrain = xgb.DMatrix(X_trainval, label=y_trainval)
        folds = list(tscv.split(X_trainval, y_trainval))

        cv_res = xgb.cv(
            params=self._params,
            dtrain=dtrain,
            num_boost_round=self._max_rounds,
            folds=folds,
            early_stopping_rounds=self._esr,
            verbose_eval=False,
            seed=self._params.get("random_state", 42),
        )
        best_rounds = len(cv_res)

        # Final model on trainval with best_rounds
        self.model = XGBRegressor(
            n_estimators=best_rounds,
            **self._params,
        ).fit(X_trainval, y_trainval)

        self.feature_names_ = list(X_trainval.columns)

        # Evaluate on held-out test
        y_pred = self.model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        self.set_metrics(r2=r2, rmse=rmse)
        return self

    # Optional sklearn-style fit if you’ve already split
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Simple fit without CV; you can plug in your own CV upstream if you like
        self.model = XGBRegressor(**self._params).fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    # ---- inference -----------------------------------------------------------
    def predict(self, prediction_df: pd.DataFrame) -> int:
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        missing = [c for c in self.feature_names_ if c not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X = prediction_df[self.feature_names_]
        y_hat = float(self.model.predict(X)[0])
        return int(round(y_hat))
