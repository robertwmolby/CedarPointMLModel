import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

from cpml.predictive.models.base import CPModel


class CPLinearRegressionModel(CPModel):
    """
    Linear Regression with a preprocessing pipeline:
      - season_week normalized to season_week/26
      - MinMax scaling on temperature/rain columns
    Keeps legacy create_model(...) entrypoint and exposes predict(single-row DF)->int.
    """
    DEFAULT_SERIALIZER = "joblib"
    CLASS_NAME = "CPLinearRegressionModel"

    def __init__(self):
        super().__init__(model_name="linear_regression")
        self._rmse = None
        self._r2 = None
        self.model: Pipeline | None = None
        self.coeffs: dict[str, float] | None = None
        self._input_feature_names: list[str] = []   # raw, pre-transform columns

        # config used in pipeline
        self._cols_to_scale: List[str] = ["forecast_temp", "actual_temp", "forecast_rain", "actual_rain"]
        self._season_week_col = "season_week"

    # --- training (legacy entrypoint) -----------------------------------------
    def create_model(self, crowd_level_df: pd.DataFrame):
        df = crowd_level_df.copy()

        X = df.drop(columns=["crowd_level"])
        y = df["crowd_level"].astype(float)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self._input_feature_names = list(X_train.columns)  # expected columns at inference

        # --- build preprocessing
        transformers = []

        cols_to_scale = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
        if cols_to_scale:
            transformers.append(("minmax", MinMaxScaler(), cols_to_scale))

        pre = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",      # keep everything else as-is
            verbose_feature_names_out=False,
        )

        # Pipeline: preprocessor -> LinearRegression
        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("lr", LinearRegression()),
            ]
        )

        pipe.fit(X_train, y_train)
        self.model = pipe
        self.feature_names_ = list(X_train.columns)  # inputs expected by the pipeline

        # Eval
        y_pred = pipe.predict(X_val)
        r2 = float(r2_score(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        self.set_metrics(r2=r2, rmse=rmse)

        # Coefficients mapped to transformed names
        try:
            # get_feature_names_out is available on ColumnTransformer (sklearn >=1.0)
            transformed_names = pipe.named_steps["pre"].get_feature_names_out()
            lr = pipe.named_steps["lr"]
            coefs = lr.coef_.ravel().tolist()
            self.set_metrics(coefs = {name: float(w) for name, w in zip(transformed_names, coefs)})
        except Exception:
            self.coeffs = None  # optional; not critical for inference

        return self

    # Optional sklearn-style fit if you pass already-preprocessed X,y
    def fit(self, X: pd.DataFrame, y: pd.Series):
        lr = LinearRegression().fit(X, y)
        # Wrap in a degenerate pipeline so serialization stays consistent
        self.model = Pipeline(steps=[("identity", FunctionTransformer(lambda a: a, validate=False)), ("lr", lr)])
        self.feature_names_ = list(X.columns)
        self._input_feature_names = list(X.columns)
        return self

    # --- inference ------------------------------------------------------------
    def predict(self, prediction_df: pd.DataFrame) -> int:
        if self.model is None:
            raise RuntimeError("Model is not trained.")

        # Ensure required training columns exist
        missing = [c for c in self.feature_names_ if c not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Feed the raw row to the pipeline; it handles scaling/transform
        X = prediction_df[self.feature_names_]
        y_hat = float(self.model.predict(X)[0])
        return int(round(y_hat))
