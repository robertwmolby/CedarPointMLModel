import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

from cpml.predictive.models.base import CPModel


class CPPolyLinearRegressionModel(CPModel):
    """
    Linear Regression with PolynomialFeatures on `season_week`.
    - Keeps legacy create_model(...)
    - Predict(single-row DF) -> int
    - Saves feature_names_ (expected raw inputs) + metrics for meta
    """
    DEFAULT_SERIALIZER = "joblib"
    CLASS_NAME = "CPPolyLinearRegressionModel"

    def __init__(self, degree: int = 3, include_bias: bool = False):
        super().__init__(model_name="linear_regression_poly")
        self._rmse = None
        self._r2 = None
        self.model: Pipeline | None = None
        self.coeffs: dict[str, float] | None = None
        self._degree = degree
        self._include_bias = include_bias
        self.feature_names_: list[str] = []  # expected raw input columns (pre-transform)

    # --- training (legacy entrypoint) -----------------------------------------
    def create_model(self, crowd_level_df: pd.DataFrame):
        df = crowd_level_df.copy()

        if "season_week" not in df.columns:
            raise ValueError("Expected 'season_week' column for polynomial features.")

        # Split
        X = df.drop(columns=["crowd_level"])
        y = df["crowd_level"].astype(float)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.feature_names_ = list(X_train.columns)

        # Build transformer: poly(season_week) + passthrough others
        poly = PolynomialFeatures(degree=self._degree, include_bias=self._include_bias)
        ct = ColumnTransformer(
            transformers=[
                ("poly_week", poly, ["season_week"]),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        pipe = Pipeline(steps=[("pre", ct), ("lr", LinearRegression())])
        pipe.fit(X_train, y_train)
        self.model = pipe

        # Metrics
        y_pred = pipe.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2 = float(r2_score(y_val, y_pred))
        self.set_metrics(r2=r2, rmse=rmse)

        # Coefficients on transformed feature names (optional)
        try:
            transformed_names = pipe.named_steps["pre"].get_feature_names_out()
            coefs = pipe.named_steps["lr"].coef_.ravel().tolist()
            self.set_metrics(coeffs={name: float(w) for name, w in zip(transformed_names, coefs)})
        except Exception:
            self.coeffs = None

        return self

    # Optional sklearn-style fit if you’re passing already-prepped X,y
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if "season_week" not in X.columns:
            raise ValueError("Expected 'season_week' in X for polynomial expansion.")
        poly = PolynomialFeatures(degree=self._degree, include_bias=self._include_bias)
        ct = ColumnTransformer(
            transformers=[("poly_week", poly, ["season_week"])],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.model = Pipeline(steps=[("pre", ct), ("lr", LinearRegression())]).fit(X, y)
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
