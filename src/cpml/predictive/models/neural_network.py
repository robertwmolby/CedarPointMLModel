import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import R2Score, RootMeanSquaredError

from cpml.predictive.models.base import CPModel


class CPNeuralNetwork(CPModel):
    DEFAULT_SERIALIZER = "keras_dir"

    def __init__(self):
        super().__init__()
        self._rmse = None
        self._r2 = None
        self.model: keras.Sequential | None = None
        self.scaler_: MinMaxScaler | None = None
        self.scale_cols_: List[str] = []
        self.feature_names_: List[str] = []  # post-one-hot, pre-scaling column order
        self.metrics: dict = {}

    def create_model(self, crowd_level_df: pd.DataFrame):
        # One-hot day_of_week to align with your other models
        df = crowd_level_df.copy()
        X = df.drop(columns=["crowd_level"])
        y = df["crowd_level"].astype(float)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit scaler on numeric columns (if present)
        scale_cols = [c for c in ["forecast_temp", "actual_temp", "forecast_rain", "actual_rain"] if c in X_train.columns]
        scaler = MinMaxScaler() if scale_cols else None
        if scaler:
            X_train = X_train.copy()
            X_val   = X_val.copy()
            X_train.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
            X_val.loc[:,   scale_cols] = scaler.transform(X_val[scale_cols])

        n_features = X_train.shape[1]
        model = keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse",
                      metrics=[RootMeanSquaredError(name="rmse"), R2Score(name="r2")])

        model.fit(X_train.values, y_train.values,  # use .values to avoid Keras DF warnings
                  validation_data=(X_val.values, y_val.values),
                  epochs=50, batch_size=32, verbose=0)

        # Keras returns [loss, rmse, r2] in this order
        loss, rmse, r2 = model.evaluate(X_val.values, y_val.values, verbose=0)

        # Persist everything needed for inference + save
        self.model = model
        self.scaler_ = scaler
        self.scale_cols_ = scale_cols
        self.feature_names_ = list(X.columns)  # expected columns BEFORE scaling (after dummies)
        self.set_metrics(r=float(r2), rmse=float(rmse))
        return self

    def predict(self, prediction_df: pd.DataFrame) -> int:
        if self.model is None:
            raise RuntimeError("Model is not trained/loaded.")

        # One-hot and align columns to training layout
        df = pd.get_dummies(prediction_df, columns=["day_of_week"])
        missing = [c for c in self.feature_names_ if c not in df.columns]
        # Add any missing dummy columns with zeros
        for c in missing:
            df[c] = 0
        # Extra cols get dropped by reindex
        X = df.reindex(columns=self.feature_names_, fill_value=0)

        # Apply same scaling
        if self.scaler_ and self.scale_cols_:
            # Some scale cols might be absent after dummies—guard with intersection
            cols = [c for c in self.scale_cols_ if c in X.columns]
            if cols:
                X = X.copy()
                X.loc[:, cols] = self.scaler_.transform(X[cols])

        y_hat = float(self.model.predict(X.values, verbose=0).ravel()[0])
        return int(round(y_hat))
