# cpml/predictive/predictor.py
import numpy as np
import pandas as pd
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
from cpml.predictive.prediction_request import PredictionRequest
from cpml.predictive.park_closed_error import ParkClosedError
from cpml.common.cp_date_handler import is_open
from cpml.predictive.load import load_model


class CPPredictor:
    """
    General-purpose predictor for Cedar Point crowd level.

    Typical usage:
        predictor = CPPredictor(storage_dir=STORAGE_DIR)  # loads artifacts/latest
        y = predictor.predict(date(2025, 10, 29), actual_temp=50, actual_rain=0)

    Or:
        req = PredictionRequest(date(2025, 10, 29), actual_temp=50, actual_rain=0)
        y = predictor.predict_from_request(req)
    """

    def __init__(
        self,
        storage_dir: Union[str, Path],
        *,
        model_path: Optional[Union[str, Path]] = None,
        latest_subpath: str = "artifacts/latest",
        validate_open: bool = True,
    ) -> None:
        """
        :param storage_dir: Root directory used by the notebook (STORAGE_DIR).
        :param model_path: Optional explicit model artifact path (file or Keras dir).
        :param latest_subpath: Relative path under storage_dir for the 'latest' pointer.
        :param validate_open: If True, raises ParkClosedError when date is closed.
        """
        self.storage_dir = Path(storage_dir)
        self.latest_dir = self.storage_dir / latest_subpath
        self._explicit_model_path = Path(model_path) if model_path else None
        self._model = None  # CPModel instance
        self._model_path: Optional[Path] = None
        self._validate_open = validate_open

    # ------------------------
    # Public API
    # ------------------------
    def predict(
        self,
        prediction_date: date,
        actual_temp: float,
        actual_rain: float,
        **kwargs: Any,
    ) -> float:
        """
        Convenience wrapper that builds a PredictionRequest and predicts.
        """
        if self._validate_open and not is_open(prediction_date):
            raise ParkClosedError(f"Park is closed on {prediction_date.isoformat()}")

        req = PredictionRequest(prediction_date, actual_temp, actual_rain, **kwargs)
        return self.predict_from_request(req)

    def predict_from_request(self, request: PredictionRequest) -> float:
        """
        Predict from a fully-populated PredictionRequest.
        Returns a scalar float crowd level.
        """
        if not is_open(request.prediction_date):
            raise ParkClosedError(f"Park is closed on {request.prediction_date.isoformat()}")
        self._ensure_model_loaded()

        df = self._build_df(request)
        self._align_features(df)

        y = self._model.predict(df)
        # squeeze -> scalar
        return float(np.squeeze(y))

    def model_metrics(self) -> Dict[str, float]:
        """
        Returns metrics saved with the model (e.g., {"r2": ..., "rmse": ...}).
        """
        self._ensure_model_loaded()
        return dict(self._model.metrics or {})

    def model_feature_names(self) -> Iterable[str]:
        """
        Returns the feature names the model expects (order matters).
        """
        self._ensure_model_loaded()
        return list(getattr(self._model, "feature_names_", []) or [])

    def model_path(self) -> Path:
        """
        Returns the resolved model path currently in use.
        """
        self._ensure_model_loaded()
        return self._model_path

    # ------------------------
    # Internals
    # ------------------------
    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        model_path = self._explicit_model_path or self._pick_latest_model_path(self.latest_dir)
        self._model = load_model(model_path)
        self._model_path = Path(model_path)

    @staticmethod
    def _pick_latest_model_path(latest_dir: Path) -> Path:
        """
        Resolve a model artifact inside artifacts/latest.
        Supports:
          - Keras directory containing model.meta.json
          - File-based models with a sibling *.meta.json
        """
        if not latest_dir.exists():
            raise FileNotFoundError(f"{latest_dir} does not exist")

        # File-based models (.joblib, .json, .txt) with *.meta.json
        metas = list(latest_dir.glob("*.meta.json"))
        if metas:
            metas.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return metas[0]  # load_model() accepts meta path or model path
        raise FileNotFoundError(f"No model artifacts found under {latest_dir}")

    @staticmethod
    def _build_df(request: PredictionRequest) -> pd.DataFrame:
        d = asdict(request)
        df = pd.DataFrame([d])
        df.drop(columns=["year", "prediction_date"], errors="ignore", inplace=True)
        return df

    def _align_features(self, df: pd.DataFrame) -> None:
        """
        Reorder/add missing columns to match the model's expected feature order.
        Missing cols are created (filled with 0/False). Extra cols are dropped.
        """
        expected_features = list(self.model_feature_names())
        if not expected_features:
            # If feature_names_ not captured, assume current DF is fine.
            return

        # Add missing columns
        for expected_feature in expected_features:
            if expected_feature not in df.columns:
                # default fill for missing features (numeric 0 / False)
                df[expected_feature] = 0

        # Drop extras
        extra_features = [column for column in df.columns if column not in expected_features]
        if extra_features:
            df.drop(columns=extra_features, inplace=True)

        # Reorder to expected
        df = df.reindex(columns=expected_features, copy=False)
