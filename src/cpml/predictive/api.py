# cpml/predictive/api.py
from __future__ import annotations

import os
import numpy as np
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse

from cpml.predictive.predictor import CPPredictor
from cpml.predictive.prediction_request import PredictionRequest
from cpml.predictive.park_closed_error import ParkClosedError


app = FastAPI(title="Cedar Point Predictor API", version="1.0.0")

# ---------- Pydantic I/O ----------
class MinimalPayload(BaseModel):
    date: str | date
    temp: float
    rain: float

    @field_validator("date", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> str:
        return v.isoformat() if isinstance(v, date) else str(v)

class PredictIn(BaseModel):
    payload: Optional[MinimalPayload] = None  # also accept raw top-level

class PredictOut(BaseModel):
    prediction: float
    model_path: str
    features_used: list[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    derived_values: Dict[str, Any] = Field(default_factory=dict)   # full asdict(PredictionRequest)

# ---------- App state ----------
PREDICTOR: Optional[CPPredictor] = None

def _make_predictor() -> CPPredictor:
    """
    Env knobs:
      CP_STORAGE_DIR   -> root storage (default: "storage")
      CP_MODEL_PATH    -> explicit path to model *or* meta file (optional)
      CP_LATEST_SUB    -> subdir under storage for "latest" (default: "artifacts/latest")
      CP_VALIDATE_OPEN -> "0"/"1" (default: "1")
    """
    storage_dir = Path(os.getenv("CP_STORAGE_DIR", "storage"))
    model_path = os.getenv("CP_MODEL_PATH", None)
    latest_sub  = os.getenv("CP_LATEST_SUB", "artifacts/latest")
    validate_open = os.getenv("CP_VALIDATE_OPEN", "1") not in ("0", "false", "False", "")

    return CPPredictor(
        storage_dir=storage_dir,
        model_path=model_path,
        latest_subpath=latest_sub,
        validate_open=validate_open,
    )

@app.on_event("startup")
def _on_startup():
    global PREDICTOR
    PREDICTOR = _make_predictor()
    # Force-load so startup fails fast if artifacts are wrong
    try:
        _ = PREDICTOR.model_path()
    except Exception as e:
        # Convert to HTTPException so uvicorn shows a clean error
        raise RuntimeError(f"Failed to load model/artifacts: {e}")

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/features")
def features():
    if PREDICTOR is None:
        raise HTTPException(503, "Predictor not initialized")
    try:
        feats = list(PREDICTOR.model_feature_names())
        mets = dict(PREDICTOR.model_metrics())
        mpath = str(PREDICTOR.model_path())
        return {
            "model_path": mpath,
            "feature_names": feats,
            "metrics": mets,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to read model metadata: {e}")

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn = Body(default=None)):
    if PREDICTOR is None:
        raise HTTPException(503, "Predictor not initialized")

    # Normalize minimal input
    if body and body.payload:
        minimal = body.payload.model_dump()
    else:
        raise HTTPException(403, f"Provide payload with date, temp, and rain!!!. Body: {body}")

    # Build your domain request (this derives all flags like halloweekend, holidays, etc.)
    req = PredictionRequest(
        prediction_date=date.fromisoformat(minimal["date"]),
        actual_temp=float(minimal["temp"]),
        actual_rain=float(minimal["rain"]),
    )

    # Build a DataFrame exactly like CPPredictor does (to echo what is used)
    # We’ll reuse predictor internals to avoid drift:
    try:
        # Build DF then align to model features to get the exact inputs consumed
        print(1)
        df = PREDICTOR._build_df(req)                 # dataclass -> 1-row DF
        print(2)
        PREDICTOR._align_features(df)                 # add/drop/reorder to model expectation
        print(3)
        feature_names = list(PREDICTOR.model_feature_names())
        print(4)
        print(5)

        # Do the prediction using the predictor path (it re-builds df, but that’s fine)
        try:
            print(6)
            y = PREDICTOR.predict_from_request(req)
            print(7)
        except ParkClosedError as pe:
            raise HTTPException(422, str(pe))
        metrics = {k: _py(v) for k, v in PREDICTOR.model_metrics().items()}
        feature_values = {n: _py(df.iloc[0][n]) for n in feature_names}
        derived_values = {k: _py(v) for k, v in asdict(req).items()}
        response: PredictOut = PredictOut(
            prediction=float(y),
            model_path=str(PREDICTOR.model_path()),
            features_used=feature_names,
            metrics=metrics,
            derived_values=derived_values
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

def _py(v):
    if isinstance(v, np.generic):  # np.int64, np.float32, etc.
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v