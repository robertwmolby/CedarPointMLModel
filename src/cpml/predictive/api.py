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
from cpml.predictive.predictor import CPPredictor
from cpml.predictive.prediction_request import PredictionRequest
from cpml.predictive.park_closed_error import ParkClosedError

DAY_NAMES = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

app = FastAPI(
    title="Cedar Point Crowd Prediction API",
    description=(
        "Predicts expected crowd level at Cedar Point for a given date and weather.\n\n"
        "- Uses trained ML models stored in `storage/artifacts`\n"
        "- Input: date (YYYY-MM-DD), temp (°F), rain (inches)\n"
        "- Output: predicted crowd level plus derived features used by the model."
    ),
    version="1.0.0",
)


# ---------- Pydantic I/O ----------
class MinimalPayload(BaseModel):
    attendance_date: date = Field(
        description="Target date (YYYY-MM-DD or parseable string)."
    )
    temp: float = Field(description="Expected high temperature in °F.")
    rain: float = Field(description="Expected rainfall in inches (0.0 = no rain).")
    model_config = {
        "json_schema_extra": {
            "example": {
                "attendance_date": "2026-08-16",
                "temp": 78.0,
                "rain": 0.0
            }
        }
    }

    @field_validator("attendance_date", mode="before")
    def validate_date(cls, attendance_date):
        try:
            if isinstance(attendance_date, date):
                return attendance_date
            return date.fromisoformat(attendance_date)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Required: YYYY-MM-DD."
            )
    @field_validator("temp")
    def validate_temp(cls, input_temp):
        try:
            fv = float(input_temp)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid temperature. Must be a number."
            )
        return fv
    @field_validator("rain")
    def validate_rain(cls, input_rain):
        try:
            fv = float(input_rain)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid rainfall amount. Must be a number."
            )
        return fv



class PredictIn(BaseModel):
    payload: Optional[MinimalPayload] = None  # also accept raw top-level

class PredictOut(BaseModel):
    prediction: float = Field(
        ...,
        description="Predicted Cedar Point crowd level (0–100 scale)."
    )

    day_details: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full preprocessed feature dictionary used by the model, "
            "including derived fields (e.g., 'month', 'holiday', 'coaster_mania')."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 45,
                "day_details": {
                    "attendance_date": "2025-08-16",
                    "temp": 75.0,
                    "rain": 0.2,
                    "month": 8,
                    "day": 16,
                    "is_weekend": True,
                    "halloweekend": False,
                    "labor_day": False,
                    "day": "Saturday"
                }
            }
        }
    }

# ---------- App state ----------
PREDICTOR: Optional[CPPredictor] = None
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
# api.py → predictive → cpml → src → project_root
def _make_predictor() -> CPPredictor:
    cp_storage_dir = os.getenv("CP_STORAGE_DIR")
    model_path = os.getenv("CP_MODEL_PATH", None)
    latest_sub = os.getenv("CP_LATEST_SUB", "artifacts/latest")
    validate_open = os.getenv("CP_VALIDATE_OPEN", "1") not in ("0", "false", "False", "")

    if cp_storage_dir:
        storage_dir = Path(cp_storage_dir)
    else:
        storage_dir = PROJECT_ROOT / "storage"

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
@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Simple uptime check used by Kubernetes probes.",
)
def health():
    return {"ok": True}


@app.get(
    "/features",
    tags=["Model Info"],
    summary="Retrieve model metadata",
    description=(
            "Returns the model path, feature names, and metrics saved during training."
    ),
)
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

@app.post(
    "/predict",
    response_model=PredictOut,
    tags=["Prediction"],
    summary="Predict Cedar Point crowd level",
    description=(
        "Predicts expected Cedar Point crowd level for the supplied date and weather.\n\n"
        "**Request format:**\n"
        "- `date`: ISO format (YYYY-MM-DD)\n"
        "- `temp`: high temperature in °F\n"
        "- `rain`: rainfall in inches\n\n"
        "The response includes the crowd prediction and all derived features used by the model.\n\n"
        "If the park will be closed on the date provided, this will return a http status 422."
    ),
)
def predict(body: PredictIn = Body(default=None)):
    if PREDICTOR is None:
        raise HTTPException(503, "Predictor not initialized")

    # Normalize minimal input
    if body and body.payload:
        minimal = body.payload.model_dump()
    else:
        raise HTTPException(403, f"Provide payload with date, temp, and rain!!!. Body: {body}")

    # Build your domain request (this derives all flags like halloweekend, holidays, etc.)
    raw_date = minimal["attendance_date"]
    if isinstance(raw_date, date):
        prediction_date = raw_date
    else:
        prediction_date = date.fromisoformat(raw_date)
    req = PredictionRequest(
        prediction_date=prediction_date,
        actual_temp=float(minimal["temp"]),
        actual_rain=float(minimal["rain"]),
    )

    # Build a DataFrame exactly like CPPredictor does (to echo what is used)
    # We’ll reuse predictor internals to avoid drift:
    try:
        # Build DF then align to model features to get the exact inputs consumed
        df = PREDICTOR._build_df(req)                 # dataclass -> 1-row DF
        PREDICTOR._align_features(df)                 # add/drop/reorder to model expectation
        feature_names = list(PREDICTOR.model_feature_names())

        # Do the prediction using the predictor path (it re-builds df, but that’s fine)
        try:
            y = PREDICTOR.predict_from_request(req)
        except ParkClosedError as pe:
            raise HTTPException(422, str(pe))
        derived_values = {k: _py(v) for k, v in asdict(req).items()}

        # Swap one-hot day of week with day name.
        for i in range(7):
            key = f"day_of_week_{i}"
            if derived_values.get(key) is True:
                day = DAY_NAMES[i]
                break
        for i in range(7):
            derived_values.pop(f"day_of_week_{i}", None)
        derived_values["day"] = day
        response: PredictOut = PredictOut(
            prediction=float(y),
            day_details=derived_values
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