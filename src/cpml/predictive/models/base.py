# cpml/predictive/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import json, time, sys, importlib
import pandas as pd

try:
    import joblib  # only needed for "joblib" serializer
except Exception:
    joblib = None


def _lib_versions() -> Dict[str, str]:
    vers = {"python": sys.version.split()[0]}
    for lib in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "tensorflow"):
        try:
            vers[lib] = importlib.import_module(lib).__version__
        except Exception:
            pass
    return vers


@dataclass
class ModelMeta:
    model_class: str
    model_name: str
    created_at: float
    feature_names: list[str]
    metrics: Dict[str, float]
    serializer: str                 # "joblib" | "xgboost_json" | "lightgbm_txt" | "keras_dir"
    lib_versions: Dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_path(p: Path) -> "ModelMeta":
        return ModelMeta(**json.loads(p.read_text()))


class CPModel(ABC):
    """
    Backward compatible base:
      - Keep rmse/r_squared properties & create_model(df)/predict(df)->int
      - Add fit(X,y), evaluate(X,y), metrics, feature_names_, save/load with meta
    Subclasses:
      - set self.model (native estimator) after train
      - set self.feature_names_ (list[str])
      - may override DEFAULT_SERIALIZER
    """
    DEFAULT_SERIALIZER: str = "joblib"  # subclasses may override
    CLASS_NAME: str = "CPModel"

    # If your subclass wants to accept an xgboost Booster directly, set this True.
    ACCEPTS_BOOSTER: bool = False

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model: Any = None
        self._rmse: Optional[float] = None
        self._r2: Optional[float] = None
        self.metrics: Dict[str, float] = {}
        self.feature_names_: list[str] = []
        self.model_name: str = model_name or self.__class__.__name__

    # --- training API ---------------------------------------------------------
    @abstractmethod
    def create_model(self, crowd_level_df: pd.DataFrame):
        """
        Legacy training entry. Subclasses likely implement this already.
        Should set: self.model, self.feature_names_, self._rmse/_r2 or self.metrics.
        """
        raise NotImplementedError

    def set_metrics(self, **new_metrics: float) -> None:
        """
        Add or update metric values in the metrics dictionary.
        """
        if not hasattr(self, "metrics") or self.metrics is None:
            self.metrics = {}

        for key, value in new_metrics.items():
            self.metrics[key] = value

    # New preferred fit; default adapter calls create_model if you haven’t migrated
    def fit(self, X, y=None):
        """
        Preferred training entry: set self.model, self.feature_names_.
        If not overridden, falls back to legacy create_model(pd.DataFrame).
        """
        if isinstance(X, pd.DataFrame) and y is None:
            # Old style: single DF with features+label; delegate
            return self.create_model(X)
        else:
            # If a subclass overrides fit, it will replace this.
            raise NotImplementedError(
                f"{self.__class__.__name__}.fit(X,y) not implemented; "
                f"either implement fit or call create_model(df)."
            )

    @abstractmethod
    def predict(self, prediction_df: pd.DataFrame) -> int:
        """
        Keep your current signature for now.
        For batch inference later, you can add predict_many(X)->np.ndarray in subclass.
        """
        raise NotImplementedError

    # --- persistence ----------------------------------------------------------
    def save_model(self, model_file_name: str):
        """
        Save native model + sidecar meta json.
        model_file_name can be a directory or a full stem path (without extension).
        We will append the proper extension based on serializer.
        """
        out = Path(model_file_name)
        out.parent.mkdir(parents=True, exist_ok=True)

        # If 'model_file_name' is a directory, synthesize a timestamped stem
        if out.suffix == "" and out.is_dir():
            ts = time.strftime("%Y%m%d-%H%M%S")
            stem = out / f"{self.model_name}-{ts}"
        elif out.suffix == "":
            stem = out
        else:
            # strip extension if someone passed one
            stem = out.with_suffix("")

        serializer = getattr(self, "DEFAULT_SERIALIZER", "joblib")
        model_path = self._save_native_model(stem, serializer)

        # Guards so a half-trained model still writes meta without crashing
        feature_names = getattr(self, "feature_names_", []) or []
        metrics = (getattr(self, "metrics", None) or {}).copy()

        meta = ModelMeta(
            model_class=self.__class__.__name__,
            model_name=self.model_name,
            created_at=time.time(),
            feature_names=feature_names,
            metrics=metrics,
            serializer=serializer,
            lib_versions=_lib_versions(),
        )
        meta_path = stem.with_suffix(".meta.json")
        meta_path.write_text(meta.to_json())
        return model_path, meta_path

    @classmethod
    def load_model(cls, meta_path: str | Path) -> "CPModel":
        """
        Generic loader using the *.meta.json file. Returns an instance of the correct subclass.
        """
        meta_p = Path(meta_path)
        meta = ModelMeta.from_path(meta_p)
        subcls = _resolve_model_class(meta.model_class)
        inst: CPModel = subcls(model_name=meta.model_name)  # type: ignore[call-arg]
        inst.feature_names_ = meta.feature_names
        inst.metrics = meta.metrics
        if "rmse" in meta.metrics:
            inst._rmse = float(meta.metrics["rmse"])
        if "r2" in meta.metrics:
            inst._r2 = float(meta.metrics["r2"])
        inst.model = _load_native_model(meta_p, meta.serializer, accepts_booster=getattr(inst, "ACCEPTS_BOOSTER", False))
        return inst

    # --- helpers --------------------------------------------------------------
    def _save_native_model(self, stem: Path, serializer: str) -> Path:
        if serializer == "joblib":
            if joblib is None:
                raise RuntimeError("joblib is required for 'joblib' serializer")
            path = stem.with_suffix(".bin")
            joblib.dump(self.model, path)
            return path

        if serializer == "xgboost_json":
            import xgboost as xgb
            path = stem.with_suffix(".json")
            if hasattr(self.model, "save_model"):
                self.model.save_model(str(path))       # XGBRegressor or Booster
            else:
                # raw Booster expected
                assert isinstance(self.model, xgb.Booster)
                self.model.save_model(str(path))
            return path

        if serializer == "lightgbm_txt":
            path = stem.with_suffix(".txt")
            # handle LGBMRegressor or Booster
            booster = getattr(self.model, "booster_", None)
            if booster is None:
                booster = self.model  # assume it's already a Booster
            booster.save_model(str(path))
            return path

        if serializer == "keras_dir":
            path = stem.with_suffix(".keras")
            self.model.save(str(path))
            return path

        raise ValueError(f"Unknown serializer: {serializer}")


# -------- module-level helpers (kept out of class for clarity) ----------------

def _resolve_model_class(class_name: str):
    """
    Greedy import over known modules to find the subclass symbol by name.
    Add new files here as you create them.
    """
    candidates = [
        "cpml.predictive.models.linear",
        "cpml.predictive.models.polylinear",
        "cpml.predictive.models.random_forest",
        "cpml.predictive.models.neural_network",
        "cpml.predictive.models.gradient_boost",
        "cpml.predictive.models.xgboost_model",
        "cpml.predictive.models.lightgbm",
    ]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, class_name):
                return getattr(mod, class_name)
        except Exception:
            pass
    raise ImportError(f"Could not resolve model class {class_name}")


def _load_native_model(meta_file: Path, serializer: str, *, accepts_booster: bool) -> Any:
    stem = meta_file.with_suffix("").with_suffix("")  # drop .meta then get stem

    if serializer == "joblib":
        if joblib is None:
            raise RuntimeError("joblib is required for 'joblib' serializer")
        return joblib.load(stem.with_suffix(".bin"))

    if serializer == "xgboost_json":
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(stem.with_suffix(".json")))
        if accepts_booster:
            return booster
        # Wrap in regressor for familiar API
        reg = xgb.XGBRegressor()
        reg._Booster = booster
        return reg

    if serializer == "lightgbm_txt":
        import lightgbm as lgb
        # Return a Booster; many users call predict() on the sklearn wrapper too,
        # but Booster.predict() works for inference.
        return lgb.Booster(model_file=str(stem.with_suffix(".txt")))

    if serializer == "keras_dir":
        import tensorflow as tf
        return tf.keras.models.load_model(str(stem.with_suffix(".keras")))

    raise ValueError(f"Unknown serializer: {serializer}")
