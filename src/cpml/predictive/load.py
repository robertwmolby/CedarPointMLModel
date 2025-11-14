# cpml/predictive/load.py
from pathlib import Path
import json
from cpml.predictive.models import MODEL_REGISTRY

def load_model(model_path: str | Path):
    """
    Generic loader that figures out which CPModel subclass to use
    based on the meta.json and calls its classmethod load_model().
    """
    model_path = Path(model_path)
    # Determine meta.json path
    meta_path = model_path
    if meta_path.is_dir():
        meta_path = meta_path / "model.meta.json"
    elif not str(meta_path).endswith(".meta.json"):
        meta_path = meta_path.with_suffix(meta_path.suffix + ".meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta.get("model_name")
    if not model_name:
        raise ValueError(f"No model_name in meta.json at {meta_path}")

    # Look up appropriate subclass
    ModelCls = MODEL_REGISTRY.get(model_name)
    if ModelCls is None:
        raise ValueError(f"Unknown model type '{model_name}' in {meta_path}")

    # Delegate to its classmethod
    return ModelCls.load_model(str(model_path))
