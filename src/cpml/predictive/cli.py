from pathlib import Path
import typer
import pandas as pd
from cpml.predictive.models.base import CPModel
import re, json

app = typer.Typer()

def _is_meta_json(p: Path) -> bool:
    if not (p.is_file() and p.suffix == ".json"):
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False

    has_features = ("feature_names" in d) or ("features" in d)
    has_model_id = any(k in d for k in ("model", "model_class", "model_name"))
    has_metrics = isinstance(d.get("metrics"), dict)

    return has_features and has_model_id and has_metrics

def _resolve_meta(path: Path, debug: bool = False) -> Path:
    if path.is_dir():
        # Look for any file ending in .meta.json
        for cand in path.glob("*.meta.json"):
            print(cand)
            if _is_meta_json(cand):
                return cand

        # (optional) fallback: check top-level JSONs that look like meta
        for cand in path.glob("*.json"):
            print(cand)
            if _is_meta_json(cand):
                return cand

        raise FileNotFoundError(f"No .meta.json file found under directory: {path}")

    if path.exists() and path.is_file():
        if _is_meta_json(path):
            return path
        raise FileNotFoundError(f"Not a valid meta JSON: {path}")

    raise FileNotFoundError(f"No such file or directory: {path}")

@app.command()
def predict(meta_path: str, json_payload: str, debug: bool = False):
    meta_file = _resolve_meta(Path(meta_path), debug)
    if debug:
        print("RAW:", repr(json_payload))
    payload = _parse_payload_loose(json_payload)
    if debug:
        print("PARSED:", payload)
    m = CPModel.load_model(meta_file)
    row = {k: payload.get(k, None) for k in m.feature_names_}
    df = pd.DataFrame([row])
    print(int(m.predict(df)))


@app.command()
def features(meta_path: str):
    m = CPModel.load_model(_resolve_meta(Path(meta_path)))
    print(json.dumps({
        "model": m.__class__.__name__,
        "features": m.feature_names_,
        "metrics": m.metrics,
    }, indent=2))



def _parse_payload_loose(s: str) -> dict:
    """
    Accepts JSON, JSON-ish (unquoted keys), single quotes, trailing commas.
    Converts to strict JSON and returns a dict.
    """
    s = s.strip()

    # If PowerShell wrapped the whole thing in quotes, strip once.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"') and "{" in s:
        s = s[1:-1].strip()

    # First try strict JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Normalize single quotes (strings) → double quotes
    s = re.sub(r"'", '"', s)

    # Quote unquoted object keys: { month: 8, day: 15 } → { "month": 8, "day": 15 }
    s = re.sub(r'(?P<prefix>[{\s,])(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:',
               r'\g<prefix>"\g<key>":', s)

    # Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # Convert JS-style literals if they ever show up
    s = re.sub(r'\btrue\b', 'true', s, flags=re.IGNORECASE)
    s = re.sub(r'\bfalse\b', 'false', s, flags=re.IGNORECASE)
    s = re.sub(r'\bnull\b', 'null', s, flags=re.IGNORECASE)

    return json.loads(s)

if __name__ == "__main__":
    app()
