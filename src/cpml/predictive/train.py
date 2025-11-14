# cpml/predictive/train.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import time
import json
import shutil
import argparse
import pandas as pd

from cpml.predictive.preprocess import prepare_for_modeling
from cpml.predictive.models import MODEL_CLASSES
from cpml.common.logging import get_logger

logger = get_logger("predictive.train")

def train_models(
    df: pd.DataFrame,
    storage_dir: Path | str,
    run_tag: str | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """
    Trains every class in MODEL_CLASSES, saves each model under artifacts/run_<tag>/<ModelName>,
    returns (results_df, best_model_path).
    """
    storage_dir = Path(storage_dir)
    run_tag = run_tag or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_root = storage_dir / "artifacts" / f"run_{run_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    cp_model_df = prepare_for_modeling(df)

    # ---- train loop ----
    results: list[dict] = []
    for ModelCls in MODEL_CLASSES:
        name = ModelCls.__name__
        try:
            m = ModelCls()
            t0 = time.perf_counter()
            m.create_model(cp_model_df.copy())  # must set m.feature_names_ and m.metrics
            fit_s = time.perf_counter() - t0

            model_dir = out_root / name
            model_dir.mkdir(parents=True, exist_ok=True)
            m.save_model(model_dir)

            r2   = (m.metrics or {}).get("r2")
            rmse = (m.metrics or {}).get("rmse")

            results.append({
                "Model": name,
                "Serializer": getattr(m, "DEFAULT_SERIALIZER", None),
                "R2": r2,
                "RMSE": rmse,
                "FitSeconds": round(fit_s, 3),
                "Path": str(model_dir),
                "Features": ",".join(m.feature_names_ or []),
            })
            logger.info("Trained %-28s R2=%s RMSE=%s time=%0.2fs -> %s",
                        name,
                        f"{r2:.4f}" if r2 is not None else "NA",
                        f"{rmse:.3f}" if rmse is not None else "NA",
                        fit_s, model_dir)
        except Exception as e:
            logger.warning("Model %s failed: %s", name, e)
            results.append({
                "Model": name, "Serializer": None, "R2": None, "RMSE": None,
                "FitSeconds": None, "Path": None, "Features": None, "Error": str(e)
            })

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    (out_root / "results.json").write_text(json.dumps(results, indent=2))
    results_df.to_csv(out_root / "results.csv", index=False)

    # ---- pick best and update 'latest' pointer ----
    best_path = None
    best = results_df.dropna(subset=["R2"]).sort_values(by="R2", ascending=False).head(1)
    if not best.empty:
        best_path = Path(best.iloc[0]["Path"])
        latest = out_root.parent / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(best_path, latest)
        logger.info("Copied best model to %s", latest)
    return results_df, best_path

def _cli() -> None:
    p = argparse.ArgumentParser(description="Train all predictive models.")
    p.add_argument("--input-csv", help="Optional: load df from CSV (expects columns used by prepare_for_modeling).")
    p.add_argument("--storage-dir", required=True, help="Root directory for artifacts (was STORAGE_DIR in the notebook).")
    p.add_argument("--run-tag", help="Optional run tag like 20251113T205940Z.")
    args = p.parse_args()

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        raise SystemExit("--input-csv is required when running as a script.")

    results_df, best = train_models(df, args.storage_dir, args.run_tag)
    print(results_df.head(20).to_string(index=False))
    if best:
        print("latest ->", best)

if __name__ == "__main__":
    _cli()
