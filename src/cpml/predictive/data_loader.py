from pathlib import Path
import pandas as pd

def load_crowd_csv(csv_file: str | Path) -> pd.DataFrame:
    """Load the predictive model training dataset."""
    path = Path(csv_file)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")
    return pd.read_csv(path)
