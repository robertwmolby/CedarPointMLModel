from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def feature_importance(
    df: pd.DataFrame,
    label: str = "crowd_level",
    drop_cols: tuple[str, ...] = ("date",),
    n_estimators: int = 500,
    random_state: int = 0,
) -> pd.Series:
    """
    Compute feature importances for a label using a RandomForestRegressor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and the target label.
    label : str, default="crowd_level"
        Name of the target column.
    drop_cols : tuple[str, ...], default=("date",)
        Columns to drop before training.
    n_estimators : int, default=500
        Number of trees in the random forest.
    random_state : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Sorted feature importances (0–1) indexed by feature name.
    """
    # Drop rows missing the label
    Xy = df.dropna(subset=[label]).copy()

    # Separate target
    y = Xy.pop(label)

    # Drop non-feature columns if present
    for col in drop_cols:
        if col in Xy.columns:
            Xy.drop(columns=[col], inplace=True)

    # Keep only numeric features (avoid object/categorical issues)
    X = Xy.select_dtypes(include="number")

    if X.empty:
        raise ValueError("No numeric features available for importance calculation.")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)

    return pd.Series(
        model.feature_importances_, index=X.columns, name="importance"
    ).sort_values(ascending=False)
