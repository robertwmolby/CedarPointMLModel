import pandas as pd
from sklearn.preprocessing import StandardScaler

INTENSITY_MAP = {
    "Kiddie": 0,
    "Family": 1,
    "Thrill": 2,
    "Extreme": 3,
}

def drop_missing_parks(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["amusement_park"].notna()]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Boolean flag for inversions
    df["has_inversions"] = df["inversion_count"] > 0

    # Numerical intensity level
    df["intensity_level"] = df["imputed_intensity"].map(INTENSITY_MAP)

    return df

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_feature_columns = [
        "imputed_length",
        "imputed_height",
        "imputed_duration",
        "imputed_speed",
    ]

    cat_columns = [
        "type",
        "design",
        "manufacturer",
        "imputed_restraints",
        "imputed_intensity",
        "imputed_duration",
        "imputed_inversion_count",
        "country",
    ]

    # One-hot encode
    cat_one_hot = pd.get_dummies(df[cat_columns])

    # Combine all features
    X = pd.concat(
        [df[numeric_feature_columns], df["intensity_level"], cat_one_hot],
        axis=1
    )

    return X

def scale_numeric_features(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame of scaled numeric features and the fitted scaler.
    """
    scaler = StandardScaler()

    scaled = scaler.fit_transform(df[numeric_cols])

    scaled_df = pd.DataFrame(
        scaled,
        columns=numeric_cols,
        index=df.index
    )

    return scaled_df, scaler

def combine_scaled_and_categorical(
    scaled_numeric: pd.DataFrame,
    full_feature_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace the numeric columns in X with their scaled versions.
    """
    # drop the original numeric columns
    cleaned = full_feature_matrix.drop(columns=scaled_numeric.columns)

    # merge scaled numeric + remaining features
    final = pd.concat([scaled_numeric, cleaned], axis=1)

    return final
