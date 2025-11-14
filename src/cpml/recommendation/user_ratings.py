import pandas as pd


def build_rated_features(
    user_ratings_df: pd.DataFrame,
    roller_coaster_df: pd.DataFrame,
    X_final: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter to rated coasters, join with coaster metadata, and
    extract the corresponding feature rows from X_final.

    Returns:
        rated_df: ratings joined with coaster metadata
        rated_features: feature matrix for rated coasters (same row order)
    """
    # Step 1: Filter out unrated coasters
    rated_only_df = user_ratings_df[user_ratings_df["rating"].notnull()]

    # Step 2: Merge with coaster metadata
    rated_df = rated_only_df.merge(
        roller_coaster_df.reset_index(),  # 'id' comes from index
        left_on="roller_coaster_id",
        right_on="id",
    )

    # shift user ratings so that negative ratings push a user "away" from similar
    # coasters rather than providing a slight push towards them.
    user_ratings_df["rating"] -= 3

    # Step 3: Feature rows for those coasters
    rated_features = X_final.loc[rated_df["roller_coaster_id"]]

    return rated_df, rated_features
