# recommendation_engine.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from cpml.recommendation.data_access import load_table
from cpml.recommendation.preprocess import (
    drop_missing_parks,
    add_engineered_features,
    build_feature_matrix,
    scale_numeric_features,
    combine_scaled_and_categorical,
)
from cpml.recommendation.user_ratings import build_rated_features
from cpml.recommendation.recommend import (
    build_user_profile,
    score_all_items,
    recommend_for_user as _recommend_for_user,
)

NUMERIC_FEATURE_COLUMNS = [
    "imputed_length",
    "imputed_height",
    "imputed_duration",
    "imputed_speed",
]


class RecommendationEngine:
    def __init__(self, coaster_table: str):
        """
        Build all coaster-side data once at startup.
        """
        # 1) Load coaster metadata
        df = load_table(coaster_table)

        # Make sure id is the index
        df = df.set_index("id")

        # 2) Basic cleaning / filtering
        df = drop_missing_parks(df)

        # 3) Add engineered features (has_inversions, intensity_level, etc.)
        df = add_engineered_features(df)

        # 4) Build full feature matrix (numeric + one-hot cats)
        X = build_feature_matrix(df)

        # 5) Scale numeric features, then combine back
        X_numeric_scaled, scaler = scale_numeric_features(df, NUMERIC_FEATURE_COLUMNS)
        X_final = combine_scaled_and_categorical(X_numeric_scaled, X)

        self.coaster_df = df
        self.X_final = X_final
        self.scaler = scaler

    def recommend_for_user(
        self,
        user_countries: list[str],
        user_ratings_df: pd.DataFrame,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """
        Main entry point:
        - Takes a DataFrame of user ratings (roller_coaster_id, rating, ...)
        - Returns top-k recommended coasters the user hasn't rated.
        """

        # 1) Join user ratings with coaster metadata and features
        rated_df, rated_features = build_rated_features(
            user_ratings_df=user_ratings_df,
            roller_coaster_df=self.coaster_df,
            X_final=self.X_final,
        )

        if rated_df.empty:
            raise ValueError("User has no rated coasters to base recommendations on.")

        # 2) Build user preference vector
        user_profile = build_user_profile(rated_df, rated_features)

        # 3) Score all coasters vs the user profile
        similarities = score_all_items(self.X_final, user_profile)
        # 4) Filter out already-rated and return top-k
        top_recs = _recommend_for_user(
            user_ratings_df=user_ratings_df,
            roller_coaster_df=self.coaster_df,
            countries=user_countries,
            similarities=similarities,
            top_k=top_k,
        )

        return top_recs

    def recommend_similar_to_coaster(self, coaster_id: int, top_k: int = 20) -> pd.DataFrame:
        """
        Item-item recommendations: given a coaster_id, return most similar coasters
        regardless of user ratings.
        """
        if coaster_id not in self.coaster_df.index:
            raise KeyError(f"Coaster id {coaster_id} not found.")

        idx = self.coaster_df.index.get_loc(coaster_id)
        target_vec = self.X_final.iloc[[idx]]

        sims = cosine_similarity(target_vec, self.X_final)[0]
        self.coaster_df["similarity_item"] = sims

        # drop itself
        recs = (
            self.coaster_df[self.coaster_df.index != coaster_id]
            .sort_values(by="similarity_item", ascending=False)
            .head(top_k)
        )

        return recs[
            ["name", "similarity_item", "type", "design", "imputed_intensity", "amusement_park"]
        ]
