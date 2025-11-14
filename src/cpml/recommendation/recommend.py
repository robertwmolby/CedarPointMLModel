import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profile(rated_df: pd.DataFrame,
                       rated_features: pd.DataFrame) -> pd.Series:
    """
    Build a user preference vector as a weighted average of rated item features.
    """
    ratings = rated_df["rating"].to_numpy().reshape(-1, 1)

    # Guard against all-zero or empty ratings
    total = ratings.sum()
    if total == 0:
        raise ValueError("User has no positive/valid ratings to build a profile.")

    user_profile_vec = (rated_features.to_numpy() * ratings).sum(axis=0) / total

    return pd.Series(user_profile_vec, index=rated_features.columns)

def score_all_items(X_final: pd.DataFrame,
                    user_profile: pd.Series) -> np.ndarray:
    """
    Compute cosine similarity between all items and a user profile vector.
    """
    sims = cosine_similarity(
        X_final,
        user_profile.to_numpy().reshape(1, -1)
    ).flatten()
    return sims

import numpy as np
import pandas as pd

def recommend_for_user(
    user_ratings_df: pd.DataFrame,
    roller_coaster_df: pd.DataFrame,
    similarities: np.ndarray,
    countries: list[str],
    top_k: int = 100
) -> pd.DataFrame:
    """
    Attach similarity scores, enrich with average rating + country boosts,
    drop already-rated items, and return top-k recommendations.

    Requires:
      - roller_coaster_df: has columns [id, amusement_park, country, ...]
      - user_ratings_df: has columns [roller_coaster_id, rating, ...]
      - similarities: 1D array aligned with roller_coaster_df rows
      - add_average_ratings_and_score(df, country) defined elsewhere
    """
    # Copy base coaster data and attach similarity
    df = roller_coaster_df.copy()
    df["similarity"] = similarities

    # Enrich with avg_rating + score (country + rating boosts)
    df = add_average_ratings_and_score(df, user_countries=countries)

    # Find coasters the user has already rated
    rated_ids = set(user_ratings_df[user_ratings_df["rating"].notna() & user_ratings_df["roller_coaster_id"].notna()]["roller_coaster_id"])

    # Drop already-rated coasters
    unrated = df[~df.index.isin(rated_ids)]

    # Rank by our enriched score (not just similarity) and return top_k
    top_recs = (
        unrated.sort_values(by="score", ascending=False)
        .head(top_k)
    )

    return top_recs

def add_average_ratings_and_score(df: pd.DataFrame, user_countries: list[str]) -> pd.DataFrame:
    """
    2. Add `score` based on:
         - base: similarity
         - +40% if roller coaster is in the countries
         - then multiply by (1 + avg_rating * 0.1)
           (rating 1–5 => up to +50% boost)

       Requires df to have: similarity, country
    """

    # Base score from similarity
    score = df["similarity"].astype(float).copy()

    # 1) Country boost: +40% if same country
    country_match = df["country"].isin(user_countries)
    score *= (1 + 0.4 * country_match.astype(float))   # 1.4x where match, 1.0x otherwise

    # 2) Rating boost: multiply by (1 + rating * 0.1)
    #    rating in [1, 5] => factor in [1.1, 1.5]
    score *= (1 + df["average_rating"].astype(float) * 0.1)

    df["score"] = score

    return df

