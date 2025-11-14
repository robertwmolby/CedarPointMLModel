# api.py

from typing import List, Optional
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommendation_engine import RecommendationEngine
import numpy as np


COASTER_TABLE = os.getenv("COASTER_TABLE", "roller_coasters")
engine = RecommendationEngine(coaster_table=COASTER_TABLE)

app = FastAPI(title="Coaster Recommendation API")


# ---------- Request / Response Models ----------
class UserRating(BaseModel):
    coaster_id: int
    rating: float


class UserRecommendRequest(BaseModel):
    countries: List[str]
    ratings: List[UserRating]
    top_k: Optional[int] = 20


# ---------- Helper ----------

def df_with_id(df: pd.DataFrame) -> list[dict]:
    """
    Reset index so the coaster id is returned as `coaster_id`,
    and convert NaN/inf to None so JSON can handle it.
    """
    if df.index.name is None:
        df_reset = df.reset_index().rename(columns={"index": "coaster_id"})
    else:
        df_reset = df.reset_index().rename(columns={df.index.name: "coaster_id"})

    # Replace NaN and +/- inf with None for JSON compatibility
    df_reset = df_reset.replace([np.nan, np.inf, -np.inf], None)

    return df_reset.to_dict(orient="records")


# ---------- Endpoints ----------

@app.post("/recommendations/user")
def recommend_for_user(request: UserRecommendRequest):
    """
    POST:
    - Body:
        {
          "countries": ["United States", "Canada"],
          "ratings": [
            {"coaster_id": 123, "rating": 4.5},
            {"coaster_id": 456, "rating": 3.0}
          ],
          "top_k": 25   # optional
        }
    - Uses RecommendationEngine.recommend_for_user
    """

    if not request.ratings:
        raise HTTPException(status_code=400, detail="At least one rating is required.")

    # Build user_ratings_df expected by the engine
    user_ratings_df = pd.DataFrame(
        [
            {"roller_coaster_id": r.coaster_id, "rating": r.rating}
            for r in request.ratings
        ]
    )

    try:
        recs = engine.recommend_for_user(
            user_countries=request.countries,
            user_ratings_df=user_ratings_df,
            top_k=request.top_k or 20,
        )
    except ValueError as e:
        # e.g., no valid rated coasters
        raise HTTPException(status_code=400, detail=str(e))

    return df_with_id(recs)


@app.get("/recommendations/similar/{coaster_id}")
def recommend_similar(coaster_id: int, top_k: int = 20):
    """
    GET:
      /recommendations/similar/{coaster_id}?top_k=10

    - Calls RecommendationEngine.recommend_similar_to_coaster
    """

    try:
        recs = engine.recommend_similar_to_coaster(
            coaster_id=coaster_id,
            top_k=top_k,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return df_with_id(recs)
