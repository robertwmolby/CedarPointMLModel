from typing import List, Optional
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from cpml.recommendation.recommendation_engine import RecommendationEngine


# ---------- Configuration / Engine ----------

COASTER_TABLE = os.getenv("COASTER_TABLE", "roller_coasters")
engine = RecommendationEngine(coaster_table=COASTER_TABLE)

app = FastAPI(
    title="Coaster Recommendation API",
    version="1.0.0",
    description=(
        "Content-based recommendation service for roller coasters.\n\n"
        "This API exposes endpoints for:\n"
        "- Getting personalized coaster recommendations based on a user's ratings\n"
        "- Finding coasters similar to a given coaster\n"
        "- Basic health checks for orchestration / monitoring\n\n"
        "Use the `/docs` path for an interactive Swagger UI, or `/redoc` for "
        "a ReDoc-based documentation view."
    ),
)


# ---------- Request / Response Models ----------


class UserRating(BaseModel):
    """
    Single user rating for a particular coaster.

    This model is used inside `UserRecommendRequest` to describe the
    user's historical preferences.

    Example
    -------
    ```json
    {
      "coaster_id": 594,
      "rating": 4.5
    }
    ```
    """

    coaster_id: int = Field(
        ...,
        description="Unique identifier of the roller coaster that was rated.",
        example=594,
    )
    rating: float = Field(
        ...,
        ge=0.0,
        le=5.0,
        description="User's rating for the coaster on a 1–5 scale.",
        example=4.5,
    )

    class Config:
        # Pydantic model configuration with an example payload
        schema_extra = {
            "example": {
                "coaster_id": 594,
                "rating": 4.5,
            }
        }


class UserRecommendRequest(BaseModel):
    """
    Request body for user-based recommendations.

    The engine uses:
    - `countries` to filter out coasters in regions the user cannot / will not visit.
    - `ratings` as the user's explicit preference signals.
    - `top_k` to limit the number of recommendations returned.

    Example
    -------
    ```json
    {
      "countries": ["United States", "Canada"],
      "ratings": [
        { "coaster_id": 594, "rating": 4.5 },
        { "coaster_id": 6992, "rating": 4.5 },
        { "coaster_id": 18, "rating": 1.5 }
      ],
      "top_k": 10
    }
    ```
    """

    countries: List[str] = Field(
        ...,
        description=(
            "List of country names the user has access to (e.g., where they might "
            "realistically visit parks). Coasters outside these countries are filtered out."
        ),
        example=["United States", "Canada"],
    )
    ratings: List[UserRating] = Field(
        ...,
        min_items=1,
        description=(
            "List of coasters the user has rated. At least one rating is required "
            "to compute personalized recommendations."
        ),
    )
    top_k: Optional[int] = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of recommendations to return. Defaults to 10.",
        example=15,
    )

    class Config:
        schema_extra = {
            "example": {
                "countries": ["United States", "Canada"],
                "ratings": [
                    {"coaster_id": 594, "rating": 4.5},
                    {"coaster_id": 6992, "rating": 4.5},
                    {"coaster_id": 18, "rating": 1.5}
                ],
                "top_k": 10,
            }
        }


class HealthResponse(BaseModel):
    """Simple response model for the `/health` endpoint."""

    status: str = Field(
        ...,
        description="Health status indicator. `ok` means the API process is running.",
        example="ok",
    )


class CoasterRecommendation(BaseModel):
    """
    Generic coaster recommendation result.

    The engine returns a DataFrame that always contains an ID column plus
    a set of metadata / score columns (e.g., name, park, country, score).

    This model guarantees at least `coaster_id` is present, while allowing
    additional dynamic fields from the engine.
    """

    coaster_id: int = Field(
        ...,
        description="Unique identifier of the recommended roller coaster.",
        example=789,
    )

    class Config:
        # Allow arbitrary additional fields returned from the api
        extra = "allow"
        schema_extra = {
            "example": {
                "coaster_id": 3570,
                "name": "Maverick",
                "amusement_park": "Cedar Point",
                "type": "Steel",
                "design": "Sit Down",
                "status": "Operating",
                "manufacturer": "Intamin Amusement Rides",
                "model": "Custom",
                "length": 4450,
                "height": 105,
                "drop": 100,
                "inversion_count": 2,
                "speed": 70,
                "vertical_angle": 95,
                "duration": 150,
                "restraints": "null",
                "g_force": "null",
                "intensity": "Extreme",
                "country": "United States",
                "average_rating": 4.79,
                "intensity_level": 3,
                "score": 1.93
            }
        }


# ---------- Helper ----------


def df_with_id(df: pd.DataFrame) -> List[dict]:
    """
    Convert a DataFrame of coaster data into a cleaned, JSON-serializable list.

    Steps:
    - Reset index to expose the coaster ID as `coaster_id`
    - Drop unwanted fields
    - Drop ALL columns starting with `imputed_`
    - Convert NaN / +/- inf to None
    - Round score to 2 decimals
    - Cast selected numeric fields to integers
    """

    # Normalize index to become `coaster_id`
    if df.index.name is None:
        df_reset = df.reset_index().rename(columns={"index": "coaster_id"})
    else:
        df_reset = df.reset_index().rename(columns={df.index.name: "coaster_id"})

    # ------------------------------------------------------------------
    # 1. Drop explicit unwanted fields
    # ------------------------------------------------------------------
    EXCLUDE_COLS = {
        "url",
        "status",
        "exclude_record",
        "standardized_coaster_name",
        "standardized_park_name",
        "similarity_item",
        "similarity",
    }

    df_reset = df_reset.drop(
        columns=[c for c in EXCLUDE_COLS if c in df_reset.columns],
        errors="ignore",
    )

    # ------------------------------------------------------------------
    # 2. Drop ALL imputed_* columns (your request)
    # ------------------------------------------------------------------
    imputed_cols = [c for c in df_reset.columns if c.startswith("imputed_")]
    if imputed_cols:
        df_reset = df_reset.drop(columns=imputed_cols, errors="ignore")

    # ------------------------------------------------------------------
    # 3. Cast selected fields to integers
    # ------------------------------------------------------------------
    INT_FIELDS = ["coaster_id", "height", "inversion_count", "speed"]
    for col in INT_FIELDS:
        if col in df_reset.columns:
            df_reset[col] = (
                pd.to_numeric(df_reset[col], errors="coerce")  # ensure numeric/NaN
                .round(0)  # remove fraction
                .astype("Int64")  # nullable int
            )

    # ------------------------------------------------------------------
    # 4. Round score to 2 decimals
    # ------------------------------------------------------------------
    if "score" in df_reset.columns:
        df_reset["score"] = df_reset["score"].round(2)

    # ------------------------------------------------------------------
    # 5. Convert NaN / inf to None for JSON
    # ------------------------------------------------------------------
    df_reset = df_reset.replace([np.nan, np.inf, -np.inf], None)

    return df_reset.to_dict(orient="records")


# ---------- Endpoints ----------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health() -> HealthResponse:
    """
    Lightweight liveness probe.

    Used by Kubernetes / orchestrators / load balancers to verify that the
    API process is up and responding. It does **not** perform a deep dependency
    check (e.g., database connectivity); that should be handled by more
    specific diagnostics if needed.
    """
    return HealthResponse(status="ok")


@app.post(
    "/recommendations/user",
    response_model=List[CoasterRecommendation],
    summary="Get personalized coaster recommendations for a user",
    tags=["Recommendations"],
)
def recommend_for_user(request: UserRecommendRequest) -> List[CoasterRecommendation]:
    """
    Compute user-personalized roller coaster recommendations.

    Description
    -----------
    This endpoint takes:
    - A list of **countries** where the user can realistically access parks.
    - A set of **ratings** that describe the user's preferences.
    - An optional **top_k** to limit the number of items returned.

    The underlying recommendation engine:
    - Builds a user preference profile from the ratings.
    - Scores all coasters available in the specified countries.
    - Returns the top `top_k` coasters, sorted by descending recommendation score.

    Recommendations take into account general similarity as well as average
    user rating of the coasters where available and give a strong preference to
    roller coasters that are in countries where the user can get to based on
    the country provided without encountering geographical or issues posed by
    governmental restrictions.

    Parameters
    ----------
    request:
        `UserRecommendRequest` body containing countries, user ratings,
        and optional `top_k` limit.

    Returns
    -------
    List[CoasterRecommendation]
        List of recommended coasters with `coaster_id` and metadata columns
        produced by the engine (e.g., name, park, score).
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

    return df_with_id(recs)  # FastAPI will coerce to List[CoasterRecommendation]


@app.get(
    "/recommendations/similar/{coaster_id}",
    response_model=List[CoasterRecommendation],
    summary="Get coasters similar to a given coaster",
    tags=["Recommendations"],
)
def recommend_similar(
    coaster_id: int = Path(
        ...,
        description="Identifier of the coaster you want similar recommendations for.",
        example=3570,
    ),
    top_k: int = Query(
        20,
        ge=1,
        le=100,
        description="Maximum number of similar coasters to return. Defaults to 20.",
        example=10,
    ),
) -> List[CoasterRecommendation]:
    """
    Retrieve coasters that are similar to a given coaster.

    FYI - coaster id given in example is Maverick at Cedar Point

    Description
    -----------
    This endpoint:
    - Looks up the specified `coaster_id`.
    - Uses the precomputed feature space in the recommendation engine.
    - Returns the top `top_k` most similar coasters.

    Parameters
    ----------
    coaster_id:
        ID of the base coaster from which to compute similarity.
    top_k:
        Maximum number of similar coasters to return.

    Returns
    -------
    List[CoasterRecommendation]
        List of similar coasters, starting with the most similar first.
    """
    try:
        recs = engine.recommend_similar_to_coaster(
            coaster_id=coaster_id,
            top_k=top_k,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return df_with_id(recs)
