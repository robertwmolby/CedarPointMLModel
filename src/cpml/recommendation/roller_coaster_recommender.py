import numpy as np
import pandas as pd
from keras.src.layers import IntegerLookup
import tensorflow as tf
from tensorflow.keras import layers
import numpy.typing as npt

def determine_index_for_rider_id(user_lookup: IntegerLookup, db_user_id: int) -> int:
    arr: np.ndarray  = np.array([db_user_id], dtype=np.int64)
    return int(tf.get_static_value(user_lookup(arr))[0])


def create_recommendations(model: tf.keras.Model, user_index: int,
                           n_items: int) -> npt.NDArray[np.float32]:
    item_idx: npt.NDArray[np.int32] = np.arange(n_items, dtype=np.int32)
    user_idx: npt.NDArray[np.int32] = np.full(n_items, user_index, dtype=np.int32)
    scores: npt.NDArray[np.float32] = model.predict([user_idx, item_idx], batch_size=4096, verbose=0)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return scores

def top_k_from_scores(
    scores: npt.NDArray[np.float32],
    k: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """
    Returns (topk_indices, topk_scores) sorted by descending score.
    """
    k = int(min(k, scores.size))
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    # Handle NaNs by treating them as -inf so they don't float to the top
    safe_scores = np.where(np.isnan(scores), -np.inf, scores)

    # Get top-k candidate positions (unordered)
    cand = np.argpartition(safe_scores, -k)[-k:]
    # Order those candidates by score desc
    order = np.argsort(safe_scores[cand])[::-1]
    topk_idx = cand[order].astype(np.int64)
    topk_scores = safe_scores[topk_idx].astype(np.float32)
    return topk_idx, topk_scores

def recommend_top_k(
    model: tf.keras.Model,
    user_index: int,
    n_items: int,
    k: int,
    hide_item_indices: set[int] | None = None,
) -> list[dict[str, float | int]]:
    """
    End-to-end: scores all items, optionally masks some, and returns the top-k items.
    Output uses model item indices for now (we’ll map to DB next).
    """
    scores = create_recommendations(model, user_index, n_items)

    # Optionally push excluded items to -inf (e.g., already-rated/ridden)
    if hide_item_indices:
        mask = np.isin(np.arange(n_items), list(hide_item_indices))
        scores = scores.copy()
        scores[mask] = -np.inf

    topk_idx, topk_scores = top_k_from_scores(scores, k)
    return [
        {"item_index": int(i), "score": float(s)}
        for i, s in zip(topk_idx, topk_scores)
    ]


