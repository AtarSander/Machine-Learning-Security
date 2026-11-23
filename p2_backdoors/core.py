"""Scoring and exact-rank fusion used by the submitted solution."""

import numpy as np
from einops import rearrange


def global_score(
    raw: np.ndarray,
    transfer: np.ndarray,
    scale: float = 1.4826,
    threshold: float = 0.8,
    bonus: float = 1.0,
) -> np.ndarray:
    """Within-model MAD and positive-only global-patch transfer bonus."""
    median = np.median(raw, axis=1, keepdims=True)
    mad = np.median(np.abs(raw - median), axis=1, keepdims=True)
    if np.any(mad == 0):
        raise ValueError("Zero model MAD: the winning rule has no calibrated fallback.")
    return (raw - median) / (scale * mad) + bonus * (transfer > threshold)


def specificity_numerators(
    own: np.ndarray, total: np.ndarray, models: int
) -> np.ndarray:
    """Integer delta counts: M*own - sum(all), before max over source classes."""
    return models * own - total


def fuse(winner: np.ndarray, specificity: np.ndarray) -> np.ndarray:
    """Equal global midranks; integer ties preserved, no within-model reranking."""

    def twice_rank(values: np.ndarray) -> np.ndarray:
        flat = rearrange(values, "m t -> (m t)")
        unique, inverse, counts = np.unique(
            flat, return_inverse=True, return_counts=True
        )
        del unique
        ends = np.cumsum(counts)
        return (2 * ends - counts + 1)[inverse]

    # Twice each rank, then average two normalized ranks: denominator 4*N.
    scores = (twice_rank(winner) + twice_rank(specificity)) / (4 * winner.size)
    return rearrange(scores, "(m t) -> m t", m=winner.shape[0])
