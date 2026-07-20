from __future__ import annotations

import numpy as np


def _as_flat_float(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def mean_squared_error(reference, candidate) -> float:
    reference_arr = _as_flat_float(reference)
    candidate_arr = _as_flat_float(candidate)
    return float(np.mean(np.square(reference_arr - candidate_arr)))


def mean_absolute_error(reference, candidate) -> float:
    reference_arr = _as_flat_float(reference)
    candidate_arr = _as_flat_float(candidate)
    return float(np.mean(np.abs(reference_arr - candidate_arr)))


def r2_score(reference, candidate) -> float:
    reference_arr = _as_flat_float(reference)
    candidate_arr = _as_flat_float(candidate)
    residual = reference_arr - candidate_arr
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(reference_arr - np.mean(reference_arr))))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def cosine_similarity(reference, candidate) -> float:
    reference_arr = _as_flat_float(reference)
    candidate_arr = _as_flat_float(candidate)
    reference_norm = float(np.linalg.norm(reference_arr))
    candidate_norm = float(np.linalg.norm(candidate_arr))

    if reference_norm == 0.0 and candidate_norm == 0.0:
        return 1.0
    if reference_norm == 0.0 or candidate_norm == 0.0:
        return 0.0

    return float(np.dot(reference_arr, candidate_arr) / (reference_norm * candidate_norm))
