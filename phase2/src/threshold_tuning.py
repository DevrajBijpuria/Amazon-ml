"""Challenge metric and threshold sweep.

The challenge scores macro-averaged F0.5 per Source-1 entity: an entity with no true
matches scores 1.0 for an empty prediction and 0.0 for any prediction. The threshold
is chosen on that metric, on validation entities only. Pooled pair-level TP/FP/FN are
reported alongside; their FN include positives that blocking never retrieved.
"""
import numpy as np
import pandas as pd

BETA2 = 0.25


def entity_f05(n_pred: np.ndarray, n_true: np.ndarray, tp: np.ndarray) -> np.ndarray:
    """Per-entity F0.5 from counts, with the challenge's empty-vs-empty = 1.0 rule."""
    n_pred, n_true, tp = (np.asarray(a, dtype=np.float64) for a in (n_pred, n_true, tp))
    p = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    r = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    den = BETA2 * p + r
    f = np.divide((1 + BETA2) * p * r, den, out=np.zeros_like(tp), where=den > 0)
    return np.where((n_pred == 0) & (n_true == 0), 1.0, f)


def evaluate(scored: pd.DataFrame, n_true: pd.Series, threshold: float) -> dict:
    """Metrics at one threshold.

    scored: one row per candidate pair with columns s1 (entity id), score, label.
    n_true: ground-truth match count for EVERY evaluated S1 entity (index = entity id),
            including entities with zero matches or zero candidates.
    """
    sel = scored[scored["score"] >= threshold]
    n_pred = sel.groupby("s1").size().reindex(n_true.index, fill_value=0).to_numpy()
    tp = sel.groupby("s1")["label"].sum().reindex(n_true.index, fill_value=0).to_numpy()
    truth = n_true.to_numpy()
    f = entity_f05(n_pred, truth, tp)
    TP, FP, FN = int(tp.sum()), int(n_pred.sum() - tp.sum()), int(truth.sum() - tp.sum())
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    zero = truth == 0
    return {
        "threshold": round(float(threshold), 4),
        "macro_f05": float(f.mean()) if len(f) else 0.0,
        "tp": TP, "fp": FP, "fn": FN,
        "precision": p, "recall": r,
        "pooled_f05": (1 + BETA2) * p * r / (BETA2 * p + r) if BETA2 * p + r else 0.0,
        "entities": int(len(truth)),
        "entities_with_prediction": int((n_pred > 0).sum()),
        "average_predictions_per_entity": float(n_pred.mean()) if len(n_pred) else 0.0,
        "zero_match_entities": int(zero.sum()),
        "zero_match_entities_scored_1": int((f[zero] == 1.0).sum()),
        "macro_f05_matched_entities": float(f[~zero].mean()) if (~zero).any() else 0.0,
    }


def sweep(scored: pd.DataFrame, n_true: pd.Series, grid: list) -> dict:
    """Evaluate every threshold; select the best macro F0.5 (ties -> higher threshold)."""
    results = [evaluate(scored, n_true, t) for t in grid]
    best = max(results, key=lambda r: (round(r["macro_f05"], 12), r["threshold"]))
    return {"selection_metric": "macro F0.5 per Source-1 entity (challenge metric)",
            "selected_threshold": best["threshold"], "selected": best,
            "selected_on_grid_edge": best["threshold"] in (results[0]["threshold"], results[-1]["threshold"]),
            "sweep": results}


def threshold_grid(cfg: dict) -> list:
    n = int(round((cfg["grid_stop"] - cfg["grid_start"]) / cfg["grid_step"])) + 1
    return [round(cfg["grid_start"] + i * cfg["grid_step"], 4) for i in range(n)]
