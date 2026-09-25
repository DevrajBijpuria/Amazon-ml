"""Challenge metric and threshold sweep.

The challenge scores macro-averaged F0.5 per Source-1 entity: an entity with no true
matches scores 1.0 for an empty prediction and 0.0 for any prediction. The threshold
is chosen on that metric, on validation entities only. Pooled pair-level TP/FP/FN are
reported alongside; their FN include positives that blocking never retrieved.
"""
import numpy as np
import pandas as pd

BETA2 = 0.25
EMPTY_ADDRESS_COLUMN = "target_address_missing"  # 1 when the target's normalized address is empty (address_missing_2)


def accept(scores, threshold: float, target_address_missing=None, empty_target_address_threshold=None) -> np.ndarray:
    """The single match decision used by validation metrics and by every prediction writer.

    Policy disabled (``empty_target_address_threshold`` None): score >= threshold, exactly as before.
    Policy enabled: a pair whose target has an empty normalized address needs
    score >= max(threshold, empty_target_address_threshold); every other pair keeps score >= threshold."""
    s = np.asarray(scores)
    if empty_target_address_threshold is None:
        return s >= threshold
    if target_address_missing is None:
        raise ValueError("empty-target-address policy is enabled but the target-address-missing flag was not supplied")
    strict = max(threshold, empty_target_address_threshold)
    return s >= np.where(np.asarray(target_address_missing).astype(bool), strict, threshold)


def accept_frame(scored: pd.DataFrame, threshold: float, empty_target_address_threshold=None) -> np.ndarray:
    """``accept`` on a scored frame (columns score and, when the policy is enabled, EMPTY_ADDRESS_COLUMN)."""
    flag = scored[EMPTY_ADDRESS_COLUMN].to_numpy() if empty_target_address_threshold is not None else None
    return accept(scored["score"].to_numpy(), threshold, flag, empty_target_address_threshold)


def policy_threshold(th: dict):
    """The empty-target-address threshold recorded in threshold_results.json (None = policy disabled)."""
    return (th.get("empty_target_address_policy") or {}).get("empty_target_address_threshold")


def entity_f05(n_pred: np.ndarray, n_true: np.ndarray, tp: np.ndarray) -> np.ndarray:
    """Per-entity F0.5 from counts, with the challenge's empty-vs-empty = 1.0 rule."""
    n_pred, n_true, tp = (np.asarray(a, dtype=np.float64) for a in (n_pred, n_true, tp))
    p = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    r = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    den = BETA2 * p + r
    f = np.divide((1 + BETA2) * p * r, den, out=np.zeros_like(tp), where=den > 0)
    return np.where((n_pred == 0) & (n_true == 0), 1.0, f)


def evaluate(scored: pd.DataFrame, n_true: pd.Series, threshold: float, empty_target_address_threshold=None) -> dict:
    """Metrics at one threshold (optionally under the empty-target-address policy, see ``accept``).

    scored: one row per candidate pair with columns s1 (entity id), score, label.
    n_true: ground-truth match count for EVERY evaluated S1 entity (index = entity id),
            including entities with zero matches or zero candidates.
    """
    sel = scored[accept_frame(scored, threshold, empty_target_address_threshold)]
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
