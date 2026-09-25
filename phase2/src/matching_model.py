"""Labels, entity-level split, controlled negative sampling and the logistic-regression matcher."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def label_pairs(cands: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    """1 where (s1_idx, t_idx) is a ground-truth match, else 0 (positional indexes)."""
    merged = cands[["s1_idx", "t_idx"]].merge(positives[["s1_idx", "t_idx"]].assign(_pos=1),
                                              how="left", on=["s1_idx", "t_idx"])
    return merged["_pos"].fillna(0).to_numpy(np.int8)


def split_entities(s1_ids: list, validation_fraction: float, seed: int) -> set:
    """Entity-level split: returns the Source-1 ids held out for validation."""
    ids = np.array(sorted(s1_ids))
    rng = np.random.default_rng(seed)
    n_val = int(round(len(ids) * validation_fraction))
    return set(ids[rng.permutation(len(ids))[:n_val]].tolist())


def sample_training_pairs(labels: np.ndarray, feats: pd.DataFrame, negative_ratio: float,
                          hard_fraction: float, seed: int) -> np.ndarray:
    """Row positions to train on: every positive, plus ``negative_ratio`` negatives per positive,
    ``hard_fraction`` of them the most name+address-similar non-matches, the rest uniform random."""
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    n_neg = min(len(neg), int(round(negative_ratio * len(pos))))
    n_hard = int(round(n_neg * hard_fraction))
    hardness = (feats["name_char3_jaccard"].to_numpy() + feats["address_char3_jaccard"].to_numpy())[neg]
    order = np.lexsort((neg, -hardness))  # most similar first, ties by position
    hard = neg[order[:n_hard]]
    rest = np.setdiff1d(neg, hard)
    rng = np.random.default_rng(seed)
    easy = rng.choice(rest, size=min(len(rest), n_neg - n_hard), replace=False) if n_neg > n_hard else rest[:0]
    return np.sort(np.concatenate([pos, hard, easy]))


def train_model(X: pd.DataFrame, y: np.ndarray, model_cfg: dict, seed: int):
    """StandardScaler + LogisticRegression (B0), or HistGradientBoostingClassifier (Phase 2.5 M1)."""
    if model_cfg.get("type") == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=seed, early_stopping=False,
                                               max_iter=model_cfg["max_iter"])
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=model_cfg["C"], max_iter=model_cfg["max_iter"], random_state=seed))
    return model.fit(X.to_numpy(np.float64), y)


def pair_metrics(y: np.ndarray, pred: np.ndarray, beta: float = 0.5) -> dict:
    """Pair-level confusion matrix, precision, recall and F-beta."""
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if b2 * p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f05": f}


def coefficients(model, features: list) -> dict:
    """Standardized logistic-regression coefficients by feature (interpretability); {} for other models."""
    if not hasattr(model, "steps"):
        return {}
    lr = model[-1]
    return {f: round(float(c), 4) for f, c in zip(features, lr.coef_[0])} | {"intercept": round(float(lr.intercept_[0]), 4)}


def save_model(model, features: list, model_dir: Path, extra: dict) -> None:
    """matcher.joblib + feature_schema.json (exact training column order)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "matcher.joblib")
    (model_dir / "feature_schema.json").write_text(
        json.dumps({"features": features, **extra}, indent=2), encoding="utf-8")


def load_model(model_dir: Path):
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    return joblib.load(model_dir / "matcher.joblib"), schema["features"]
