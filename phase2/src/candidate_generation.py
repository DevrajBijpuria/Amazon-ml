"""Blocking and candidate generation (S1 -> one target source at a time).

Every rule maps a record to zero or more string keys, prefixed with the exact
normalized country (open set: whatever label the record has, never mapped).
Keys are integer-coded jointly for S1 and target, target keys whose frequency
exceeds ``max_block_size`` are skipped (and recorded), and candidates are the
union of rule-level equi-joins, processed in S1 chunks. No Cartesian product.
"""
import logging
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from feature_engineering import RULE_BITS, STATE_CODES, STREET_CANON, skeleton

STREET_TYPES = set(STREET_CANON) | set(STREET_CANON.values()) | STATE_CODES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- keys
def _country_prefix(rec: pd.DataFrame, cfg: dict) -> pd.Series:
    return rec["country"] + "|" if cfg["use_country_block"] else pd.Series("", index=rec.index)


def _long(idx, keys) -> pd.DataFrame:
    df = pd.DataFrame({"idx": np.asarray(idx, dtype=np.int64), "key": list(keys)})
    return df[df["key"] != ""]


def _address_tokens(addr: str) -> set:
    """Alphabetic address tokens that can identify a locality (no state codes / street types)."""
    return {t for t in addr.split() if len(t) >= 4 and t.isalpha() and t not in STREET_TYPES}


def _name_tokens(s: str) -> set:
    return set(s.split())


def _phonetic_tokens(s: str) -> set:
    """Consonant skeleton of each (romanized) core-name token: yunik ~ unique, investament ~ investment."""
    return {k for k in (skeleton(t) for t in s.split() if t.isascii()) if k}


TOKENIZERS = {"name_core": ("name_core", _name_tokens), "addr_norm": ("addr_norm", _address_tokens),
              "name_phon": ("name_core", _phonetic_tokens)}


def rare_token_frequencies(records: list, kind: str) -> Counter:
    """Document frequency of (country, token) over all given record tables."""
    column, tokens = TOKENIZERS[kind]
    df = Counter()
    for rec in records:
        for country, text in zip(rec["country"].tolist(), rec[column].tolist()):
            df.update(f"{country}|{t}" for t in tokens(text))
    return df


def _rarest_tokens(rec, kind, token_df, k):
    """(position, token) for the k rarest tokens per record that occur at least twice overall
    (a token seen once, e.g. a typo, can never pair two records)."""
    column, tokens = TOKENIZERS[kind]
    out = []
    for i, (country, text) in enumerate(zip(rec["country"], rec[column])):
        toks = sorted((token_df[f"{country}|{t}"], t) for t in tokens(text)
                      if len(t) >= 3 and token_df[f"{country}|{t}"] >= 2)
        out += [(i, t) for _, t in toks[:k]]
    return out


def _rarest(rec, cp, kind, token_df, k):
    pairs = _rarest_tokens(rec, kind, token_df, k)
    return _long([i for i, _ in pairs], [cp[i] + t for i, t in pairs])


def _with_locality(rec, cp, token_df, k, column, tag):
    """Experimental address-first keys: a structured field (house/postal/state) + a rare locality token."""
    vals = rec[column].to_numpy()
    pairs = [(i, t) for i, t in _rarest_tokens(rec, "addr_norm", token_df, k) if vals[i]]
    return _long([i for i, _ in pairs], [f"{cp[i]}{tag}|{vals[i]}|{t}" for i, t in pairs])


def rule_keys(rec: pd.DataFrame, rule: str, cfg: dict, token_df: dict = None) -> pd.DataFrame:
    """Long table (idx, key) of blocking keys for one rule; records without a key are absent."""
    cp = _country_prefix(rec, cfg).tolist()
    idx = np.arange(len(rec))
    if rule == "exact_name":  # compact core name, and its sorted-unique-token form
        keys = [p + k if len(k) >= 2 else "" for p, k in zip(cp, rec["name_key"])]
        srt = [p + "s|" + s if len(s) >= 2 and s != k else "" for p, s, k in zip(cp, rec["name_sorted"], rec["name_key"])]
        return pd.concat([_long(idx, keys), _long(idx, srt)], ignore_index=True)
    if rule == "name_phonetic":
        return _long(idx, [p + k if len(k) >= 4 else "" for p, k in zip(cp, rec["name_phonetic"])])
    if rule == "exact_address":
        sorted_tokens = [" ".join(sorted(set(a.split()))) if len(a.split()) >= 2 else "" for a in rec["addr_norm"]]
        return _long(idx, [p + k if k else "" for p, k in zip(cp, sorted_tokens)])
    if rule == "numeric":
        hs = [p + h + "|s|" + s if h and s else "" for p, h, s in zip(cp, rec["house"], rec["street"])]
        hp = [p + h + "|p|" + z if h and z else "" for p, h, z in zip(cp, rec["house"], rec["postal"])]
        return pd.concat([_long(idx, hs), _long(idx, hp)], ignore_index=True)
    if rule == "name_prefix":
        n = cfg["name_prefix_length"]
        keys = [p + k[:n] if len(k) >= n else "" for p, k in zip(cp, rec["name_key"])]
        return _long(idx, keys)
    if rule == "address_prefix":
        n = cfg["address_prefix_length"]
        compact = rec["addr_norm"].str.replace(" ", "", regex=False)
        keys = [p + k[:n] if len(k) >= n else "" for p, k in zip(cp, compact)]
        return _long(idx, keys)
    if rule == "name_ngram":  # deterministic signature: rarest core-name tokens
        return _rarest(rec, cp, "name_core", token_df["name_core"], cfg["rare_tokens_per_name"])
    if rule == "address_token":  # deterministic signature: rarest locality tokens of the address
        return _rarest(rec, [p + "a|" for p in cp], "addr_norm", token_df["addr_norm"], cfg["rare_tokens_per_address"])
    # ---- Phase 2.5 experimental rules (off unless enabled in the config)
    if rule == "name_phonetic_token":  # rarest per-token consonant skeletons (cross-script spelling)
        return _rarest(rec, [p + "pt|" for p in cp], "name_phon", token_df["name_phon"], cfg["rare_tokens_per_name"])
    if rule == "house_locality":
        return _with_locality(rec, cp, token_df["addr_norm"], cfg["rare_tokens_per_address"], "house", "hl")
    if rule == "postal_locality":
        return _with_locality(rec, cp, token_df["addr_norm"], cfg["rare_tokens_per_address"], "postal", "pl")
    if rule == "state_locality":
        return _with_locality(rec, cp, token_df["addr_norm"], cfg["rare_tokens_per_address"], "state", "sl")
    if rule == "name_addr_composite":  # Phase 3: rarest-2 core-name tokens x rarest-2 locality tokens
        addr = {}
        for i, t in _rarest_tokens(rec, "addr_norm", token_df["addr_norm"], cfg["rare_tokens_per_address"]):
            addr.setdefault(i, []).append(t)
        pairs = [(i, f"{cp[i]}na|{n}|{a}") for i, n in _rarest_tokens(rec, "name_core", token_df["name_core"],
                                                                       cfg["rare_tokens_per_name"]) for a in addr.get(i, [])]
        return _long([i for i, _ in pairs], [k for _, k in pairs])
    if rule == "address_signature":  # sorted non-numeric, non-state comma components (order-free locality)
        sigs = []
        for p, comps in zip(cp, rec["components"]):
            parts = sorted(c for c in comps.split("|") if c and c not in STATE_CODES and not any(ch.isdigit() for ch in c))
            sigs.append(p + "sig|" + "|".join(parts) if len(parts) >= 2 else "")
        return _long(idx, sigs)
    raise ValueError(rule)


# ---------------------------------------------------------------- index
RULE_FLAGS = {"exact_name": "use_exact_name_block", "exact_address": "use_exact_address_block",
              "numeric": "use_numeric_block", "name_prefix": "use_name_prefix_block",
              "address_prefix": "use_address_prefix_block", "name_ngram": "use_name_char_ngram_block",
              "name_phonetic": "use_name_phonetic_block", "address_token": "use_address_token_block",
              # Phase 2.5 experimental rules: absent from the B0 config, so off by default
              "name_phonetic_token": "use_name_phonetic_token_block", "house_locality": "use_house_locality_block",
              "postal_locality": "use_postal_locality_block", "state_locality": "use_state_locality_block",
              "address_signature": "use_address_signature_block",
              # Phase 3 experimental rule: off unless enabled in the config
              "name_addr_composite": "use_name_addr_composite_block"}
_TOKEN_DFS = {"name_ngram": ("name_core",), "address_token": ("addr_norm",), "name_phonetic_token": ("name_phon",),
              "house_locality": ("addr_norm",), "postal_locality": ("addr_norm",), "state_locality": ("addr_norm",),
              "name_addr_composite": ("name_core", "addr_norm")}


def enabled_rules(cfg: dict) -> list:
    return [r for r, flag in RULE_FLAGS.items() if cfg.get(flag, False)]


def rule_tables(s1: pd.DataFrame, tgt: pd.DataFrame, cfg: dict, rules: list = None) -> dict:
    """Integer-coded S1 and target keys for each rule, with target key frequencies (no cutoff applied)."""
    rules = rules or enabled_rules(cfg)
    token_df = {kind: rare_token_frequencies([s1, tgt], kind) for kind in {k for r in rules for k in _TOKEN_DFS.get(r, ())}}
    tables = {}
    for rule in rules:
        sk, tk = rule_keys(s1, rule, cfg, token_df), rule_keys(tgt, rule, cfg, token_df)
        codes, uniques = pd.factorize(pd.concat([sk["key"], tk["key"]], ignore_index=True))
        sk = pd.DataFrame({"s1_idx": sk["idx"].to_numpy(np.int32), "code": codes[:len(sk)]})
        tk = pd.DataFrame({"t_idx": tk["idx"].to_numpy(np.int32), "code": codes[len(sk):]}).drop_duplicates()
        tables[rule] = {"s1": sk.drop_duplicates(), "t": tk, "freq": tk["code"].value_counts(), "uniques": uniques}
    return tables


def apply_cutoff(tables: dict, max_block_size, by_rule: dict = None) -> dict:
    """Blocking index per rule: target keys holding more than ``max_block_size`` records are skipped
    (``None`` = no cutoff, diagnostic only); ``by_rule`` overrides the cutoff for the rules it names."""
    blocking = {}
    for rule, tb in tables.items():
        tk, freq, uniques = tb["t"], tb["freq"], tb["uniques"]
        limit = (by_rule or {}).get(rule, max_block_size)
        skipped = freq[freq > limit] if limit is not None else freq.iloc[:0]
        is_skipped = tk["code"].isin(skipped.index)
        blocking[rule] = {
            "s1": tb["s1"],
            "t": tk[~is_skipped],
            "t_skipped": tk[is_skipped],
            "skipped": pd.DataFrame({"code": skipped.index.to_numpy(), "key": [str(uniques[c]) for c in skipped.index],
                                     "frequency": skipped.to_numpy()}).sort_values(["frequency", "key"],
                                                                                   ascending=[False, True]),
            "target_keys": int(len(freq)),
            "target_keyed_records": int(tk["t_idx"].nunique()),
            "max_block_size": limit,
        }
        log.info("  %s: %d target keys, %d skipped (> %s records)", rule, len(freq), len(skipped), limit)
    return blocking


def build_blocking(s1: pd.DataFrame, tgt: pd.DataFrame, cfg: dict) -> dict:
    """Build every rule's target index once; returns per-rule S1 keys, target keys and skipped keys."""
    return apply_cutoff(rule_tables(s1, tgt, cfg), cfg["max_block_size"], cfg.get("max_block_size_by_rule"))


# ---------------------------------------------------------------- candidates
def _cap(cands: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame, limit: int):
    """Keep at most ``limit`` candidates per S1: most rules first, then max(name, address) edit similarity,
    then target position. Returns (kept, dropped)."""
    sizes = cands.groupby("s1_idx")["t_idx"].transform("size")
    over = cands[sizes > limit].copy()
    if over.empty:
        return cands, cands.iloc[:0]
    over["n_rules"] = [int(b).bit_count() for b in over["blocks"].to_numpy()]
    a, b = s1["name_key"].to_numpy()[over["s1_idx"].to_numpy()], tgt["name_key"].to_numpy()[over["t_idx"].to_numpy()]
    c, d = s1["addr_norm"].to_numpy()[over["s1_idx"].to_numpy()], tgt["addr_norm"].to_numpy()[over["t_idx"].to_numpy()]
    over["sim"] = [max(Levenshtein.normalized_similarity(w, x), Levenshtein.normalized_similarity(y, z) if y and z else 0.0)
                   for w, x, y, z in zip(a, b, c, d)]  # best of name and address, so address-only positives survive
    over = over.sort_values(["s1_idx", "n_rules", "sim", "t_idx"], ascending=[True, False, False, True])
    rank = over.groupby("s1_idx").cumcount()
    keep = over[rank < limit][["s1_idx", "t_idx", "blocks"]]
    dropped = over[rank >= limit][["s1_idx", "t_idx", "blocks"]]
    kept = pd.concat([cands[sizes <= limit], keep]).sort_values(["s1_idx", "t_idx"], ignore_index=True)
    return kept, dropped.reset_index(drop=True)


def iter_candidates(blocking: dict, s1: pd.DataFrame, tgt: pd.DataFrame, cfg: dict):
    """Yield (s1_lo, s1_hi, precap, kept, dropped) per S1 chunk; candidate frames hold
    s1_idx, t_idx and a ``blocks`` bitmask of the rules that retrieved the pair."""
    step = cfg["source1_chunk_size"]
    for lo in range(0, len(s1), step):
        hi = min(lo + step, len(s1))
        parts = []
        for rule, blk in blocking.items():
            sk = blk["s1"][(blk["s1"]["s1_idx"] >= lo) & (blk["s1"]["s1_idx"] < hi)]
            pairs = sk.merge(blk["t"], on="code")[["s1_idx", "t_idx"]].drop_duplicates()
            pairs["blocks"] = np.int32(RULE_BITS[rule])
            parts.append(pairs)
        precap = (pd.concat(parts, ignore_index=True).groupby(["s1_idx", "t_idx"], as_index=False)["blocks"].sum()
                  if parts else pd.DataFrame(columns=["s1_idx", "t_idx", "blocks"]))
        kept, dropped = _cap(precap, s1, tgt, cfg["max_candidates_per_source1"])
        yield lo, hi, precap, kept, dropped


# ---------------------------------------------------------------- recall accounting
def lost_to_skipped_keys(blocking: dict, missing: pd.DataFrame) -> dict:
    """Recall cost of skipping (correction 4). ``missing`` = positive pairs absent from the final
    candidates. Returns, per rule, the lost pairs sharing any skipped key of that rule and a
    per-key count (a pair sharing two skipped keys counts under both), plus the any-rule total."""
    out, lost = {}, []
    for rule, blk in blocking.items():
        m = missing.merge(blk["s1"], on="s1_idx").merge(blk["t_skipped"], on=["t_idx", "code"])
        pairs = m[["s1_idx", "t_idx"]].drop_duplicates()
        out[rule] = {"pairs": int(len(pairs)), "per_code": m.groupby("code").size().to_dict()}
        lost.append(pairs)
    out["any_rule"] = int(pd.concat(lost).drop_duplicates().shape[0]) if lost else 0
    return out


def candidate_summary(sizes: np.ndarray, n_s1: int) -> dict:
    """Candidates-per-S1 distribution, counting S1 entities with no candidates as zero."""
    full = np.zeros(n_s1, dtype=np.int64)
    full[: len(sizes)] = sizes
    return {
        "candidate_pairs": int(full.sum()),
        "mean_candidates_per_s1": round(float(full.mean()), 3) if n_s1 else 0.0,
        "median_candidates_per_s1": float(np.median(full)) if n_s1 else 0.0,
        "p95_candidates_per_s1": float(np.percentile(full, 95)) if n_s1 else 0.0,
        "p99_candidates_per_s1": float(np.percentile(full, 99)) if n_s1 else 0.0,
        "max_candidates_per_s1": int(full.max()) if n_s1 else 0,
        "s1_with_zero_candidates": int((full == 0).sum()),
    }
