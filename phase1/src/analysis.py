"""Phase 1 analyses: name/address/country variation, ground truth, match difficulty,
positive vs sampled-negative pairs, singletons and F0.5 validation metrics.

Everything here describes the TRAIN data. Nothing here generates candidates,
predicts matches or tunes thresholds.
"""
import logging
import random
import re
from collections import Counter

import pandas as pd

from normalization import normalize_address, normalize_country, normalize_name

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- patterns
# Python `re` over plain lists on purpose: pandas 3 routes str.contains through
# pyarrow's RE2, which has different Unicode/backreference semantics.
_NOT_PUNCT = r"\w\s\u0900-\u097f"  # Devanagari marks are not \w, but not punctuation either
LEGAL_SUFFIX = re.compile(
    r"(?i)(?<!\w)(inc|incorporated|llc|l\.l\.c|ltd|limited|pvt|private|corp|corporation"
    r"|co|company|llp|lp|plc|p\.?c|pllc|sarl|sas|gmbh)\.?(?!\w)")
STREET_TYPE = re.compile(
    r"(?i)(?<!\w)(st|street|rd|road|ave|av|avenue|dr|drive|blvd|boulevard|ln|lane|ct|court"
    r"|cir|circle|hwy|highway|pkwy|parkway|pl|place|marg|nagar|rue)\.?(?!\w)")

_COMMON = {
    "all_upper_latin": r"^[^a-z]*[A-Z][^a-z]*$",
    "all_lower_latin": r"^[^A-Z]*[a-z][^A-Z]*$",
    "leading_trailing_space": r"^\s|\s$",
    "multiple_spaces": r"\s{2,}",
    "non_ascii": r"[^\x00-\x7f]",
    "devanagari": r"[\u0900-\u097f]",
    "latin_diacritic": r"[\u00c0-\u024f]",
    "replacement_char": "\ufffd",
}
NAME_PATTERNS = {k: re.compile(v) for k, v in {
    **_COMMON,
    "punctuation": rf"[^{_NOT_PUNCT}]",
    "leading_symbol_junk": rf"^[^{_NOT_PUNCT}]",
    "brackets": r"[()\[\]{}]",
    "ampersand": r"&",
    "word_and": r"(?i)\band\b",
    "digits": r"\d",
    "url_or_domain": r"(?i)www\.|\.(com|net|org|in)\b",
    "embedded_id": r"(?i)\bid\s*[:#]",
    "dba_trade_name": r"(?i)\b(dba|d/b/a|t/a|trading as)\b",
    "adjacent_repeated_token": r"(?i)\b(\w+)\s+\1\b",
    "legal_suffix": LEGAL_SUFFIX.pattern,
}.items()}
ADDRESS_PATTERNS = {k: re.compile(v) for k, v in {
    **_COMMON,
    "empty": r"^\s*$",
    "line_break": r"[\r\n]",
    "starts_with_digit": r"^\s*\d",
    "starts_with_state_code": r"^[A-Z]{2},",
    "ends_with_state_code": r",\s*[A-Z]{2}$",
    "us_zip_at_end": r"\b\d{5}(-\d{4})?$",
    "six_digit_pin": r"(?<!\d)(\d{6}|\d{3}\s\d{3})(?!\d)",
    "landmark": r"(?i)(?<!\w)(near|nr|opp|opposite|behind|b/h|beside|adjacent|next to)(?!\w)",
    "house_number_marker": r"(?i)(?<!\w)(no|h\.?\s?no|kh\.?\s?no|plot|flat|shop|door)(?!\w)",
    "unit_suite": r"(?i)(?<!\w)(unit|suite|ste|apt)(?!\w)|#",
    "hyphen_slash_number": r"\d+\s*[-/]\s*\d+",
    "ordinal_number": r"(?i)\b\d+(st|nd|rd|th)\b",
    "po_box": r"(?i)\bp\.?\s?o\.?\s?box\b",
    "adjacent_repeated_token": r"(?i)\b(\w+)[\s,]+\1\b",
    "street_type": STREET_TYPE.pattern,
}.items()}


def pattern_rates(values: list, patterns: dict) -> dict:
    """Share of values matching each regex."""
    n = len(values) or 1
    return {k: round(sum(1 for v in values if p.search(v)) / n, 4) for k, p in patterns.items()}


def top(counter: Counter, k: int = 20) -> list:
    """Top-k (value, count) sorted by count desc then value, so ties are deterministic."""
    return [[key, n] for key, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]]


def variant_forms(values: list, regex: re.Pattern, k: int = 25) -> list:
    """Most frequent raw spellings matched by ``regex`` (e.g. Pvt / PVT / Pvt. / Private)."""
    return top(Counter(m.group(0) for v in values for m in regex.finditer(v)), k)


# ---------------------------------------------------------------- variation
def analyze_text_variation(df: pd.DataFrame, sample_rows: int, seed: int) -> dict:
    """Name/address variation rates overall and per country on a fixed-seed row sample."""
    sample = df.sample(n=min(sample_rows, len(df)), random_state=seed)
    names, addrs = sample["business_name"].tolist(), sample["business_address"].tolist()
    result = {
        "sample_rows": len(sample),
        "name_rates": pattern_rates(names, NAME_PATTERNS),
        "address_rates": pattern_rates(addrs, ADDRESS_PATTERNS),
        "name_rates_by_country": {}, "address_rates_by_country": {},
        "legal_suffix_forms": variant_forms(names, LEGAL_SUFFIX),
        "street_type_forms": variant_forms(addrs, STREET_TYPE),
        "address_comma_count": top(Counter(a.count(",") for a in addrs), 15),
        "top_name_tokens": top(Counter(t for v in names for t in normalize_name(v).split()), 30),
        "names_changed_by_normalization": round(
            sum(normalize_name(v) != v for v in names) / (len(names) or 1), 4),
    }
    for country, grp in sorted(sample.groupby("country"), key=lambda kv: kv[0]):
        result["name_rates_by_country"][country] = pattern_rates(
            grp["business_name"].tolist(), NAME_PATTERNS)
        result["address_rates_by_country"][country] = pattern_rates(
            grp["business_address"].tolist(), ADDRESS_PATTERNS)
    return result


def analyze_country(df: pd.DataFrame) -> dict:
    """Country label distribution, missing values, representation variants, odd values."""
    raw = df["country"]
    counts = raw.value_counts()
    norm = pd.Series([normalize_country(v) for v in counts.index], index=counts.index)
    variants = {n: sorted(norm[norm == n].index.tolist()) for n in norm.unique()
                if (norm == n).sum() > 1}
    unexpected = [v for v in counts.index
                  if v != v.strip() or not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", v.strip())]
    return {
        "distinct_raw": int(len(counts)),
        "distinct_normalized": int(norm[norm != ""].nunique()),
        "missing": int((raw.str.strip() == "").sum()),
        "distribution": top(Counter(counts.to_dict()), 50),
        "normalized_variants": variants,
        "unexpected_values": sorted(unexpected)[:20],
    }


# ---------------------------------------------------------------- ground truth
def explode_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    """One row per ground-truth positive pair: (source1_entity_id, target_id)."""
    pairs = gt.assign(target_id=gt["matched_entity_ids"].str.split(","))
    pairs = pairs.explode("target_id")[["source1_entity_id", "target_id"]]
    pairs["target_id"] = pairs["target_id"].str.strip()
    return pairs[pairs["target_id"] != ""].reset_index(drop=True)


def match_counts(gt: pd.DataFrame) -> pd.DataFrame:
    """Per Source-1 entity: number of S2, S3 and total ground-truth matches."""
    ids = gt["matched_entity_ids"]
    out = pd.DataFrame({"source1_entity_id": gt["source1_entity_id"],
                        "n_s2": ids.str.count("S2-").astype(int),
                        "n_s3": ids.str.count("S3-").astype(int)})
    out["n_matches"] = out["n_s2"] + out["n_s3"]
    return out


def _bucket(n: int) -> str:
    return "zero" if n == 0 else "one" if n == 1 else "multi"


def analyze_ground_truth(gt: pd.DataFrame, pairs: pd.DataFrame, counts: pd.DataFrame) -> dict:
    """Match-count distributions and relationship sanity checks, from the file as given."""
    per_target = pairs["target_id"].value_counts()
    has2, has3 = counts["n_s2"] > 0, counts["n_s3"] > 0
    composition = Counter({"none": int((~has2 & ~has3).sum()), "s2_only": int((has2 & ~has3).sum()),
                           "s3_only": int((~has2 & has3).sum()), "s2_and_s3": int((has2 & has3).sum())})
    n = len(counts)
    buckets = counts["n_matches"].map(_bucket).value_counts()
    return {
        "source1_rows": n,
        "source1_ids_unique": bool(gt["source1_entity_id"].is_unique),
        "positive_pairs": int(len(pairs)),
        "pairs_by_target_source": dict(sorted(pairs["target_id"].str[:3].value_counts().to_dict().items())),
        "match_bucket_counts": {b: int(buckets.get(b, 0)) for b in ("zero", "one", "multi")},
        "match_bucket_pct": {b: round(100 * buckets.get(b, 0) / n, 3) for b in ("zero", "one", "multi")},
        "matches_per_s1": dict(sorted(Counter(counts["n_matches"].tolist()).items())),
        "s2_matches_per_s1": dict(sorted(Counter(counts["n_s2"].tolist()).items())),
        "s3_matches_per_s1": dict(sorted(Counter(counts["n_s3"].tolist()).items())),
        "composition": dict(sorted(composition.items())),
        "duplicate_pairs": int(pairs.duplicated().sum()),
        "targets_linked_to_multiple_s1": int((per_target > 1).sum()),
        "targets_with_bad_prefix": int((~pairs["target_id"].str.match(r"S[23]-")).sum()),
    }


def target_coverage(source_ids: pd.Series, pairs: pd.DataFrame, prefix: str) -> dict:
    """How many S2/S3 records are referenced by ground truth, and dangling GT references."""
    gt_ids = pairs.loc[pairs["target_id"].str.startswith(prefix), "target_id"]
    referenced = source_ids.isin(gt_ids)
    return {
        "records": int(len(source_ids)),
        "referenced_by_ground_truth": int(referenced.sum()),
        "not_referenced": int((~referenced).sum()),
        "not_referenced_pct": round(100 * (~referenced).mean(), 3) if len(source_ids) else 0.0,
        "gt_ids_missing_from_source": int((~gt_ids.isin(source_ids)).sum()),
    }


# ---------------------------------------------------------------- singletons
_SINGLETON_PATTERNS = {k: NAME_PATTERNS[k] for k in ("legal_suffix", "non_ascii", "devanagari")}
_SINGLETON_ADDR = {k: ADDRESS_PATTERNS[k] for k in ("landmark", "six_digit_pin", "us_zip_at_end")}


def analyze_singletons(s1: pd.DataFrame, counts: pd.DataFrame, sample_rows: int, seed: int) -> dict:
    """Compare zero-, one- and multi-match Source-1 records."""
    df = s1.merge(counts, left_on="entity_id", right_on="source1_entity_id", how="left")
    df["n_matches"] = df["n_matches"].fillna(0).astype(int)
    df["bucket"] = df["n_matches"].map(_bucket)
    out = {}
    for bucket in ("zero", "one", "multi"):
        grp = df[df["bucket"] == bucket]
        if grp.empty:
            continue
        smp = grp.sample(n=min(sample_rows, len(grp)), random_state=seed)
        info = {
            "records": int(len(grp)),
            "country_pct": {k: round(100 * v / len(grp), 2)
                            for k, v in sorted(grp["country"].value_counts().items())},
            "address_missing_pct": round(100 * (grp["business_address"].str.strip() == "").mean(), 3),
            "name_length_mean": round(float(grp["business_name"].str.len().mean()), 2),
            "address_length_mean": round(float(grp["business_address"].str.len().mean()), 2),
            "sample_rows": len(smp),
            "name_rates": pattern_rates(smp["business_name"].tolist(), _SINGLETON_PATTERNS),
            "address_rates": pattern_rates(smp["business_address"].tolist(), _SINGLETON_ADDR),
        }
        if bucket == "one":
            info["single_match_source"] = {"S2": int((grp["n_s2"] == 1).sum()),
                                           "S3": int((grp["n_s3"] == 1).sum())}
        out[bucket] = info
    out["zero_match_pct_by_country"] = {
        c: round(100 * (g["bucket"] == "zero").mean(), 3)
        for c, g in sorted(df.groupby("country"), key=lambda kv: kv[0])}
    return out


# ---------------------------------------------------------------- similarity
def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert/delete/substitute = 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def jaccard(a: set, b: set) -> float:
    """|a∩b| / |a∪b|; NaN when both are empty (similarity undefined)."""
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def char_ngrams(text: str, n: int = 3) -> set:
    padded = f" {text} "
    return {padded[i:i + n] for i in range(max(len(padded) - n + 1, 0))} if text else set()


def _edit_sim(a: str, b: str) -> float:
    longest = max(len(a), len(b))
    return 1 - levenshtein(a, b) / longest if longest else float("nan")


# ---------------------------------------------------------------- pair sampling
def sample_s1_ids(s1_ids: pd.Series, n: int, seed: int) -> list:
    """Fixed-seed sample of Source-1 ids for pair-level analysis."""
    return sorted(s1_ids.sample(n=min(n, len(s1_ids)), random_state=seed).tolist())


def sample_negatives(s1_ids: list, records: dict, truth: dict, pool: pd.DataFrame, seed: int) -> list:
    """For each sampled S1 entity pick one random pool record of the same country
    that is NOT in its ground-truth list.

    Assumption (documented in the report): train ground truth is complete for
    Source 1, so a pair absent from it is a non-match. Pool records are drawn
    uniformly from S2/S3, i.e. these are easy negatives, not hard ones.
    """
    rng = random.Random(seed)
    by_country = {c: sorted(g["entity_id"].tolist()) for c, g in pool.groupby("country")}
    negatives = []
    for s1 in s1_ids:
        cands = by_country.get(records[s1][2], [])
        for _ in range(20):
            if not cands:
                break
            pick = rng.choice(cands)
            if pick not in truth.get(s1, set()):
                negatives.append((s1, pick))
                break
    return negatives


def pair_features(pairs: list, records: dict) -> pd.DataFrame:
    """Similarity/equality diagnostics for (s1_id, target_id, label) triples."""
    rows = []
    for s1, tgt, label in pairs:
        (na, aa, ca), (nb, ab, cb) = records[s1], records[tgt]
        nna, nnb = normalize_name(na), normalize_name(nb)
        naa, nab = normalize_address(aa), normalize_address(ab)
        tna, tnb, taa, tab = set(nna.split()), set(nnb.split()), set(naa.split()), set(nab.split())
        rows.append({
            "source1_entity_id": s1, "target_id": tgt, "label": label, "target_source": tgt[:2],
            "name_raw_eq": na == nb,
            "name_norm_eq": nna == nnb,
            "name_token_order_only": tna == tnb and nna != nnb,
            "name_token_jaccard": jaccard(tna, tnb),
            "name_char3_jaccard": jaccard(char_ngrams(nna), char_ngrams(nnb)),
            "name_edit_distance": levenshtein(nna, nnb),
            "name_edit_sim": _edit_sim(nna, nnb),
            "address_missing_either": not naa or not nab,
            "address_raw_eq": aa == ab,
            "address_norm_eq": bool(naa) and naa == nab,
            "address_token_order_only": bool(taa) and taa == tab and naa != nab,
            "address_token_jaccard": jaccard(taa, tab),
            "address_char3_jaccard": jaccard(char_ngrams(naa), char_ngrams(nab)),
            "address_edit_sim": _edit_sim(naa, nab),
            "country_raw_eq": ca == cb,
            "country_norm_eq": normalize_country(ca) == normalize_country(cb),
        })
    return pd.DataFrame(rows)


def summarize_pairs(feats: pd.DataFrame, group_cols: list) -> dict:
    """Rates for boolean features and quantiles for numeric ones, per group."""
    out = {}
    for key, grp in feats.groupby(group_cols):
        key = key if isinstance(key, str) else "/".join(map(str, key))
        stats = {"pairs": int(len(grp))}
        for col in grp.columns:
            if col in ("source1_entity_id", "target_id", "label", "target_source", *group_cols):
                continue
            s = grp[col]
            if s.dtype == bool:
                stats[col] = round(float(s.mean()), 4)
            else:
                s = s.dropna()
                stats[col] = {q: round(float(s.quantile(p)), 4) for q, p in
                              (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9))}
                stats[col]["mean"] = round(float(s.mean()), 4) if len(s) else None
        out[key] = stats
    return out


def token_differences(pos_pairs: list, records: dict, field: int, normalizer, k: int = 25) -> list:
    """Tokens that appear on only one side of a positive pair (reveals abbreviations/suffixes)."""
    diff = Counter()
    for s1, tgt in pos_pairs:
        a, b = set(normalizer(records[s1][field]).split()), set(normalizer(records[tgt][field]).split())
        diff.update(a ^ b)
    return top(diff, k)


# ---------------------------------------------------------------- F0.5
def f_beta(tp: int, fp: int, fn: int, beta: float = 0.5) -> dict:
    """Precision, recall and F-beta from counts; zero denominators yield 0.0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    score = (1 + b2) * precision * recall / denom if denom else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f_beta": score}


def entity_f05(pred: set, truth: set) -> float:
    """Challenge rule for one S1 entity: empty prediction on a singleton scores 1.0."""
    if not pred and not truth:
        return 1.0
    return f_beta(len(pred & truth), len(pred - truth), len(truth - pred))["f_beta"]


def macro_f05(pred: dict, truth: dict) -> dict:
    """Macro F0.5 over every entity in ``truth`` (missing predictions = empty), plus micro counts."""
    tp = fp = fn = 0
    scores = []
    for s1, true_set in truth.items():
        p = set(pred.get(s1, ()))
        scores.append(entity_f05(p, true_set))
        tp, fp, fn = tp + len(p & true_set), fp + len(p - true_set), fn + len(true_set - p)
    return {"entities": len(truth), "macro_f05": sum(scores) / len(scores) if scores else 0.0,
            "micro": f_beta(tp, fp, fn)}
