"""Phase 2 normalization (built on the frozen Phase 1 rules) and pair features.

Record preparation produces, per record:
  name_norm      Phase 1 normalize_name + legal-suffix canonicalization (Unicode kept)
  name_translit  name_norm with Indic scripts romanized (separate signal, spec 27)
  name_core      name_translit without legal suffixes, web tokens and 'the'/'and'
  name_key       name_core with spaces removed (catches "acmecorp.com" vs "Acme Corp")
  name_sorted    sorted unique core tokens, spaces removed (catches reordering / doubled words)
  name_phonetic  consonant skeleton of name_key (catches transliteration spelling: yunik ~ unique)
  addr_norm      Phase 1 normalize_address + <NULL> removal + street/state canonicalization
  house, postal, state, street   deterministic address parts ('' when absent)
  country        Phase 1 normalize_country: trimmed + casefolded, open set, never mapped

The same functions serve training and inference (spec 44).
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase1" / "src"))
from normalization import normalize_address as p1_address  # noqa: E402  frozen Phase 1 rules
from normalization import normalize_country  # noqa: E402
from normalization import normalize_name as p1_name  # noqa: E402

# ---------------------------------------------------------------- vocabularies
LEGAL_CANON = {
    "pvt": "private", "pvte": "private", "praivet": "private", "praivat": "private",
    "ltd": "limited", "ltda": "limited", "limitad": "limited",
    "inc": "incorporated", "corp": "corporation", "co": "company", "cos": "company",
    "प्राइवेट": "private", "लिमिटेड": "limited", "एलएलपी": "llp", "प्रा": "private", "लि": "limited",
}
LEGAL_TOKENS = {"private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp",
                "plc", "pc", "pllc", "sarl", "sas", "sa", "gmbh", "opc"}
NAME_NOISE_TOKENS = {"com", "www", "net", "org", "the", "and"}
_LEET = str.maketrans("01358", "olesb")  # digit-for-letter typos inside words: litt1e, c0astal
STREET_CANON = {"road": "rd", "street": "st", "drive": "dr", "avenue": "ave", "lane": "ln",
                "court": "ct", "place": "pl", "boulevard": "blvd", "highway": "hwy",
                "circle": "cir", "parkway": "pkwy", "terrace": "ter", "square": "sq"}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {  # English, variant and native-script spellings observed in the data -> code
    "maharashtra": "mh", "महाराष्ट्र": "mh", "delhi": "dl", "दिल्ली": "dl",
    "uttar pradesh": "up", "उत्तर प्रदेश": "up", "karnataka": "ka", "ಕರ್ನಾಟಕ": "ka",
    "tamil nadu": "tn", "தமிழ்நாடு": "tn", "gujarat": "gj", "ગુજરાત": "gj",
    "west bengal": "wb", "পশ্চিমবঙ্গ": "wb", "telangana": "tg", "ts": "tg", "తెలంగాణ": "tg",
    "haryana": "hr", "हरियाणा": "hr", "rajasthan": "rj", "राजस्थान": "rj",
    "kerala": "kl", "കേരളം": "kl", "bihar": "br", "बिहार": "br",
    "madhya pradesh": "mp", "मध्य प्रदेश": "mp", "andhra pradesh": "ap", "ఆంధ్రప్రదేశ్": "ap",
    "punjab": "pb", "ਪੰਜਾਬ": "pb", "odisha": "od", "orissa": "od", "ଓଡ଼ିଶା": "od",
    "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "himachal pradesh": "hp", "goa": "ga", "jammu and kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py",
}
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())
# (city-level "new delhi" deliberately maps through "delhi" -> "new dl" on both sides)
_STATE_NAMES = {p1_address(k): v for k, v in {**US_STATES, **IN_STATES}.items()}
_STATE_RE = re.compile(r"(?<!\S)(" + "|".join(sorted(map(re.escape, _STATE_NAMES), key=len, reverse=True))
                       + r")(?!\S)")

_MOJIBAKE = re.compile(r"[ÂâÃ][\x80-\x9f]+")  # UTF-8 punctuation decoded as Latin-1
_EMBEDDED_ID = re.compile(r"(?i)\(?\bid\s*[:#]\s*\d+\)?")
_NULL = re.compile(r"(?i)<\s*null\s*>")
_NUMBER = re.compile(r"\d+(?:\s*[-/]\s*\d+)*")
_PIN = re.compile(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)")
_POSTAL5 = re.compile(r"(?<![\d-])(\d{5})(?:-\d{4})?(?![\d-])")  # US ZIP / French code (never the first number)
AMBIGUOUS_LEGAL = {"sa", "pc", "lp", "sas", "opc"}  # legal only as the final token ("P C Jewellers" is not)

# ---------------------------------------------------------------- Indic romanization
# Every major Indic block (Devanagari .. Malayalam) shares the ISCII layout, so one
# offset table romanizes all of them. Vowel length is dropped (aa -> a) on purpose.
_INDIC_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au"}
_CONS = dict(zip(range(0x15, 0x3A), [
    "k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n", "t", "th",
    "d", "dh", "n", "n", "p", "ph", "b", "bh", "m", "y", "r", "r", "l", "l", "l", "v", "sh",
    "sh", "s", "h"]))
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x46: "e",
           0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_MALAYALAM_CLUSTERS = {"റ്റ": "ട", "ന്റ": "ന്ട"}  # ṟṟ -> ṭ, nṟ -> nṭ
_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}


def _indic(ch: str):
    cp = ord(ch)
    for base in _INDIC_BASES:
        if base <= cp < base + 0x80:
            return cp - base
    return None


def romanize(text: str) -> str:
    """Romanize Indic-script letters; other characters pass through unchanged."""
    if text.isascii():
        return text
    for old, new in _MALAYALAM_CLUSTERS.items():
        text = text.replace(old, new)
    out = []
    chars = [c for c in text if _indic(c) != _NUKTA]
    for i, ch in enumerate(chars):
        if ord(ch) in _CHILLU:
            out.append(_CHILLU[ord(ch)])
            continue
        off = _indic(ch)
        if off is None:
            out.append(ch)
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))
        elif off in _CONS:
            if not (out and out[-1] == _CONS[off]):  # geminate consonants: tt -> t
                out.append(_CONS[off])
            nxt = _indic(chars[i + 1]) if i + 1 < len(chars) else None
            # inherent vowel unless a vowel sign/virama follows or the word ends (schwa deletion)
            if nxt is not None and nxt not in _MATRAS and nxt != _VIRAMA and nxt not in _SIGNS:
                out.append("a")
        else:
            out.append(_VOWELS.get(off) or _MATRAS.get(off) or _SIGNS.get(off) or "")
    return "".join(out)


# ---------------------------------------------------------------- normalization
def _join_initials(tokens: list) -> list:
    """'l l c' -> 'llc', 'm g road' -> 'mg road' (punctuation-separated abbreviations)."""
    out, run = [], []
    for t in tokens + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        if t:
            out.append(t)
    return out


def _text(raw) -> str:
    """Any non-string (None, NaN, pandas.NA, numbers) is treated as missing."""
    return raw if isinstance(raw, str) else ""


_SKELETON_RULES = [(re.compile(p), r) for p, r in (
    (r"tion", "shan"), (r"ph", "f"), (r"ck", "k"), (r"c(?=[eiy])", "s"), (r"[cq]", "k"), (r"x", "ks"),
    (r"z", "s"), (r"w", "v"), (r"[aeiouyh]", ""), (r"(.)\1+", r"\1"))]


def skeleton(text: str) -> str:
    """Consonant skeleton of Latin text: soft/hard c, ph/f, aspirates and vowels are merged, so
    English and romanized-Indic spellings agree (unique investment ~ yunik investament)."""
    for pattern, repl in _SKELETON_RULES:
        text = pattern.sub(repl, text)
    return text


LEGAL_SKELETONS = {skeleton(w): w for w in ("private", "limited", "incorporated", "corporation", "company")}


def _canon(t: str, romanized: bool) -> str:
    """Canonical legal word for abbreviations (pvt) and, in romanized Indic names only, for
    transliterated spellings (praivet, limatid) -- a Latin name like "Pravat" is never touched."""
    if t in LEGAL_CANON:
        return LEGAL_CANON[t]
    return LEGAL_SKELETONS.get(skeleton(t), t) if romanized and len(t) >= 5 and t.isascii() else t


def _is_legal(tokens: list, i: int) -> bool:
    t = tokens[i]
    return t in LEGAL_TOKENS and (t not in AMBIGUOUS_LEGAL or i == len(tokens) - 1)


def _unleet(t: str) -> str:
    return t.translate(_LEET) if sum(c.isalpha() for c in t) >= 3 and not t.isalpha() else t


def normalize_name2(raw) -> tuple:
    """(name_norm, name_translit, name_core, name_key, has_legal_suffix, legal_set, non_latin,
    name_sorted, name_phonetic)."""
    text = _EMBEDDED_ID.sub(" ", _MOJIBAKE.sub(" ", _NULL.sub(" ", _text(raw))))
    tokens = _join_initials(p1_name(text).split())
    norm = " ".join(LEGAL_CANON.get(t, t) for t in tokens)
    romanized = not norm.isascii()
    translit_tokens = [_canon(t, romanized)
                       for t in _join_initials([_unleet(t) for t in p1_name(romanize(norm)).split()])]
    translit = " ".join(translit_tokens)
    legal = sorted({t for i, t in enumerate(translit_tokens) if _is_legal(translit_tokens, i)})
    core = " ".join(t for i, t in enumerate(translit_tokens)
                    if not _is_legal(translit_tokens, i) and t not in NAME_NOISE_TOKENS)  # '' if only suffixes
    key = core.replace(" ", "")
    return (norm, translit, core, key, bool(legal), " ".join(legal), norm != translit,
            "".join(sorted(set(core.split()))), skeleton(key) if key.isascii() else "")


def _address_text(raw: str) -> str:
    text = p1_address(_MOJIBAKE.sub(" ", _NULL.sub(" ", _text(raw))))
    text = _STATE_RE.sub(lambda m: _STATE_NAMES[m.group(1)], text)
    return " ".join(STREET_CANON.get(t, t) for t in _join_initials(text.split()))


def normalize_address2(raw, country: str = "") -> tuple:
    """(addr_norm, house, postal, state, street, components). Postal format depends only on
    whether the record is Indian (6-digit PIN); every other country, known or not, uses the
    5-digit rule, which never takes the first number of the address (the house number)."""
    raw = _MOJIBAKE.sub(" ", _NULL.sub(" ", _text(raw)))
    norm = _address_text(raw)
    nfkc = unicodedata.normalize("NFKC", raw)
    postal = ""
    if country == "india":
        m = _PIN.search(nfkc)
    else:
        first = _NUMBER.search(nfkc)
        m = next((m for m in _POSTAL5.finditer(nfkc) if not first or m.start() != first.start()), None)
    if m:
        postal = m.group(1) + (m.group(2) if country == "india" else "")
        nfkc = nfkc[:m.start()] + " " + nfkc[m.end():]  # postal digits are not a house number
    house = ""
    for m in _NUMBER.finditer(nfkc):
        cand = re.sub(r"\s+", "", m.group(0))
        if cand != postal and len(cand) <= 12:
            house = cand
            break
    comps = [c for c in (_address_text(p) for p in raw.split(",")) if c]
    # a component that is a state once digits are dropped: "tx", "or 97201", "ga 30301"
    state = next((s for s in (" ".join(w for w in c.split() if not w.isdigit()) for c in reversed(comps))
                  if s in STATE_CODES), "")
    tokens = norm.split()
    street = ""
    for i, t in enumerate(tokens):
        if any(ch.isdigit() for ch in t):
            street = next((u for u in tokens[i + 1:] if u.isalpha() and len(u) > 1
                           and u not in STREET_CANON.values() and u not in STATE_CODES), "")
            break
    return norm, house, postal, state, street, "|".join(sorted(set(comps)))


_RECORD_COLUMNS = ["name_norm", "name_translit", "name_core", "name_key", "legal_present", "legal_set",
                   "non_latin", "name_sorted", "name_phonetic",
                   "addr_norm", "house", "postal", "state", "street", "components"]


def _prepare_chunk(names: list, addrs: list, countries: list) -> pd.DataFrame:
    df = pd.DataFrame([normalize_name2(n) + normalize_address2(a, c) for n, a, c in zip(names, addrs, countries)],
                      columns=_RECORD_COLUMNS)
    return df.astype({c: "str" for c in _RECORD_COLUMNS if c not in ("legal_present", "non_latin")})


def prepare_records(df: pd.DataFrame, n_jobs: int = 1, chunk: int = 50000, keep_raw: bool = True) -> pd.DataFrame:
    """Normalized, positionally indexed record table. Workers return compact string columns,
    so peak memory stays near the size of the result."""
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    countries = [normalize_country(c) for c in df["country"].tolist()]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_prepare_chunk)(names[i:i + chunk], addrs[i:i + chunk], countries[i:i + chunk])
        for i in range(0, len(names), chunk))
    del names, addrs
    out = pd.concat(parts, ignore_index=True) if parts else _prepare_chunk([], [], [])
    out.insert(0, "entity_id", df["entity_id"].to_numpy())
    out.insert(1, "country", pd.Series(countries, dtype="str"))
    if keep_raw:
        out["name_raw"] = df["business_name"].to_numpy()
        out["address_raw"] = df["business_address"].to_numpy()
    return out


# ---------------------------------------------------------------- similarity
def jaccard(a: set, b: set) -> float:
    """|a∩b|/|a∪b|; 0.0 when either side is empty (missing never creates similarity)."""
    return len(a & b) / len(a | b) if a and b else 0.0


def char3(text: str) -> set:
    return {text[i:i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else ({text} if text else set())


def edit_sim(a: str, b: str) -> float:
    """1 - Levenshtein / max(len); 0.0 when either side is empty."""
    return Levenshtein.normalized_similarity(a, b) if a and b else 0.0


# ---------------------------------------------------------------- pair features
RULE_BITS = {"exact_name": 1, "exact_address": 2, "numeric": 4, "name_prefix": 8, "address_prefix": 16,
             "name_ngram": 32, "name_phonetic": 64, "address_token": 128,
             # Phase 2.5 experimental rules (off in B0)
             "name_phonetic_token": 256, "house_locality": 512, "postal_locality": 1024,
             "state_locality": 2048, "address_signature": 4096}
_PROVENANCE = {"matched_by_exact_name_block": 1, "matched_by_exact_address_block": 2,
               "matched_by_numeric_block": 4 | 512 | 1024, "matched_by_prefix_block": 8 | 16,
               "matched_by_ngram_block": 32 | 64 | 128 | 256 | 2048 | 4096}  # token/phonetic/signature rules
PAIR_COLUMNS = ["name_norm", "name_translit", "name_core", "name_key", "legal_present", "legal_set",
                "non_latin", "name_phonetic", "addr_norm", "house", "postal", "state", "components"]
TEXT_FEATURES = [
    "name_norm_eq", "name_key_eq", "name_token_jaccard", "name_core_token_jaccard", "name_char3_jaccard",
    "name_edit_sim", "name_translit_char3_jaccard", "name_translit_edit_sim", "name_token_set_ratio",
    "name_phonetic_eq", "name_phonetic_edit_sim",
    "name_length_1", "name_length_2", "name_length_ratio", "name_length_difference",
    "name_token_count_1", "name_token_count_2", "name_token_count_difference",
    "legal_suffix_present_1", "legal_suffix_present_2", "legal_suffix_equal", "name_script_mismatch",
    "name_missing_1", "name_missing_2", "address_norm_eq", "address_token_jaccard", "address_char3_jaccard", "address_edit_sim",
    "address_component_jaccard", "address_missing_1", "address_missing_2", "address_missing_either",
    "house_number_both_present", "house_number_equal", "postal_both_present", "postal_equal",
    "state_both_present", "state_equal", "country_equal",
]
CONTEXT_FEATURES = [
    "target_source_is_s2", "target_source_is_s3", "matched_by_exact_name_block",
    "matched_by_exact_address_block", "matched_by_numeric_block", "matched_by_prefix_block",
    "matched_by_ngram_block", "number_of_blocks_that_retrieved_pair",
]
FEATURES = TEXT_FEATURES + CONTEXT_FEATURES
FEATURE_GROUPS = {  # config flag -> columns it controls; unlisted columns are always on
    ("name", "char_ngram"): ["name_char3_jaccard", "name_translit_char3_jaccard"],
    ("name", "token_jaccard"): ["name_token_jaccard", "name_core_token_jaccard", "name_token_set_ratio"],
    ("name", "edit_similarity"): ["name_edit_sim", "name_translit_edit_sim", "name_phonetic_edit_sim"],
    ("name", "normalized_exact"): ["name_norm_eq", "name_key_eq", "name_phonetic_eq"],
    ("address", "char_ngram"): ["address_char3_jaccard"],
    ("address", "token_jaccard"): ["address_token_jaccard", "address_component_jaccard"],
    ("address", "edit_similarity"): ["address_edit_sim"],
    ("address", "normalized_exact"): ["address_norm_eq"],
    ("structural", "country_equal"): ["country_equal"],
    ("structural", "name_missing"): ["name_missing_1", "name_missing_2"],
    ("structural", "address_missing"): ["address_missing_1", "address_missing_2", "address_missing_either"],
    ("structural", "house_number_equal"): ["house_number_both_present", "house_number_equal"],
    ("structural", "postal_code_equal"): ["postal_both_present", "postal_equal"],
}


def active_features(feature_cfg: dict) -> list:
    """FEATURES minus the columns of every group switched off in the config."""
    off = {c for (grp, flag), cols in FEATURE_GROUPS.items() if not feature_cfg[grp][flag] for c in cols}
    off |= set(feature_cfg.get("exclude", []))  # Phase 2.5: explicit columns left out of the model
    return [f for f in FEATURES if f not in off]


def _pair_row(a: tuple, b: tuple, ca: str, cb: str) -> list:
    (nn1, nt1, nc1, nk1, lp1, ls1, nl1, ph1, ad1, h1, p1, s1, cp1) = a
    (nn2, nt2, nc2, nk2, lp2, ls2, nl2, ph2, ad2, h2, p2, s2, cp2) = b
    tok1, tok2 = set(nn1.split()), set(nn2.split())
    atok1, atok2 = set(ad1.split()), set(ad2.split())
    len1, len2 = len(nn1), len(nn2)
    tc1, tc2 = len(nn1.split()), len(nn2.split())
    return [
        nn1 == nn2 and bool(nn1), nk1 == nk2 and bool(nk1), jaccard(tok1, tok2),
        jaccard(set(nc1.split()), set(nc2.split())), jaccard(char3(nn1), char3(nn2)),
        edit_sim(nn1, nn2), jaccard(char3(nt1), char3(nt2)), edit_sim(nt1, nt2),
        fuzz.token_set_ratio(nc1, nc2) / 100.0 if nc1 and nc2 else 0.0,
        ph1 == ph2 and bool(ph1), edit_sim(ph1, ph2),
        len1, len2, min(len1, len2) / max(len1, len2) if max(len1, len2) else 0.0, abs(len1 - len2),
        tc1, tc2, abs(tc1 - tc2),
        lp1, lp2, lp1 and lp2 and ls1 == ls2, nl1 != nl2, not nn1, not nn2,
        ad1 == ad2 and bool(ad1), jaccard(atok1, atok2), jaccard(char3(ad1), char3(ad2)), edit_sim(ad1, ad2),
        jaccard(set(cp1.split("|")) - {""}, set(cp2.split("|")) - {""}),
        not ad1, not ad2, not ad1 or not ad2,
        bool(h1 and h2), bool(h1) and h1 == h2, bool(p1 and p2), bool(p1) and p1 == p2,
        bool(s1 and s2), bool(s1) and s1 == s2, bool(ca) and ca == cb,
    ]


def _features_chunk(a_rows: list, b_rows: list, ca: list, cb: list) -> np.ndarray:
    return np.array([_pair_row(a, b, x, y) for a, b, x, y in zip(a_rows, b_rows, ca, cb)], dtype=np.float32)


def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame, target_source: str,
                     n_jobs: int = 1, chunk: int = 50000) -> pd.DataFrame:
    """Feature matrix for candidate pairs (columns s1_idx, t_idx, blocks) in FEATURES order."""
    if pairs.empty:
        return pd.DataFrame(np.zeros((0, len(FEATURES)), dtype=np.float32), columns=FEATURES)
    i1, i2 = pairs["s1_idx"].to_numpy(), pairs["t_idx"].to_numpy()
    s1_cols = [s1[c].to_numpy() for c in PAIR_COLUMNS + ["country"]]
    t_cols = [tgt[c].to_numpy() for c in PAIR_COLUMNS + ["country"]]

    def tasks():  # rows are materialized one chunk at a time, only while dispatched
        for i in range(0, len(i1), chunk):
            a, b = i1[i:i + chunk], i2[i:i + chunk]
            a_cols, b_cols = [c[a] for c in s1_cols], [c[b] for c in t_cols]
            yield delayed(_features_chunk)(list(zip(*a_cols[:-1])), list(zip(*b_cols[:-1])),
                                           a_cols[-1].tolist(), b_cols[-1].tolist())
    parts = Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs")(tasks())
    feats = pd.DataFrame(np.vstack(parts), columns=TEXT_FEATURES)
    bits = pairs["blocks"].to_numpy()
    feats["target_source_is_s2"] = np.float32(target_source == "s2")
    feats["target_source_is_s3"] = np.float32(target_source == "s3")
    for col, mask in _PROVENANCE.items():
        feats[col] = ((bits & mask) > 0).astype(np.float32)
    feats["number_of_blocks_that_retrieved_pair"] = np.array([int(b).bit_count() for b in bits], dtype=np.float32)
    return feats[FEATURES]
