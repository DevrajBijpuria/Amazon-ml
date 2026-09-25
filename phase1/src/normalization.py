"""Conservative, deterministic text normalization for Phase 1 analysis.

Rules applied (and nothing more):
  * NFKC Unicode normalization + casefold
  * strip diacritics from Latin letters only (é -> e); Devanagari and other
    scripts keep their combining marks, which carry meaning there
  * punctuation / symbols -> single space; whitespace collapsed and trimmed
  * names only: "&" -> "and"

Not applied on purpose: abbreviation expansion (rd -> road), legal-suffix
removal, transliteration, country code<->name mapping. Those are matching
decisions for Phase 2, not safe cleaning.
"""
import re
import unicodedata

_WS = re.compile(r"\s+")
_LATIN_END = "\u0250"  # end of Latin Extended-B; bases below this are Latin


def _is_missing(value) -> bool:
    try:
        return bool(value is None or value != value)  # NaN is the only value != itself
    except TypeError:  # pandas.NA refuses bool(); treat it as missing
        return True


def _strip_latin_marks(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch) and out and out[-1] < _LATIN_END:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _clean(value) -> str:
    if _is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = _strip_latin_marks(text)
    # keep letters, digits, whitespace and combining marks (Devanagari vowel signs)
    text = "".join(
        ch if ch.isalnum() or ch.isspace() or unicodedata.category(ch)[0] == "M" else " "
        for ch in text
    )
    return _WS.sub(" ", text).strip()


def normalize_name(value) -> str:
    """Normalize a business name; None/NaN/blank -> ''."""
    if _is_missing(value):
        return ""
    return _clean(str(value).replace("&", " and "))


def normalize_address(value) -> str:
    """Normalize a business address; line breaks and punctuation become spaces."""
    return _clean(value)


def normalize_country(value) -> str:
    """Trim, collapse whitespace and casefold a country label. No code/name mapping."""
    if _is_missing(value):
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFKC", str(value))).strip().casefold()
