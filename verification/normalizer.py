"""
verification/normalizer.py
==========================
Deterministic text normalization applied to both extracted claims and
evidence snippets before comparison.

No LLM involved — all transformations are pure string/regex operations.

Normalizations performed:
  1. Lowercase
  2. Title/role canonicalization  (PM → prime minister, Dr. → doctor, etc.)
  3. Number-word → digit          (twenty-one → 21)
  4. Date format normalization    (1st Jan 2020 → 1 january 2020)
  5. Whitespace collapse
  6. Punctuation removal          (keep hyphens inside words)
"""

import re
import string
from typing import Optional


# ---------------------------------------------------------------------------
# Title / role synonyms — mapped to a canonical form for comparison
# ---------------------------------------------------------------------------
_TITLE_MAP: dict[str, str] = {
    r"\bPM\b":              "prime minister",
    r"\bP\.M\.\b":         "prime minister",
    r"\bCM\b":              "chief minister",
    r"\bPres\.\b":          "president",
    r"\bVP\b":              "vice president",
    r"\bV\.P\.\b":          "vice president",
    r"\bSec\.\b":           "secretary",
    r"\bSecy\b":            "secretary",
    r"\bGen\.\b":           "general",
    r"\bCol\.\b":           "colonel",
    r"\bSgt\.\b":           "sergeant",
    r"\bCpt\.\b":           "captain",
    r"\bDr\b\.?":           "doctor",
    r"\bProf\b\.?":         "professor",
    r"\bMr\b\.?":           "mister",
    r"\bMrs\b\.?":          "missus",
    r"\bMs\b\.?":           "miss",
    r"\bSt\.\b":            "saint",
    r"\bJr\b\.?":           "junior",
    r"\bSr\b\.?":           "senior",
    r"\bLtd\b\.?":          "limited",
    r"\bInc\b\.?":          "incorporated",
    r"\bCorp\b\.?":         "corporation",
    r"\bUSA\b":             "united states",
    r"\bU\.S\.A\.\b":      "united states",
    r"\bUS\b":              "united states",
    r"\bU\.S\.\b":         "united states",
    r"\bUK\b":              "united kingdom",
    r"\bU\.K\.\b":         "united kingdom",
    r"\bUAE\b":             "united arab emirates",
    r"\bINDIA\b":           "india",
}

# ---------------------------------------------------------------------------
# Number words → digits (covers 0–99)
# ---------------------------------------------------------------------------
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_NUMBER_WORD_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Ordinal → digit  (1st → 1, 2nd → 2, etc.)
# ---------------------------------------------------------------------------
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Month names for date normalization
# ---------------------------------------------------------------------------
_MONTH_MAP: dict[str, str] = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "may": "may", "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "oct": "october", "nov": "november", "dec": "december",
}


def _replace_number_words(text: str) -> str:
    """Replace written-out number words with digits."""
    def _repl(m: re.Match) -> str:
        tens_word = m.group(1).lower()
        ones_word = m.group(2).lower() if m.group(2) else None
        val = _ONES.get(tens_word, 0)
        if ones_word:
            val += _ONES.get(ones_word, 0)
        return str(val)
    return _NUMBER_WORD_RE.sub(_repl, text)


def _normalize_dates(text: str) -> str:
    """Normalize ordinals and abbreviated month names."""
    # 1st → 1, 2nd → 2
    text = _ORDINAL_RE.sub(r"\1", text)
    # Jan → january, Feb → february, etc.
    for abbr, full in _MONTH_MAP.items():
        text = re.sub(rf"\b{abbr}\b\.?", full, text, flags=re.IGNORECASE)
    return text


def _apply_title_map(text: str) -> str:
    """Expand abbreviations and title variants to canonical forms."""
    for pattern, replacement in _TITLE_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_punctuation(text: str) -> str:
    """
    Remove punctuation except hyphens that appear between word characters
    (to preserve compound words like 'forty-two').
    """
    # Replace all punctuation except word-internal hyphens with a space
    result = []
    for i, ch in enumerate(text):
        if ch in string.punctuation:
            # Keep hyphen between two word characters
            if ch == "-" and 0 < i < len(text) - 1:
                if text[i - 1].isalnum() and text[i + 1].isalnum():
                    result.append(ch)
                    continue
            result.append(" ")
        else:
            result.append(ch)
    return "".join(result)


def normalize(text: str) -> str:
    """
    Full normalization pipeline.
    Applies: title-map → number-words → dates → lowercase → punctuation → whitespace.
    Returns the normalized string.
    """
    if not text:
        return ""
    text = _apply_title_map(text)
    text = _replace_number_words(text)
    text = _normalize_dates(text)
    text = text.lower()
    text = _remove_punctuation(text)
    text = _collapse_whitespace(text)
    return text
