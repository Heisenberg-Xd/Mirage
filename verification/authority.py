"""
verification/authority.py
=========================
Deterministic domain-based authority scoring for evidence source URLs.

Authority is a heuristic — it reflects how likely a source is to contain
accurate, peer-reviewed, or institutionally verified information.
It is NOT a full credibility model and does not call any LLM or external API.

Authority scores (raw 0–100) are normalised to [0, 1] before being used
in the composite confidence calculation.
"""

import re
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain pattern → (authority_score_0_to_100, category_label)
# Ordered from highest to lowest specificity — first match wins.
# ---------------------------------------------------------------------------
_AUTHORITY_RULES: list[tuple[str, int, str]] = [
    # Government sources
    (r"\.gov(\.[a-z]{2})?(/|$)", 100, "Government"),
    (r"\.gov\.in(/|$)",           100, "Government"),
    (r"cia\.gov",                 100, "Government"),
    (r"who\.int",                 98,  "International Organisation"),
    (r"un\.org",                  98,  "International Organisation"),
    (r"worldbank\.org",           96,  "International Organisation"),
    # Educational / research
    (r"\.edu(\.[a-z]{2})?(/|$)",  95, "Educational"),
    (r"arxiv\.org",               95, "Research"),
    (r"pubmed\.ncbi\.nlm\.nih\.gov", 95, "Research"),
    (r"scholar\.google\.",        90, "Research Index"),
    (r"researchgate\.net",        88, "Research"),
    (r"semanticscholar\.org",     88, "Research"),
    (r"jstor\.org",               88, "Research"),
    (r"springer\.com",            87, "Research Publisher"),
    (r"nature\.com",              90, "Research Publisher"),
    (r"sciencedirect\.com",       87, "Research Publisher"),
    # Encyclopedic
    (r"wikipedia\.org",           85, "Encyclopaedia"),
    (r"britannica\.com",          85, "Encyclopaedia"),
    (r"encyclopaedia\.com",       80, "Encyclopaedia"),
    # Major news outlets
    (r"reuters\.com",             82, "News Agency"),
    (r"apnews\.com",              82, "News Agency"),
    (r"bbc\.com",                 80, "Major News"),
    (r"bbc\.co\.uk",              80, "Major News"),
    (r"nytimes\.com",             80, "Major News"),
    (r"theguardian\.com",         78, "Major News"),
    (r"washingtonpost\.com",      78, "Major News"),
    (r"thehindu\.com",            76, "Major News"),
    (r"hindustantimes\.com",      74, "News"),
    (r"ndtv\.com",                74, "News"),
    (r"indiatoday\.in",           72, "News"),
    (r"timesofindia\.indiatimes\.com", 72, "News"),
    (r"cnn\.com",                 76, "Major News"),
    (r"abcnews\.go\.com",         76, "Major News"),
    (r"nbcnews\.com",             76, "Major News"),
    (r"foxnews\.com",             65, "News"),
    (r"cnbc\.com",                74, "Financial News"),
    (r"bloomberg\.com",           78, "Financial News"),
    (r"forbes\.com",              70, "Business News"),
    # Professional / tech
    (r"techcrunch\.com",          65, "Tech News"),
    (r"wired\.com",               68, "Tech News"),
    (r"arstechnica\.com",         68, "Tech News"),
    (r"stackoverflow\.com",       62, "Technical Community"),
    (r"github\.com",              60, "Open Source"),
    (r"docs\.",                   58, "Official Documentation"),
    # General / crowdsourced — lower authority
    (r"medium\.com",              45, "Blog Platform"),
    (r"substack\.com",            40, "Newsletter"),
    (r"wordpress\.com",           35, "Blog"),
    (r"blogspot\.com",            30, "Blog"),
    (r"reddit\.com",              25, "Forum"),
    (r"quora\.com",               25, "Q&A Forum"),
    (r"yahoo\.com",               30, "Portal"),
    (r"answers\.com",             20, "Q&A Site"),
    (r"wikihow\.com",             35, "How-to Site"),
]

# Score assigned when no rule matches
_DEFAULT_SCORE: int = 35
_DEFAULT_LABEL: str = "Unknown"


def get_authority_score(url: str) -> float:
    """
    Return the normalised authority score [0.0, 1.0] for a given URL.
    Matching is done against the full lowercase URL string.
    """
    if not url:
        return _DEFAULT_SCORE / 100.0

    url_lower = url.lower()
    for pattern, score, _ in _AUTHORITY_RULES:
        if re.search(pattern, url_lower):
            return score / 100.0

    return _DEFAULT_SCORE / 100.0


def get_authority_label(url: str) -> str:
    """
    Return the human-readable authority category label for a given URL.
    """
    if not url:
        return _DEFAULT_LABEL

    url_lower = url.lower()
    for pattern, _, label in _AUTHORITY_RULES:
        if re.search(pattern, url_lower):
            return label

    return _DEFAULT_LABEL


def get_authority_tier(score_normalised: float) -> str:
    """
    Map a normalised authority score to a display tier string.
    Used by the UI for badge colouring.
    """
    if score_normalised >= 0.85:
        return "high"
    elif score_normalised >= 0.55:
        return "medium"
    else:
        return "low"


def score_all_sources(evidence: list[dict]) -> list[float]:
    """
    Convenience function: return normalised authority scores for every
    source in the evidence list.
    """
    return [get_authority_score(src.get("url", "")) for src in evidence]
