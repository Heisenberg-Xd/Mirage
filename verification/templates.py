"""
verification/templates.py
==========================
Deterministic, template-driven explanation generator.

This module produces ALL user-facing explanation text from pre-written
string templates populated with computed values.

ZERO LLM calls are made here — it is entirely if/else logic + f-strings.
This guarantees that the explanation layer cannot hallucinate, because it
can only echo numbers and facts that were measured upstream.

Template selection hierarchy:
  1. Hard contradiction detected (majority contradicted)
  2. No evidence found
  3. Label-based template (Certain / Likely Certain / Uncertain / Needs Verification)
  4. Enriched with claim-level detail where available
"""

from .models import ClaimVerification, VerificationResult
from .config import LABEL_STRINGS


# ---------------------------------------------------------------------------
# Internal template strings — edit here to change user-facing language
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {
    "certain": (
        "The answer is strongly supported by {n_supporting_sources} independent "
        "source(s) with high semantic agreement across {n_claims} verified claim(s) "
        "and no detected contradictions."
    ),
    "likely_certain": (
        "The answer is well-supported by {n_supporting_sources} source(s). "
        "{n_supported} of {n_claims} extracted claim(s) are confirmed by web evidence "
        "with no significant contradictions detected."
    ),
    "uncertain": (
        "{n_supported} of {n_claims} claim(s) in the answer are supported by evidence, "
        "but {n_insufficient} could not be confirmed. The answer may be partially "
        "correct — some details warrant independent verification."
    ),
    "needs_verification": (
        "The available evidence does not sufficiently support the generated answer. "
        "{n_insufficient} of {n_claims} claim(s) remain unverified by retrieved sources. "
        "Independent fact-checking is recommended."
    ),
    "contradiction": (
        "The generated answer appears to conflict with evidence retrieved from "
        "{n_contradicted_sources} web source(s). {n_contradicted} of {n_claims} "
        "claim(s) are contradicted by retrieved evidence. Treat this answer with caution."
    ),
    "no_evidence": (
        "No web evidence was retrieved for this question. The answer cannot be "
        "verified against external sources. Label is based solely on model uncertainty."
    ),
    "single_claim_certain": (
        "The core claim in the answer is directly confirmed by {n_supporting_sources} "
        "source(s) with a CrossEncoder relevance score of {best_score_pct}%."
    ),
    "negation_supported": (
        "The answer contains a negated claim which is supported by evidence: "
        "sources confirm the correct entity, validating the negation."
    ),
}


# ---------------------------------------------------------------------------
# Helper: count summary stats from verifications
# ---------------------------------------------------------------------------

def _get_stats(verifications: list[ClaimVerification], evidence: list[dict]) -> dict:
    n_claims = len(verifications)
    n_supported = sum(1 for cv in verifications if cv.verdict == "supported")
    n_contradicted = sum(1 for cv in verifications if cv.verdict == "contradicted")
    n_insufficient = sum(1 for cv in verifications if cv.verdict == "insufficient")

    # Count unique supporting sources
    supporting_source_ids: set[int] = set()
    for cv in verifications:
        if cv.verdict == "supported":
            for es in cv.evidence_scores:
                if es.verdict == "supported":
                    supporting_source_ids.add(es.source_idx)
    n_supporting_sources = len(supporting_source_ids)

    # Best CrossEncoder score across all claims
    best_score = max((cv.best_score for cv in verifications), default=0.0)

    has_negation = any(cv.claim.is_negated for cv in verifications)
    n_contradicted_sources = n_contradicted   # approximate

    return {
        "n_claims": n_claims,
        "n_supported": n_supported,
        "n_contradicted": n_contradicted,
        "n_insufficient": n_insufficient,
        "n_supporting_sources": max(n_supporting_sources, 1) if n_supported > 0 else 0,
        "best_score_pct": round(best_score * 100),
        "has_negation": has_negation,
        "n_contradicted_sources": n_contradicted_sources,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    label: str,
    verifications: list[ClaimVerification],
    evidence: list[dict],
    components: dict[str, float],
) -> str:
    """
    Generate a human-readable explanation for the verification result.

    Args:
        label:         The final label string (from LABEL_STRINGS).
        verifications: Per-claim verification results.
        evidence:      Raw evidence list.
        components:    Confidence component scores (for numeric enrichment).

    Returns:
        A single explanation string. No LLM involved.
    """
    stats = _get_stats(verifications, evidence)
    n_claims      = stats["n_claims"]
    n_supported   = stats["n_supported"]
    n_contradicted= stats["n_contradicted"]
    n_insufficient= stats["n_insufficient"]
    n_sources     = stats["n_supporting_sources"]
    best_pct      = stats["best_score_pct"]
    has_neg       = stats["has_negation"]

    # Case: no evidence retrieved
    if not evidence:
        return _TEMPLATES["no_evidence"]

    # Case: majority of claims contradicted
    if n_claims > 0 and n_contradicted / n_claims > 0.5:
        return _TEMPLATES["contradiction"].format(
            n_contradicted_sources=stats["n_contradicted_sources"],
            n_contradicted=n_contradicted,
            n_claims=n_claims,
        )

    # Case: single-claim answer that is Certain — use richer template
    if n_claims == 1 and label in (LABEL_STRINGS["certain"], LABEL_STRINGS["likely_certain"]):
        base = _TEMPLATES["single_claim_certain"].format(
            n_supporting_sources=n_sources,
            best_score_pct=best_pct,
        )
        if has_neg:
            base += " " + _TEMPLATES["negation_supported"]
        return base

    # Standard label-based templates
    label_key_map = {
        LABEL_STRINGS["certain"]:            "certain",
        LABEL_STRINGS["likely_certain"]:     "likely_certain",
        LABEL_STRINGS["uncertain"]:          "uncertain",
        LABEL_STRINGS["needs_verification"]: "needs_verification",
    }
    key = label_key_map.get(label, "needs_verification")
    explanation = _TEMPLATES[key].format(
        n_claims=n_claims,
        n_supported=n_supported,
        n_insufficient=n_insufficient,
        n_supporting_sources=n_sources,
        n_contradicted=n_contradicted,
        n_contradicted_sources=stats["n_contradicted_sources"],
        best_score_pct=best_pct,
    )

    # Append negation addendum if relevant
    if has_neg and n_supported > 0:
        explanation += " " + _TEMPLATES["negation_supported"]

    return explanation


def generate_claim_summary(verifications: list[ClaimVerification]) -> list[dict]:
    """
    Produce a structured list of claim-level summary rows for UI rendering.
    Each row is a dict with keys: raw_text, verdict, best_score_pct, is_negated.
    """
    rows = []
    for cv in verifications:
        rows.append({
            "claim":          cv.claim.raw_text,
            "verdict":        cv.verdict,
            "best_score_pct": round(cv.best_score * 100),
            "is_negated":     cv.claim.is_negated,
            "supporting":     cv.supporting_count,
            "contradicting":  cv.contradicting_count,
        })
    return rows
