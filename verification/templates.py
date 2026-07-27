"""
verification/templates.py
==========================
Deterministic, template-driven explanation generator.

This module produces ALL user-facing explanation text from pre-written
string templates populated with computed values.

ZERO LLM calls are made here — it is entirely if/else logic + f-strings.
"""

from .models import ClaimVerification, VerificationResult
from .config import LABEL_STRINGS


# ---------------------------------------------------------------------------
# Internal template strings — edit here to change user-facing language
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {
    "certain": (
        "The primary claim answering the user's question is strongly supported by {n_supporting_sources} "
        "authoritative source(s) with no detected contradictions."
    ),
    "likely_certain": (
        "The answer is well-supported by {n_supporting_sources} source(s). "
        "{n_supported} of {n_claims} relevant claim(s) are confirmed by web evidence "
        "with no significant contradictions detected."
    ),
    "uncertain": (
        "{n_supported} of {n_claims} relevant claim(s) in the answer are supported by evidence, "
        "but {n_insufficient} could not be confirmed. The answer may be partially "
        "correct — some details warrant independent verification."
    ),
    "needs_verification": (
        "The available evidence does not sufficiently support the generated answer. "
        "{n_insufficient} of {n_claims} relevant claim(s) remain unverified by retrieved sources. "
        "Independent fact-checking is recommended."
    ),
    "contradiction": (
        "The generated answer appears to conflict with evidence retrieved from "
        "{n_contradicted_sources} web source(s). {n_contradicted} of {n_claims} "
        "relevant claim(s) are contradicted by retrieved evidence. Treat this answer with caution."
    ),
    "no_evidence": (
        "No web evidence was retrieved for this question. The answer cannot be "
        "verified against external sources. Label is based solely on model uncertainty."
    ),
    "single_claim_certain": (
        "The core claim in the answer is directly confirmed by {n_supporting_sources} "
        "source(s) with high NLI entailment."
    ),
    "negation_supported": (
        "The answer contains a negated claim which is supported by evidence: "
        "sources confirm the correct entity, validating the negation."
    ),
    "entity_drift": (
        "The generated answer introduced entities not present in the user's question."
    ),
    "hallucination": (
        "The answer contains unrelated factual statements which reduce confidence."
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

    supporting_source_ids: set[int] = set()
    for cv in verifications:
        if cv.verdict == "supported":
            for es in cv.evidence_scores:
                if es.verdict == "supported":
                    supporting_source_ids.add(es.source_idx)
    n_supporting_sources = len(supporting_source_ids)

    best_pct = max((cv.best_relevance_score for cv in verifications), default=0.0)

    has_negation = any(cv.claim.is_negated for cv in verifications)
    n_contradicted_sources = n_contradicted

    return {
        "n_claims": n_claims,
        "n_supported": n_supported,
        "n_contradicted": n_contradicted,
        "n_insufficient": n_insufficient,
        "n_supporting_sources": max(n_supporting_sources, 1) if n_supported > 0 else 0,
        "best_score_pct": round(best_pct * 100),
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
    entity_drift_detected: bool,
    has_hallucinated_claims: bool,
) -> str:
    """
    Generate a human-readable explanation for the verification result.
    """
    # Only count claims that were actually verified (relevant to question)
    relevant_verifications = [cv for cv in verifications if getattr(cv.claim, 'is_relevant_to_question', True)]
    
    stats = _get_stats(relevant_verifications, evidence)
    n_claims      = stats["n_claims"]
    n_supported   = stats["n_supported"]
    n_contradicted= stats["n_contradicted"]
    n_insufficient= stats["n_insufficient"]
    n_sources     = stats["n_supporting_sources"]
    best_pct      = stats["best_score_pct"]
    has_neg       = stats["has_negation"]

    # Base Explanation
    explanation = ""

    if not evidence:
        explanation = _TEMPLATES["no_evidence"]
    elif n_claims > 0 and n_contradicted / n_claims > 0.5:
        explanation = _TEMPLATES["contradiction"].format(
            n_contradicted_sources=stats["n_contradicted_sources"],
            n_contradicted=n_contradicted,
            n_claims=n_claims,
        )
    elif n_claims == 1 and label in (LABEL_STRINGS["certain"], LABEL_STRINGS["likely_certain"]):
        base = _TEMPLATES["single_claim_certain"].format(
            n_supporting_sources=n_sources,
            best_score_pct=best_pct,
        )
        if has_neg:
            base += " " + _TEMPLATES["negation_supported"]
        explanation = base
    else:
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
        if has_neg and n_supported > 0:
            explanation += " " + _TEMPLATES["negation_supported"]

    # Append drift/hallucination addendums
    if entity_drift_detected:
        explanation += " " + _TEMPLATES["entity_drift"]
    if has_hallucinated_claims:
        # Avoid saying it reduced confidence if it's still 'Certain'
        if label == LABEL_STRINGS["certain"]:
            explanation += " Additional unrelated content was detected but did not affect the correctness of the answer."
        else:
            explanation += " " + _TEMPLATES["hallucination"]

    return explanation


def generate_claim_summary(verifications: list[ClaimVerification]) -> list[dict]:
    """
    Produce a structured list of claim-level summary rows for UI rendering.
    """
    rows = []
    for cv in verifications:
        is_relevant = getattr(cv.claim, 'is_relevant_to_question', True)
        
        rows.append({
            "claim":          cv.claim.raw_text,
            "verdict":        cv.verdict if is_relevant else "ignored",
            "best_score_pct": round(cv.best_relevance_score * 100),
            "best_nli_pct":   round(cv.best_nli_entailment * 100),
            "is_negated":     cv.claim.is_negated,
            "is_relevant":    is_relevant,
            "supporting":     cv.supporting_count,
            "contradicting":  cv.contradicting_count,
        })
    return rows
