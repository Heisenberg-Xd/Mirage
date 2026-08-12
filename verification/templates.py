"""
verification/templates.py
==========================
Deterministic, template-driven explanation generator for hallucination detection.

This module produces ALL user-facing explanation text from pre-written
string templates populated with computed values.

Outputs one of three states:
  - Not Hallucinating  → all relevant claims supported by evidence
  - Hallucinating      → one or more relevant claims contradicted by evidence
  - Cannot Verify      → insufficient evidence to support or contradict claims

ZERO LLM calls are made here — it is entirely if/else logic + f-strings.
"""

from .models import ClaimVerification, VerificationResult
from .config import LABEL_STRINGS


# ---------------------------------------------------------------------------
# Internal template strings — edit here to change user-facing language
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {
    "not_hallucinating": (
        "All {n_claims} relevant claim(s) directly answering the user's question are "
        "supported by {n_supporting_sources} authoritative web source(s) with no "
        "detected contradictions. The AI did not hallucinate on this answer."
    ),
    "not_hallucinating_single": (
        "The core claim in the answer is directly confirmed by {n_supporting_sources} "
        "web source(s) with high NLI entailment. No hallucination detected."
    ),
    "hallucinating": (
        "{n_contradicted} of {n_claims} relevant claim(s) that directly answer the "
        "user's question are contradicted by {n_contradicted_sources} web source(s). "
        "The AI appears to have hallucinated factual information in its response."
    ),
    "cannot_verify_partial": (
        "Only {n_supported} of {n_claims} relevant claim(s) are supported by evidence; "
        "{n_insufficient} could not be confirmed or contradicted by available sources. "
        "There is insufficient evidence to make a definitive hallucination determination."
    ),
    "cannot_verify_none": (
        "None of the {n_claims} relevant claim(s) extracted from the answer could be "
        "confirmed or contradicted by the retrieved evidence. "
        "Hallucination status cannot be determined from available sources."
    ),
    "cannot_verify_no_evidence": (
        "No web evidence was retrieved for this question. "
        "The hallucination status of the answer cannot be assessed against external sources."
    ),
    "cannot_verify_no_claims": (
        "No factual claims directly answering the user's question could be extracted "
        "from the response. Hallucination status cannot be determined."
    ),
    "entity_drift": (
        "The model answered about a different subject than what was asked, "
        "so hallucination detection could not be applied to the actual question."
    ),
    "negation_note": (
        "The answer contains a negated claim which is supported by evidence: "
        "sources confirm the correct entity, validating the negation."
    ),
    "filler_contradiction_note": (
        "The answer contains additional conversational content that is contradicted "
        "by evidence, but this did not directly answer the user's question and is not "
        "counted as hallucination of the answer itself."
    ),
}


# ---------------------------------------------------------------------------
# Helper: count summary stats from verifications
# ---------------------------------------------------------------------------

def _get_stats(verifications: list[ClaimVerification], evidence: list[dict]) -> dict:
    n_claims       = len(verifications)
    n_supported    = sum(1 for cv in verifications if cv.verdict == "supported")
    n_contradicted = sum(1 for cv in verifications if cv.verdict == "contradicted")
    n_insufficient = sum(1 for cv in verifications if cv.verdict == "insufficient")

    supporting_source_ids: set[int] = set()
    for cv in verifications:
        if cv.verdict == "supported":
            for es in cv.evidence_scores:
                if es.verdict == "supported":
                    supporting_source_ids.add(es.source_idx)
    n_supporting_sources = len(supporting_source_ids)

    best_pct = max((cv.best_nli_entailment for cv in verifications), default=0.0)

    has_negation = any(cv.claim.is_negated for cv in verifications)
    n_contradicted_sources = n_contradicted  # one source per contradicted claim (conservative)

    return {
        "n_claims":              n_claims,
        "n_supported":           n_supported,
        "n_contradicted":        n_contradicted,
        "n_insufficient":        n_insufficient,
        "n_supporting_sources":  max(n_supporting_sources, 1) if n_supported > 0 else 0,
        "best_score_pct":        round(best_pct * 100),
        "has_negation":          has_negation,
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
    Generate a human-readable hallucination-status explanation.
    """
    # Only use claims that directly answer the question
    relevant_verifications = [
        cv for cv in verifications
        if getattr(cv.claim, "is_relevant_to_question", True)
    ]

    stats          = _get_stats(relevant_verifications, evidence)
    n_claims       = stats["n_claims"]
    n_supported    = stats["n_supported"]
    n_contradicted = stats["n_contradicted"]
    n_insufficient = stats["n_insufficient"]
    n_sources      = stats["n_supporting_sources"]
    has_neg        = stats["has_negation"]

    # ---- No evidence at all ----
    if not evidence:
        return _TEMPLATES["cannot_verify_no_evidence"]

    # ---- Entity drift ----
    if entity_drift_detected:
        return _TEMPLATES["entity_drift"]

    # ---- No relevant claims ----
    if n_claims == 0:
        return _TEMPLATES["cannot_verify_no_claims"]

    # ---- Hallucinating ----
    if label == LABEL_STRINGS["hallucinating"]:
        explanation = _TEMPLATES["hallucinating"].format(
            n_contradicted=n_contradicted,
            n_claims=n_claims,
            n_contradicted_sources=stats["n_contradicted_sources"],
        )

    # ---- Not Hallucinating ----
    elif label == LABEL_STRINGS["not_hallucinating"]:
        if n_claims == 1:
            explanation = _TEMPLATES["not_hallucinating_single"].format(
                n_supporting_sources=n_sources,
            )
        else:
            explanation = _TEMPLATES["not_hallucinating"].format(
                n_claims=n_claims,
                n_supporting_sources=n_sources,
            )
        if has_neg:
            explanation += " " + _TEMPLATES["negation_note"]

    # ---- Cannot Verify ----
    else:
        if n_supported > 0:
            explanation = _TEMPLATES["cannot_verify_partial"].format(
                n_supported=n_supported,
                n_claims=n_claims,
                n_insufficient=n_insufficient,
            )
        else:
            explanation = _TEMPLATES["cannot_verify_none"].format(
                n_claims=n_claims,
            )

    # Append filler-contradiction note (does not change the label)
    if has_hallucinated_claims and label == LABEL_STRINGS["not_hallucinating"]:
        explanation += " " + _TEMPLATES["filler_contradiction_note"]

    return explanation


def generate_claim_summary(verifications: list[ClaimVerification]) -> list[dict]:
    """
    Produce a structured list of claim-level summary rows for UI rendering.
    """
    rows = []
    for cv in verifications:
        is_relevant = getattr(cv.claim, "is_relevant_to_question", True)

        rows.append({
            "claim":          cv.claim.raw_text,
            "verdict":        cv.verdict if is_relevant else "ignored",
            "best_score_pct": round(cv.best_nli_entailment * 100),
            "best_nli_pct":   round(cv.best_nli_entailment * 100),
            "is_negated":     cv.claim.is_negated,
            "is_relevant":    is_relevant,
            "supporting":     cv.supporting_count,
            "contradicting":  cv.contradicting_count,
        })
    return rows
