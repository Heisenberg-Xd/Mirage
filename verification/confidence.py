"""
verification/confidence.py
==========================
Deterministic weighted composite confidence scoring.
Now uses an 8-component formula (NLI, CE Relevance, Entity Drift, Q-Relevance, etc.)
"""

import logging

from .config import (
    CONFIDENCE_WEIGHTS,
    LABEL_THRESHOLDS,
    LABEL_STRINGS,
)
from .models import ClaimVerification
from .voting import compute_evidence_diversity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component computations
# ---------------------------------------------------------------------------

def _compute_nli_entailment_score(verifications: list[ClaimVerification]) -> float:
    """Average best NLI entailment score across claims."""
    if not verifications: return 0.0
    return sum(cv.best_nli_entailment for cv in verifications) / len(verifications)


def _compute_ce_relevance_score(verifications: list[ClaimVerification]) -> float:
    """Average CrossEncoder relevance score across claims."""
    if not verifications: return 0.0
    return sum(cv.best_relevance_score for cv in verifications) / len(verifications)


def _compute_q_relevance_score(verifications: list[ClaimVerification]) -> float:
    """Ratio of claims that are relevant to the question."""
    if not verifications: return 1.0
    relevant = sum(1 for cv in verifications if getattr(cv.claim, 'is_relevant_to_question', True))
    return relevant / len(verifications)


def _compute_authority_score(
    verifications: list[ClaimVerification],
    evidence: list[dict],
    authority_scores: list[float],
) -> float:
    """Mean normalised authority score of supporting sources."""
    if not authority_scores:
        return 0.0
    supporting_indices: set[int] = set()
    for cv in verifications:
        if cv.verdict == "supported":
            supporting_indices.add(cv.best_source_idx)
            for es in cv.evidence_scores:
                if es.verdict == "supported":
                    supporting_indices.add(es.source_idx)

    if supporting_indices:
        valid = [authority_scores[i] for i in supporting_indices if i < len(authority_scores)]
        return sum(valid) / len(valid) if valid else sum(authority_scores) / len(authority_scores)
    return sum(authority_scores) / len(authority_scores)


def _compute_support_ratio(verifications: list[ClaimVerification]) -> float:
    """supported_count / (supported_count + contradicted_count)"""
    total_sup = sum(cv.supporting_count for cv in verifications)
    total_con = sum(cv.contradicting_count for cv in verifications)
    denom = total_sup + total_con
    if denom == 0: return 0.5
    return total_sup / denom
    
def _compute_contradiction_penalty(verifications: list[ClaimVerification]) -> float:
    """1.0 if there are no contradictions, drops if there are contradictions."""
    if not verifications: return 1.0
    contradicted_claims = sum(1 for cv in verifications if cv.verdict == "contradicted")
    return 1.0 - (contradicted_claims / len(verifications))


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def assign_label(score: float) -> str:
    """Map composite confidence to human-readable label."""
    if score >= LABEL_THRESHOLDS["certain"]: return LABEL_STRINGS["certain"]
    if score >= LABEL_THRESHOLDS["likely_certain"]: return LABEL_STRINGS["likely_certain"]
    if score >= LABEL_THRESHOLDS["uncertain"]: return LABEL_STRINGS["uncertain"]
    return LABEL_STRINGS["needs_verification"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confidence(
    verifications: list[ClaimVerification],
    evidence: list[dict],
    authority_scores: list[float],
    entity_alignment_score: float = 1.0
) -> tuple[float, str, dict[str, float]]:
    
    w = CONFIDENCE_WEIGHTS

    nli_entailment = _compute_nli_entailment_score(verifications)
    ce_relevance   = _compute_ce_relevance_score(verifications)
    q_relevance    = _compute_q_relevance_score(verifications)
    authority      = _compute_authority_score(verifications, evidence, authority_scores)
    support_ratio  = _compute_support_ratio(verifications)
    diversity      = compute_evidence_diversity(verifications, evidence)
    contradiction  = _compute_contradiction_penalty(verifications)

    # Weighted composite
    score = (
        w.get("nli_entailment", 0.0) * nli_entailment
        + w.get("ce_relevance", 0.0) * ce_relevance
        + w.get("entity_align", 0.0) * entity_alignment_score
        + w.get("q_relevance", 0.0) * q_relevance
        + w.get("authority", 0.0) * authority
        + w.get("diversity", 0.0) * diversity
        + w.get("support_ratio", 0.0) * support_ratio
        + w.get("contradiction", 0.0) * contradiction
    )

    # Hard penalty: if majority of claims are contradicted
    total_claims = len(verifications)
    if total_claims > 0:
        contradicted = sum(1 for cv in verifications if cv.verdict == "contradicted")
        if contradicted / total_claims > 0.5:
            score = min(score, 0.35)

    if not evidence:
        score = min(score, 0.20)

    score = max(0.0, min(1.0, score))
    label = assign_label(score)

    components = {
        "nli_score": nli_entailment,
        "ce_score": ce_relevance,
        "entity_score": entity_alignment_score,
        "q_relevance_score": q_relevance,
        "authority_avg": authority,
        "support_ratio": support_ratio,
        "diversity_score": diversity,
        "contradiction_penalty": contradiction
    }

    return score, label, components
