"""
verification/confidence.py
==========================
Deterministic weighted composite confidence scoring.

Takes the per-claim voting results and computes a single [0, 1]
confidence score using a configurable weighted formula, then maps
it to a label via configurable thresholds.

Formula:
    score = w_sem * semantic
          + w_agr * agreement
          + w_aut * authority
          + w_sup * support_ratio
          + w_div * diversity

All components are in [0, 1].  All weights sum to 1.0.
Every result is 100% reproducible — no randomness, no LLM.
"""

import logging
from typing import Optional

from .config import (
    CONFIDENCE_WEIGHTS,
    LABEL_THRESHOLDS,
    LABEL_STRINGS,
)
from .models import ClaimVerification, VerificationResult
from .voting import compute_evidence_diversity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component computations
# ---------------------------------------------------------------------------

def _compute_semantic_score(verifications: list[ClaimVerification]) -> float:
    """
    Weighted average of best CrossEncoder scores across all claims.
    Claims with more supporting evidence are up-weighted slightly.
    """
    if not verifications:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for cv in verifications:
        # Give slightly more weight to claims that have more support
        weight = 1.0 + cv.supporting_count * 0.2
        weighted_sum += cv.best_score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _compute_agreement_score(verifications: list[ClaimVerification]) -> float:
    """
    Fraction of claims with verdict 'supported'.
    Range [0, 1].
    """
    if not verifications:
        return 0.0
    supported = sum(1 for cv in verifications if cv.verdict == "supported")
    return supported / len(verifications)


def _compute_authority_score(
    verifications: list[ClaimVerification],
    evidence: list[dict],
    authority_scores: list[float],
) -> float:
    """
    Mean normalised authority score of the sources that are supporting
    at least one claim.  If no source is supporting, use mean of all.
    """
    if not authority_scores:
        return 0.0

    # Collect indices of supporting sources
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
    else:
        return sum(authority_scores) / len(authority_scores)


def _compute_support_ratio(verifications: list[ClaimVerification]) -> float:
    """
    supported_count / (supported_count + contradicted_count)
    Returns 0.5 (neutral) if nothing is contradicted or supported.
    """
    total_sup = sum(cv.supporting_count for cv in verifications)
    total_con = sum(cv.contradicting_count for cv in verifications)
    denom = total_sup + total_con
    if denom == 0:
        return 0.5  # neutral — no strong signal either way
    return total_sup / denom


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def assign_label(score: float) -> str:
    """
    Map a composite confidence score [0, 1] to a human-readable label.
    Uses thresholds from config.py.
    """
    if score >= LABEL_THRESHOLDS["certain"]:
        return LABEL_STRINGS["certain"]
    elif score >= LABEL_THRESHOLDS["likely_certain"]:
        return LABEL_STRINGS["likely_certain"]
    elif score >= LABEL_THRESHOLDS["uncertain"]:
        return LABEL_STRINGS["uncertain"]
    else:
        return LABEL_STRINGS["needs_verification"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confidence(
    verifications: list[ClaimVerification],
    evidence: list[dict],
    authority_scores: list[float],
) -> tuple[float, str, dict[str, float]]:
    """
    Compute the final composite confidence score, label, and component breakdown.

    Args:
        verifications:    Per-claim ClaimVerification results from voting.
        evidence:         Raw evidence list (list of dicts with 'url', 'content').
        authority_scores: Normalised authority scores, one per evidence source.

    Returns:
        (confidence_score, label, components_dict)
          confidence_score: float in [0, 1]
          label:            human-readable label string
          components_dict:  dict with keys semantic, agreement, authority,
                            support_ratio, diversity — for UI display
    """
    w = CONFIDENCE_WEIGHTS

    semantic      = _compute_semantic_score(verifications)
    agreement     = _compute_agreement_score(verifications)
    authority     = _compute_authority_score(verifications, evidence, authority_scores)
    support_ratio = _compute_support_ratio(verifications)
    diversity     = compute_evidence_diversity(verifications, evidence)

    # Weighted composite
    score = (
        w["semantic"]      * semantic
        + w["agreement"]   * agreement
        + w["authority"]   * authority
        + w["support_ratio"] * support_ratio
        + w["diversity"]   * diversity
    )

    # Hard penalty: if majority of claims are contradicted, cap at 0.35
    total_claims = len(verifications)
    if total_claims > 0:
        contradicted = sum(1 for cv in verifications if cv.verdict == "contradicted")
        if contradicted / total_claims > 0.5:
            score = min(score, 0.35)

    # Hard penalty: if no evidence was found at all
    if not evidence:
        score = min(score, 0.20)

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    label = assign_label(score)

    components = {
        "semantic":      semantic,
        "agreement":     agreement,
        "authority":     authority,
        "support_ratio": support_ratio,
        "diversity":     diversity,
    }

    logger.debug(
        f"Confidence components: {components} → score={score:.4f} → label={label}"
    )

    return score, label, components
