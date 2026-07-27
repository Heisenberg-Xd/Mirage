"""
verification/confidence.py
==========================
Deterministic rule-based confidence scoring.
Focuses entirely on factual claims that answer the user's question.
"""

import logging
from .config import LABEL_STRINGS
from .models import ClaimVerification

logger = logging.getLogger(__name__)

def compute_confidence(
    verifications: list[ClaimVerification],
    evidence: list[dict],
    authority_scores: list[float],
    entity_alignment_score: float = 1.0,
    has_false_hallucination: bool = False
) -> tuple[float, str, list[str]]:
    """
    Compute the final rule-based confidence score, label, and logic trace.
    
    Args:
        verifications: All claim verifications.
        evidence: Raw evidence list.
        authority_scores: Source authority scores.
        entity_alignment_score: Score from entity alignment.
        has_false_hallucination: True if an irrelevant claim is contradicted.
        
    Returns:
        (confidence_score, label, logic_trace)
    """
    logic_trace = []
    
    # 1. Filter to only relevant claims for the base label
    relevant_verifs = [cv for cv in verifications if getattr(cv.claim, 'is_relevant_to_question', True)]
    n_relevant = len(relevant_verifs)
    n_rel_sup = sum(1 for cv in relevant_verifs if cv.verdict == "supported")
    
    # 2. Base Rules
    if n_relevant == 0:
        base_score = 0.25
        label = LABEL_STRINGS["needs_verification"]
        logic_trace.append("No relevant claims found to answer the question → Base: Needs Verification (25%)")
    elif n_rel_sup == n_relevant:
        base_score = 0.98
        label = LABEL_STRINGS["certain"]
        logic_trace.append(f"All relevant claims ({n_rel_sup}/{n_relevant}) supported → Base: Certain (98%)")
    elif n_rel_sup > 0:
        base_score = 0.70
        label = LABEL_STRINGS["uncertain"]
        logic_trace.append(f"Some relevant claims ({n_rel_sup}/{n_relevant}) supported → Base: Uncertain (70%)")
    else:
        base_score = 0.25
        label = LABEL_STRINGS["needs_verification"]
        logic_trace.append(f"No relevant claims ({0}/{n_relevant}) supported → Base: Needs Verification (25%)")
        
    # 3. Penalties
    final_score = base_score
    
    # Penalty: Irrelevant False Information (Hallucination)
    if has_false_hallucination:
        final_score -= 0.15
        logic_trace.append("False extra context introduced (Contradicted irrelevant claim) → Penalty: -15%")
        
    # Penalty: Entity Drift (if the answer drifts from the question subjects significantly)
    if entity_alignment_score < 0.6:
        final_score -= 0.05
        logic_trace.append(f"Entity drift detected (Score: {round(entity_alignment_score*100)}%) → Penalty: -5%")
        
    # Penalty: No Evidence at all
    if not evidence:
        final_score = min(final_score, 0.20)
        logic_trace.append("No web evidence found → Cap at 20%")
        
    # Penalty: Overall weak authority
    if authority_scores:
        avg_auth = sum(authority_scores) / len(authority_scores)
        if avg_auth < 0.4:
            final_score -= 0.05
            logic_trace.append(f"Weak evidence authority (Avg: {round(avg_auth*100)}) → Penalty: -5%")

    # Clamp to [0, 1]
    final_score = max(0.0, min(1.0, final_score))
    
    # Note: We do NOT change the label if penalties bring the score down, 
    # to maintain the strict definition that label depends on relevant claims.
    # We just adjust the raw confidence percentage.
    # Exception: if score drops below 0.8, and it was "Certain", maybe drop to "Likely Certain"
    if label == LABEL_STRINGS["certain"] and final_score < 0.9:
        label = LABEL_STRINGS["likely_certain"]
        logic_trace.append("Penalties lowered Certainty below 90% → Label adjusted to Likely Certain")

    return final_score, label, logic_trace
