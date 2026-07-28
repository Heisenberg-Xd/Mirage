"""
verification/confidence.py
==========================
Deterministic rule-based confidence scoring for hallucination detection.

The confidence score represents confidence IN the hallucination assessment:
  - Not Hallucinating  → all relevant claims are supported by evidence (95–100%)
  - Hallucinating      → one or more relevant claims are contradicted   (10–30%)
  - Cannot Verify      → insufficient evidence to support or contradict (40–60%)

Only claims that directly answer the user's question are considered.
Conversational filler is ignored unless it introduces a contradictory fact.
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
    has_false_hallucination: bool = False,
) -> tuple[float, str, list[str]]:
    """
    Compute the final hallucination label, confidence score, and logic trace.

    Args:
        verifications: All claim verifications (relevant + irrelevant).
        evidence: Raw evidence list.
        authority_scores: Source authority scores.
        entity_alignment_score: Score from entity alignment (unused for label,
            kept for the logic trace and minor confidence adjustment).
        has_false_hallucination: True if an irrelevant/filler claim is
            contradicted — does NOT change the label, but is noted in trace.

    Returns:
        (confidence_score, label, logic_trace)
    """
    logic_trace: list[str] = []

    # ------------------------------------------------------------------
    # 1. Separate relevant claims from conversational filler
    # ------------------------------------------------------------------
    relevant_verifs = [
        cv for cv in verifications
        if getattr(cv.claim, "is_relevant_to_question", True)
    ]
    n_relevant     = len(relevant_verifs)
    n_supported    = sum(1 for cv in relevant_verifs if cv.verdict == "supported")
    n_contradicted = sum(1 for cv in relevant_verifs if cv.verdict == "contradicted")
    n_insufficient = sum(1 for cv in relevant_verifs if cv.verdict == "insufficient")

    logic_trace.append(
        f"Relevant claims: {n_relevant} total — "
        f"{n_supported} supported, {n_contradicted} contradicted, "
        f"{n_insufficient} insufficient"
    )

    # ------------------------------------------------------------------
    # 2. No evidence at all → Cannot Verify
    # ------------------------------------------------------------------
    if not evidence:
        label = LABEL_STRINGS["cannot_verify"]
        score = 0.45
        logic_trace.append("No web evidence found → Cannot Verify (45%)")
        return score, label, logic_trace

    # ------------------------------------------------------------------
    # 3. No relevant claims extracted → Cannot Verify
    # ------------------------------------------------------------------
    if n_relevant == 0:
        label = LABEL_STRINGS["cannot_verify"]
        score = 0.45
        logic_trace.append(
            "No relevant claims found to answer the question → Cannot Verify (45%)"
        )
        return score, label, logic_trace

    # ------------------------------------------------------------------
    # 4. Core hallucination decision tree
    # ------------------------------------------------------------------
    if n_contradicted > 0:
        # Any contradicted relevant claim → Hallucinating
        label = LABEL_STRINGS["hallucinating"]
        # Confidence in the Hallucinating assessment scales with how many
        # claims are contradicted (more contradicted = more confident it's hallucinating)
        contradiction_ratio = n_contradicted / n_relevant
        base_score = 0.15 + contradiction_ratio * 0.15   # 15–30%
        logic_trace.append(
            f"{n_contradicted}/{n_relevant} relevant claim(s) contradicted by evidence "
            f"→ Hallucinating (base {round(base_score * 100)}%)"
        )
    elif n_supported == n_relevant:
        # All relevant claims supported → Not Hallucinating
        label = LABEL_STRINGS["not_hallucinating"]
        base_score = 0.98
        logic_trace.append(
            f"All relevant claims ({n_supported}/{n_relevant}) supported by evidence "
            f"→ Not Hallucinating (base 98%)"
        )
    else:
        # Mix of supported + insufficient, or all insufficient → Cannot Verify
        label = LABEL_STRINGS["cannot_verify"]
        if n_supported > 0:
            # Some support, but not all — partial evidence
            support_ratio = n_supported / n_relevant
            base_score = 0.40 + support_ratio * 0.15   # 40–55%
            logic_trace.append(
                f"Only {n_supported}/{n_relevant} relevant claims supported, "
                f"{n_insufficient} insufficient → Cannot Verify (base {round(base_score * 100)}%)"
            )
        else:
            # No support at all, no contradiction either
            base_score = 0.48
            logic_trace.append(
                f"No relevant claims ({n_insufficient}/{n_relevant}) have supporting "
                f"or contradicting evidence → Cannot Verify (base 48%)"
            )

    # ------------------------------------------------------------------
    # 5. Confidence adjustments (do NOT change the label)
    # ------------------------------------------------------------------
    final_score = base_score

    # Boost Not Hallucinating confidence if authority is high
    if label == LABEL_STRINGS["not_hallucinating"] and authority_scores:
        avg_auth = sum(authority_scores) / len(authority_scores)
        if avg_auth >= 0.7:
            final_score = min(1.0, final_score + 0.02)
            logic_trace.append(
                f"High-authority sources (avg {round(avg_auth * 100)}%) → +2% confidence"
            )
        elif avg_auth < 0.4:
            final_score -= 0.03
            logic_trace.append(
                f"Weak evidence authority (avg {round(avg_auth * 100)}%) → -3% confidence"
            )

    # Minor penalty if filler/irrelevant content is contradicted
    # (noted in trace but does NOT change the hallucination label)
    if has_false_hallucination and label == LABEL_STRINGS["not_hallucinating"]:
        final_score -= 0.02
        logic_trace.append(
            "Contradicted filler content detected (not the direct answer) → -2% note"
        )

    # ------------------------------------------------------------------
    # 6. Clamp to valid range
    # ------------------------------------------------------------------
    final_score = max(0.05, min(1.0, final_score))
    logic_trace.append(f"Final confidence: {round(final_score * 100)}% → {label}")

    return final_score, label, logic_trace
