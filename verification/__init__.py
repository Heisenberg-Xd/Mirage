"""
verification/__init__.py
========================
Public API for the hybrid deterministic fact verification engine.

External callers (app.py) should only import `run_verification` from here.
All internal modules are implementation details.

Pipeline order:
    1. Claim extraction      (claim_extractor.py)
    2. Authority scoring     (authority.py)
    3. CrossEncoder scoring  (cross_encoder.py)
    4. Evidence voting       (voting.py)
    5. Confidence scoring    (confidence.py)
    6. Explanation template  (templates.py)
    → VerificationResult
"""

import logging
from .models import VerificationResult, Claim, ClaimVerification
from .claim_extractor import extract_claims
from .authority import score_all_sources, get_authority_label, get_authority_tier
from .cross_encoder import load_cross_encoder, score_all_claims
from .voting import vote_all_claims
from .confidence import compute_confidence
from .templates import generate_explanation, generate_claim_summary

logger = logging.getLogger(__name__)


def run_verification(
    answer: str,
    evidence: list[dict],
    question: str = "",
) -> VerificationResult:
    """
    Run the complete hybrid verification pipeline on a Groq-generated answer.

    Args:
        answer:   The raw answer string from Groq/Llama (stored, never re-sent).
        evidence: List of evidence dicts from Tavily, each with 'content', 'url', 'title'.
        question: Original user question (unused in verification, kept for logging).

    Returns:
        VerificationResult with all fields populated.

    This function is the ONLY entry point the UI needs.
    It calls NO LLMs — all computation is deterministic.
    """
    logger.info(f"[run_verification] question='{question[:80]}', n_evidence={len(evidence)}")

    # ------------------------------------------------------------------ #
    # Step 1 — Extract atomic claims from the answer
    # ------------------------------------------------------------------ #
    claims: list[Claim] = extract_claims(answer)
    logger.info(f"Extracted {len(claims)} claim(s)")

    # Handle edge case: no claims (e.g. "I don't know.")
    if not claims:
        return _empty_result(answer, evidence)

    # ------------------------------------------------------------------ #
    # Step 2 — Score authority of each evidence source
    # ------------------------------------------------------------------ #
    authority_scores: list[float] = score_all_sources(evidence)

    # ------------------------------------------------------------------ #
    # Step 3 — CrossEncoder: score every claim against every evidence para
    # ------------------------------------------------------------------ #
    model = load_cross_encoder()
    evidence_paragraphs = [
        src.get("content", "") for src in evidence if src.get("content", "").strip()
    ]

    # Map from actual paragraph list back to original evidence indices
    # (in case some evidence items have empty content)
    para_to_evidence_idx = [
        i for i, src in enumerate(evidence) if src.get("content", "").strip()
    ]

    claims_texts = [c.text for c in claims]
    all_scores_raw = score_all_claims(claims_texts, evidence_paragraphs, model)

    # Expand scores back to full evidence list length (fill 0.0 for empty-content sources)
    n_evidence = len(evidence)
    all_scores: list[list[float]] = []
    for claim_scores_raw in all_scores_raw:
        full_scores = [0.0] * n_evidence
        for para_i, ev_i in enumerate(para_to_evidence_idx):
            if para_i < len(claim_scores_raw):
                full_scores[ev_i] = claim_scores_raw[para_i]
        all_scores.append(full_scores)

    # ------------------------------------------------------------------ #
    # Step 4 — Evidence voting per claim
    # ------------------------------------------------------------------ #
    verifications: list[ClaimVerification] = vote_all_claims(claims, evidence, all_scores)

    # ------------------------------------------------------------------ #
    # Step 5 — Composite confidence score + label
    # ------------------------------------------------------------------ #
    confidence_score, label, components = compute_confidence(
        verifications, evidence, authority_scores
    )
    confidence_pct = round(confidence_score * 100)

    # ------------------------------------------------------------------ #
    # Step 6 — Deterministic explanation
    # ------------------------------------------------------------------ #
    explanation = generate_explanation(label, verifications, evidence, components)

    # ------------------------------------------------------------------ #
    # Aggregate counts for UI
    # ------------------------------------------------------------------ #
    supported_count    = sum(1 for cv in verifications if cv.verdict == "supported")
    contradicted_count = sum(1 for cv in verifications if cv.verdict == "contradicted")
    insufficient_count = sum(1 for cv in verifications if cv.verdict == "insufficient")

    return VerificationResult(
        label=label,
        confidence_score=confidence_score,
        confidence_pct=confidence_pct,
        explanation=explanation,
        claims=claims,
        claim_verifications=verifications,
        evidence=evidence,
        authority_scores=authority_scores,
        semantic_score=components["semantic"],
        agreement_score=components["agreement"],
        authority_avg=components["authority"],
        support_ratio=components["support_ratio"],
        diversity_score=components["diversity"],
        supported_count=supported_count,
        contradicted_count=contradicted_count,
        insufficient_count=insufficient_count,
    )


def _empty_result(answer: str, evidence: list[dict]) -> VerificationResult:
    """Produce a safe default VerificationResult when no claims are extracted."""
    return VerificationResult(
        label="Needs Verification",
        confidence_score=0.0,
        confidence_pct=0,
        explanation=(
            "No verifiable factual claims could be extracted from the answer. "
            "This may indicate the answer is a refusal, disclaimer, or opinion."
        ),
        claims=[],
        claim_verifications=[],
        evidence=evidence,
        authority_scores=score_all_sources(evidence),
    )
