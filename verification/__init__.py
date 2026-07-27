"""
verification/__init__.py
========================
Public API for the hybrid deterministic fact verification engine.

Pipeline order:
    1. Question Analysis & Entity Extraction
    2. Answer Entity Extraction & Alignment
    3. Claim extraction
    4. Claim filtering (relevance to question)
    5. Authority scoring
    6. CrossEncoder scoring (relevance)
    7. NLI scoring (entailment/contradiction)
    8. Evidence voting
    9. Confidence scoring
    10. Explanation template
    → VerificationResult
"""

import logging
from .models import VerificationResult, Claim, ClaimVerification
from .question_parser import parse_question
from .entity_extractor import extract_entities
from .entity_alignment import check_entity_alignment
from .claim_extractor import extract_claims
from .claim_filter import filter_claims_by_question
from .authority import score_all_sources, get_authority_label, get_authority_tier
from .cross_encoder import load_cross_encoder, score_all_claims
from .nli import load_nli_model, score_all_nli
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
    Run the complete deterministic NLI-based fact verification pipeline.
    """
    logger.info(f"[run_verification] question='{question[:80]}', n_evidence={len(evidence)}")

    # ------------------------------------------------------------------ #
    # Steps 1-3: Entity Alignment & Question parsing
    # ------------------------------------------------------------------ #
    intent_data = parse_question(question)
    q_entities = extract_entities(question)
    a_entities = extract_entities(answer)
    
    drift_detected, entity_alignment_score = check_entity_alignment(q_entities, a_entities)

    # ------------------------------------------------------------------ #
    # Step 4: Extract atomic claims from the answer
    # ------------------------------------------------------------------ #
    claims: list[Claim] = extract_claims(answer)
    logger.info(f"Extracted {len(claims)} claim(s)")

    if not claims:
        return _empty_result(answer, evidence)

    # ------------------------------------------------------------------ #
    # Step 5: Question Relevance Filter
    # ------------------------------------------------------------------ #
    claims, has_hallucinated_claims = filter_claims_by_question(claims, question)

    # ------------------------------------------------------------------ #
    # Step 6: Score authority of each evidence source
    # ------------------------------------------------------------------ #
    authority_scores: list[float] = score_all_sources(evidence)

    # ------------------------------------------------------------------ #
    # Step 7: CrossEncoder & NLI Inference
    # ------------------------------------------------------------------ #
    ce_model = load_cross_encoder()
    nli_model = load_nli_model()
    
    evidence_paragraphs = [
        src.get("content", "") for src in evidence if src.get("content", "").strip()
    ]

    para_to_evidence_idx = [
        i for i, src in enumerate(evidence) if src.get("content", "").strip()
    ]

    claims_texts = [c.text for c in claims]
    
    # CE Relevance
    all_ce_scores_raw = score_all_claims(claims_texts, evidence_paragraphs, ce_model)
    # NLI Probabilities
    all_nli_scores_raw = score_all_nli(claims_texts, evidence_paragraphs, nli_model)

    n_evidence = len(evidence)
    all_ce_scores = []
    all_nli_scores = []
    
    from .models import NLIScore
    
    for i in range(len(claims_texts)):
        # Expand CE
        full_ce = [0.0] * n_evidence
        claim_ce = all_ce_scores_raw[i] if i < len(all_ce_scores_raw) else []
        for para_i, ev_i in enumerate(para_to_evidence_idx):
            if para_i < len(claim_ce):
                full_ce[ev_i] = claim_ce[para_i]
        all_ce_scores.append(full_ce)
        
        # Expand NLI
        full_nli = [NLIScore(0.0, 0.0, 1.0) for _ in range(n_evidence)]
        claim_nli = all_nli_scores_raw[i] if i < len(all_nli_scores_raw) else []
        for para_i, ev_i in enumerate(para_to_evidence_idx):
            if para_i < len(claim_nli):
                full_nli[ev_i] = claim_nli[para_i]
        all_nli_scores.append(full_nli)

    # ------------------------------------------------------------------ #
    # Step 8: Evidence voting per claim (using NLI)
    # ------------------------------------------------------------------ #
    verifications: list[ClaimVerification] = vote_all_claims(
        claims, evidence, all_ce_scores, all_nli_scores
    )

    # ------------------------------------------------------------------ #
    # Step 9: Composite confidence score + label
    # ------------------------------------------------------------------ #
    confidence_score, label, components = compute_confidence(
        verifications, evidence, authority_scores, entity_alignment_score
    )
    confidence_pct = round(confidence_score * 100)

    # ------------------------------------------------------------------ #
    # Step 10: Deterministic explanation
    # ------------------------------------------------------------------ #
    explanation = generate_explanation(
        label, verifications, evidence, components,
        drift_detected, has_hallucinated_claims
    )

    # ------------------------------------------------------------------ #
    # Aggregate counts for UI (only counting relevant claims)
    # ------------------------------------------------------------------ #
    relevant_verifs = [cv for cv in verifications if getattr(cv.claim, 'is_relevant_to_question', True)]
    
    supported_count    = sum(1 for cv in relevant_verifs if cv.verdict == "supported")
    contradicted_count = sum(1 for cv in relevant_verifs if cv.verdict == "contradicted")
    insufficient_count = sum(1 for cv in relevant_verifs if cv.verdict == "insufficient")

    return VerificationResult(
        label=label,
        confidence_score=confidence_score,
        confidence_pct=confidence_pct,
        explanation=explanation,
        claims=claims,
        claim_verifications=verifications,
        evidence=evidence,
        authority_scores=authority_scores,
        question_entities=q_entities,
        answer_entities=a_entities,
        entity_drift_detected=drift_detected,
        has_hallucinated_claims=has_hallucinated_claims,
        nli_score=components["nli_score"],
        ce_score=components["ce_score"],
        entity_score=components["entity_score"],
        q_relevance_score=components["q_relevance_score"],
        authority_avg=components["authority_avg"],
        support_ratio=components["support_ratio"],
        diversity_score=components["diversity_score"],
        contradiction_penalty=components["contradiction_penalty"],
        supported_count=supported_count,
        contradicted_count=contradicted_count,
        insufficient_count=insufficient_count,
    )


def _empty_result(answer: str, evidence: list[dict]) -> VerificationResult:
    return VerificationResult(
        label="Needs Verification",
        confidence_score=0.0,
        confidence_pct=0,
        explanation="No verifiable factual claims could be extracted.",
        claims=[],
        claim_verifications=[],
        evidence=evidence,
        authority_scores=score_all_sources(evidence),
    )
