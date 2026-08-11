"""
verification/__init__.py
========================
Public API for the hybrid deterministic hallucination detection engine.

Pipeline order:
    1. Question Analysis & Entity Extraction
    2. Answer Entity Extraction & Alignment (Entity Drift → Cannot Verify)
    3. Claim extraction
    4. Claim filtering — keep only claims that directly answer the question
    5. Authority scoring
    6. NLI scoring (entailment/contradiction per claim)
    7. Evidence voting
    8. Hallucination confidence scoring
    9. Explanation template
    → VerificationResult  (label: Not Hallucinating / Cannot Verify / Hallucinating)
"""

import time
import traceback
import logging
from .models import VerificationResult, Claim, ClaimVerification
from .question_parser import parse_question
from .entity_extractor import extract_entities
from .entity_alignment import check_entity_alignment
from .claim_extractor import extract_claims
from .claim_filter import filter_claims_by_question
from .authority import score_all_sources, get_authority_label, get_authority_tier
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
    # Steps 1-2: Question parsing & Entity Alignment
    # ------------------------------------------------------------------ #
    print("      [Verification] Step 1-2: Question Parsing & Entity Extraction...", end=" ", flush=True)
    step_t = time.time()
    try:
        intent_data = parse_question(question)
        q_entities = extract_entities(question)
        a_entities = extract_entities(answer)
        drift_detected, entity_alignment_score, primary_q, primary_a = check_entity_alignment(q_entities, a_entities)
        print(f"SUCCESS ({time.time() - step_t:.2f}s)")
    except Exception as e:
        print(f"FAIL ({time.time() - step_t:.2f}s)")
        print(f"Exception in Step 1-2: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise

    # ------------------------------------------------------------------ #
    # Entity Drift Short-Circuit
    # ------------------------------------------------------------------ #
    if drift_detected:
        logger.warning(f"Entity Drift Detected! Q_entity: '{primary_q}', A_entity: '{primary_a}'")
        explanation = (
            f"The model answered about a different entity ('{primary_a}') instead of the one "
            f"requested by the user ('{primary_q}'). Hallucination status cannot be determined "
            f"because the answer does not address the actual question."
        )
        return VerificationResult(
            label="Cannot Verify",
            confidence_score=0.45,
            confidence_pct=45,
            explanation=explanation,
            claims=[],
            claim_verifications=[],
            evidence=evidence,
            authority_scores=score_all_sources(evidence) if evidence else [],
            question_entities=q_entities,
            answer_entities=a_entities,
            primary_q_entity=primary_q,
            primary_a_entity=primary_a,
            entity_drift_detected=True,
            logic_trace=["Entity drift detected → Pipeline short-circuited → Cannot Verify (45%)"]
        )

    # ------------------------------------------------------------------ #
    # Step 3: Extract atomic claims from the answer
    # ------------------------------------------------------------------ #
    print("      [Verification] Step 3: Claim Extraction...", end=" ", flush=True)
    step_t = time.time()
    try:
        claims: list[Claim] = extract_claims(answer)
        print(f"SUCCESS ({time.time() - step_t:.2f}s)")
        logger.info(f"Extracted {len(claims)} claim(s)")
    except Exception as e:
        print(f"FAIL ({time.time() - step_t:.2f}s)")
        print(f"Exception in Step 3: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise

    if not claims:
        return _empty_result(answer, evidence)

    # ------------------------------------------------------------------ #
    # Step 4: Question Relevance Filter
    # ------------------------------------------------------------------ #
    claims, has_hallucinated_claims = filter_claims_by_question(claims, question)

    # ------------------------------------------------------------------ #
    # Step 5: Score authority of each evidence source
    # ------------------------------------------------------------------ #
    authority_scores: list[float] = score_all_sources(evidence)

    # ------------------------------------------------------------------ #
    # Step 6: NLI Inference
    # ------------------------------------------------------------------ #
    print("      [Verification] Step 6: NLI Scoring...", end=" ", flush=True)
    step_t = time.time()
    try:
        nli_model = load_nli_model()
        
        evidence_paragraphs = [
            src.get("content", "") for src in evidence if src.get("content", "").strip()
        ]

        para_to_evidence_idx = [
            i for i, src in enumerate(evidence) if src.get("content", "").strip()
        ]

        claims_texts = [c.text for c in claims]
        
        # NLI Probabilities
        all_nli_scores_raw = score_all_nli(claims_texts, evidence_paragraphs, nli_model)

        n_evidence = len(evidence)
        all_nli_scores = []
        
        from .models import NLIScore
        
        for i in range(len(claims_texts)):
            # Expand NLI
            full_nli = [NLIScore(0.0, 0.0, 1.0) for _ in range(n_evidence)]
            claim_nli = all_nli_scores_raw[i] if i < len(all_nli_scores_raw) else []
            for para_i, ev_i in enumerate(para_to_evidence_idx):
                if para_i < len(claim_nli):
                    full_nli[ev_i] = claim_nli[para_i]
            all_nli_scores.append(full_nli)
            
        print(f"SUCCESS ({time.time() - step_t:.2f}s)")
    except Exception as e:
        print(f"FAIL ({time.time() - step_t:.2f}s)")
        print(f"Exception in Step 6-7: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise

    # ------------------------------------------------------------------ #
    # Step 7: Evidence voting per claim (using NLI)
    # ------------------------------------------------------------------ #
    verifications: list[ClaimVerification] = vote_all_claims(
        claims, evidence, all_nli_scores, authority_scores
    )

    # ------------------------------------------------------------------ #
    # Step 8: Composite confidence score + label
    # ------------------------------------------------------------------ #
    
    # Calculate extra context flags
    irrelevant_verifs = [cv for cv in verifications if not getattr(cv.claim, 'is_relevant_to_question', True)]
    has_false_hallucination = any(cv.verdict == "contradicted" for cv in irrelevant_verifs)
    has_unverified_context = any(cv.verdict == "insufficient" for cv in irrelevant_verifs)

    confidence_score, label, logic_trace = compute_confidence(
        verifications, evidence, authority_scores, entity_alignment_score, has_false_hallucination
    )
    confidence_pct = round(confidence_score * 100)

    # ------------------------------------------------------------------ #
    # Step 9: Deterministic explanation
    # ------------------------------------------------------------------ #
    explanation = generate_explanation(
        label, verifications, evidence, {},
        drift_detected, has_false_hallucination
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
        primary_q_entity=primary_q,
        primary_a_entity=primary_a,
        entity_drift_detected=drift_detected,
        has_false_hallucination=has_false_hallucination,
        has_unverified_context=has_unverified_context,
        logic_trace=logic_trace,
        supported_count=supported_count,
        contradicted_count=contradicted_count,
        insufficient_count=insufficient_count,
    )


def _empty_result(answer: str, evidence: list[dict]) -> VerificationResult:
    return VerificationResult(
        label="Cannot Verify",
        confidence_score=0.45,
        confidence_pct=45,
        explanation="No verifiable factual claims could be extracted from the answer. Hallucination status cannot be determined.",
        claims=[],
        claim_verifications=[],
        evidence=evidence,
        authority_scores=score_all_sources(evidence) if evidence else [],
        logic_trace=["No claims extracted → Cannot Verify (45%)"]
    )
