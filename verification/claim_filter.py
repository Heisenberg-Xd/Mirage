"""
verification/claim_filter.py
============================
Filters extracted claims by checking their relevance to the original user question.
Uses the NLI model to score (Question, Claim).
"""

import logging
from .config import CLAIM_RELEVANCE_THRESHOLD
from .models import Claim
from .nli import load_nli_model, score_all_nli

logger = logging.getLogger(__name__)

def filter_claims_by_question(claims: list[Claim], question: str) -> tuple[list[Claim], bool]:
    """
    Check each claim against the user's question to determine if it's relevant.
    Updates the `is_relevant_to_question` field on the Claim objects.
    
    Returns:
        (claims, has_hallucinated_claims)
        has_hallucinated_claims: True if any claim was marked as irrelevant.
    """
    if not question or not claims:
        return claims, False
        
    model = load_nli_model()
    
    # We use NLI to check if the claim is related to the question.
    claim_texts = [c.text for c in claims]
    
    # score_all_nli expects claims_texts and evidence_paragraphs
    # Here, question is the "claim" and claim_texts are the "evidence"
    # Actually, score_all_nli takes (claims_texts, evidence_paragraphs) and returns [claim][evidence]
    nli_scores = score_all_nli([question], claim_texts, model)
    
    has_hallucinated = False
    
    for i, claim in enumerate(claims):
        nli_res = nli_scores[0][i] if (nli_scores and nli_scores[0] and i < len(nli_scores[0])) else None
        
        # A claim is relevant if the NLI model finds entailment or contradiction with the question.
        # If it's purely neutral, it's not addressing the question.
        # Or, we can just use a generic low threshold on non-neutrality.
        score = (nli_res.entailment + nli_res.contradiction) if nli_res else 0.0
        
        if score < CLAIM_RELEVANCE_THRESHOLD:
            claim.is_relevant_to_question = False
            has_hallucinated = True
            logger.info(f"Filtered claim as irrelevant to question (score={score:.2f}): {claim.raw_text}")
        else:
            claim.is_relevant_to_question = True
            
    return claims, has_hallucinated
