"""
verification/claim_filter.py
============================
Filters extracted claims by checking their relevance to the original user question.
Uses the CrossEncoder to score (Question, Claim).
"""

import logging
from .config import CLAIM_RELEVANCE_THRESHOLD
from .models import Claim
from .cross_encoder import load_cross_encoder, score_claim_against_evidence

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
        
    model = load_cross_encoder()
    
    # We treat the question as the "claim" and the actual claims as the "evidence" for the CrossEncoder
    # to see if the claim is relevant to the question.
    claim_texts = [c.text for c in claims]
    relevance_scores = score_claim_against_evidence(question, claim_texts, model)
    
    has_hallucinated = False
    
    for i, claim in enumerate(claims):
        score = relevance_scores[i] if i < len(relevance_scores) else 0.0
        # If the claim is highly irrelevant to the question, we flag it.
        # Note: sometimes a claim answers the question but uses different words. 
        # CrossEncoder is generally good at semantic relevance.
        if score < CLAIM_RELEVANCE_THRESHOLD:
            claim.is_relevant_to_question = False
            has_hallucinated = True
            logger.info(f"Filtered claim as irrelevant to question (score={score:.2f}): {claim.raw_text}")
        else:
            claim.is_relevant_to_question = True
            
    return claims, has_hallucinated
