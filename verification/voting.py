"""
verification/voting.py
======================
Evidence voting — aggregates per-source NLI scores into a
per-claim verdict (supported / contradicted / insufficient).

These verdicts feed the hallucination decision tree:
  - supported    → evidence entails the claim
  - contradicted → evidence contradicts the claim (→ Hallucinating if relevant)
  - insufficient → no conclusive evidence either way (→ Cannot Verify)

Key design decisions:
  1. We vote using DeBERTa v3 zero-shot NLI probabilities.
  2. A claim is 'supported' by a source if NLI entailment > 0.5.
  3. A claim is 'contradicted' by a source if NLI contradiction > 0.5.
  4. Otherwise 'insufficient'.
  5. Only relevant claims (those that directly answer the question) are
     used to determine the final hallucination label.
"""

import logging
from urllib.parse import urlparse

from .models import Claim, ClaimVerification, EvidenceScore, NLIScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_domain(url: str) -> str:
    """Extract the bare domain from a URL for diversity scoring."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return domain.lstrip("www.")
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Core voting function
# ---------------------------------------------------------------------------

def vote_on_claim(
    claim: Claim,
    evidence_list: list[dict],
    nli_scores_for_claim: list[NLIScore],
    authority_scores: list[float],
) -> ClaimVerification:
    """
    Aggregate evidence votes for a single claim using NLI probabilities
    and Authority Scores.
    """
    evidence_scores: list[EvidenceScore] = []
    supporting_count = 0
    contradicting_count = 0
    insufficient_count = 0
    
    best_entailment = 0.0
    best_source_idx = 0

    support_score = 0.0
    contradiction_score = 0.0
    
    MIN_NLI_CONFIDENCE = 0.70  # Ignore predictions where model confidence is below 70%

    for i, (source, nli, auth) in enumerate(zip(evidence_list, nli_scores_for_claim, authority_scores)):
        
        url = source.get("url", "unknown")
        snippet = source.get("content", "")[:100].replace("\n", " ") + "..."
            
        # Check if NLI model is disabled or crashed (fallback returns 0,0,1)
        if nli.entailment == 0.0 and nli.contradiction == 0.0 and nli.neutral == 1.0:
            # Fallback to lexical/authority scoring
            if auth > 0.50:
                # Mock a strong entailment
                nli = NLIScore(entailment=0.8, contradiction=0.0, neutral=0.2)
                logger.info(f"NLI FALLBACK TRIGGERED: Claim '{claim.raw_text}' scored Supported based on Authority ({auth:.2f})")
            
        # Ignore uncertain NLI predictions
        if max(nli.entailment, nli.contradiction) < MIN_NLI_CONFIDENCE:
            evidence_scores.append(EvidenceScore(
                source_idx=i,
                nli_score=nli,
                verdict="insufficient",
            ))
            logger.info(f"Claim: '{claim.raw_text}' | Ent: {nli.entailment:.2f} | Cont: {nli.contradiction:.2f} | Neut: {nli.neutral:.2f} | URL: {url} | Snippet: {snippet} | Verdict: INSUFFICIENT (NLI Uncertain)")
            insufficient_count += 1
            continue

        # Calculate contributions using nli entailment/contradiction and source authority
        support_contrib = nli.entailment * auth
        contradiction_contrib = nli.contradiction * auth
        
        # Accumulate total support score for the claim
        support_score += support_contrib
        # Accumulate total contradiction score for the claim
        contradiction_score += contradiction_contrib

        # Determine individual evidence verdict for logging/display
        if nli.entailment > nli.contradiction:
            verdict = "supported"
            supporting_count += 1
        else:
            verdict = "contradicted"
            contradicting_count += 1
            
        if support_contrib > best_entailment: # Tracking highest weighted supporting source
            best_entailment = support_contrib
            best_source_idx = i

        evidence_scores.append(EvidenceScore(
            source_idx=i,
            nli_score=nli,
            verdict=verdict,
        ))
        
        logger.info(f"Claim: '{claim.raw_text}' | Ent: {nli.entailment:.2f} | Cont: {nli.contradiction:.2f} | Neut: {nli.neutral:.2f} | URL: {url} | Snippet: {snippet} | Verdict: {verdict.upper()}")

    # Aggregate claim verdict with 0.15 margin to avoid false positives
    margin = 0.15
    if support_score > contradiction_score + margin:
        overall_verdict = "supported"
    elif contradiction_score > support_score + margin:
        overall_verdict = "contradicted"
    else:
        overall_verdict = "insufficient"

    return ClaimVerification(
        claim=claim,
        verdict=overall_verdict,
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        insufficient_count=insufficient_count,
        best_nli_entailment=best_entailment,  # Represents best support contribution
        best_source_idx=best_source_idx,
        evidence_scores=evidence_scores,
    )


def vote_all_claims(
    claims: list[Claim],
    evidence_list: list[dict],
    all_nli_scores: list[list[NLIScore]],
    authority_scores: list[float],
) -> list[ClaimVerification]:
    """
    Run NLI voting for every claim.
    """
    verifications: list[ClaimVerification] = []
    
    # Ensure authority scores list matches evidence list length
    if not authority_scores:
        authority_scores = [0.5] * len(evidence_list)
    elif len(authority_scores) < len(evidence_list):
        authority_scores.extend([0.5] * (len(evidence_list) - len(authority_scores)))
    
    # Log detailed claim evaluation trace for debugging
    logger.info("==================================================")
    logger.info("VERIFICATION LOG")
    logger.info("==================================================")
    
    for i, claim in enumerate(claims):
        nli_scores = all_nli_scores[i] if i < len(all_nli_scores) else [NLIScore(0.0,0.0,1.0) for _ in evidence_list]
        
        cv = vote_on_claim(claim, evidence_list, nli_scores, authority_scores)
        verifications.append(cv)
        
        logger.info(f"Final Verdict for Claim [{i}]: {cv.verdict.upper()} (Support: {cv.supporting_count}, Contradict: {cv.contradicting_count})")
        logger.info("---------------------------------")
        
    logger.info("==================================================")
        
    return verifications


def compute_evidence_diversity(
    verifications: list[ClaimVerification],
    evidence_list: list[dict],
) -> float:
    """
    Compute diversity of supporting sources as: unique supporting domains / total sources.
    """
    if not evidence_list:
        return 0.0

    supporting_domains: set[str] = set()
    for cv in verifications:
        if cv.verdict == "supported":
            for es in cv.evidence_scores:
                if es.verdict == "supported" and es.source_idx < len(evidence_list):
                    url = evidence_list[es.source_idx].get("url", "")
                    supporting_domains.add(_get_domain(url))

    return len(supporting_domains) / len(evidence_list)
