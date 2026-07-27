"""
verification/voting.py
======================
Evidence voting — aggregates per-source NLI scores into a
per-claim verdict (supported / contradicted / insufficient).

Key design decisions:
  1. We vote using DeBERTa v3 zero-shot NLI probabilities.
  2. A claim is 'supported' by a source if NLI entailment > 0.5.
  3. A claim is 'contradicted' by a source if NLI contradiction > 0.5.
  4. Otherwise 'insufficient'.
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
    relevance_scores_for_claim: list[float],
    nli_scores_for_claim: list[NLIScore],
) -> ClaimVerification:
    """
    Aggregate evidence votes for a single claim using NLI probabilities.
    """
    evidence_scores: list[EvidenceScore] = []
    supporting_count = 0
    contradicting_count = 0
    insufficient_count = 0
    
    best_relevance = 0.0
    best_entailment = 0.0
    best_source_idx = 0

    for i, (source, rel_score, nli) in enumerate(zip(evidence_list, relevance_scores_for_claim, nli_scores_for_claim)):
        
        # Determine verdict using NLI probabilities
        if nli.entailment > 0.5:
            verdict = "supported"
            supporting_count += 1
        elif nli.contradiction > 0.5:
            verdict = "contradicted"
            contradicting_count += 1
        else:
            verdict = "insufficient"
            insufficient_count += 1

        if rel_score > best_relevance:
            best_relevance = rel_score
            
        if nli.entailment > best_entailment:
            best_entailment = nli.entailment
            best_source_idx = i

        evidence_scores.append(EvidenceScore(
            source_idx=i,
            raw_relevance_score=rel_score,
            sigmoid_relevance_score=rel_score,
            nli_score=nli,
            verdict=verdict,
        ))

    # Overall claim verdict
    if supporting_count >= 1:
        # If at least one source firmly entails it, we consider the claim supported.
        # (This is more robust than majority voting when evidence might just be unrelated paragraphs)
        overall_verdict = "supported"
    elif contradicting_count > 0:
        overall_verdict = "contradicted"
    else:
        overall_verdict = "insufficient"

    return ClaimVerification(
        claim=claim,
        verdict=overall_verdict,
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        insufficient_count=insufficient_count,
        best_relevance_score=best_relevance,
        best_nli_entailment=best_entailment,
        best_source_idx=best_source_idx,
        evidence_scores=evidence_scores,
    )


def vote_all_claims(
    claims: list[Claim],
    evidence_list: list[dict],
    all_relevance_scores: list[list[float]],
    all_nli_scores: list[list[NLIScore]],
) -> list[ClaimVerification]:
    """
    Run NLI voting for every claim.
    """
    verifications: list[ClaimVerification] = []
    
    for i, claim in enumerate(claims):
        rel_scores = all_relevance_scores[i] if i < len(all_relevance_scores) else [0.0] * len(evidence_list)
        nli_scores = all_nli_scores[i] if i < len(all_nli_scores) else [NLIScore(0.0,0.0,1.0) for _ in evidence_list]
        
        cv = vote_on_claim(claim, evidence_list, rel_scores, nli_scores)
        verifications.append(cv)
        
        logger.debug(
            f"Claim [{i}] '{claim.raw_text[:60]}' → {cv.verdict} "
            f"(support={cv.supporting_count}, best_entailment={cv.best_nli_entailment:.3f})"
        )
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
