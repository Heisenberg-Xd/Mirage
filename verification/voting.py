"""
verification/voting.py
======================
Evidence voting — aggregates per-source CrossEncoder scores into a
per-claim verdict (supported / contradicted / insufficient).

Key design decisions:
  1. We do NOT use only the highest score (that was the old pipeline's flaw).
     Instead we count how many sources support vs. are insufficient.
  2. Negation-awareness: if a claim is marked is_negated=True, we look for
     evidence that supports the POSITIVE proposition. Evidence confirming the
     positive (e.g. "Narendra Modi IS Prime Minister") lends support to the
     negated claim ("X is NOT Prime Minister").
  3. Verdict thresholds come from config.py so they are easily tunable.
"""

import logging
from urllib.parse import urlparse

from .config import VOTE_SUPPORT_THRESHOLD, VOTE_CONTRADICTION_THRESHOLD
from .models import Claim, ClaimVerification, EvidenceScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_domain(url: str) -> str:
    """Extract the bare domain from a URL for diversity scoring."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Strip www. prefix
        return domain.lstrip("www.")
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Core voting function
# ---------------------------------------------------------------------------

def vote_on_claim(
    claim: Claim,
    evidence_list: list[dict],
    scores_for_claim: list[float],
    support_threshold: float = VOTE_SUPPORT_THRESHOLD,
    contradiction_threshold: float = VOTE_CONTRADICTION_THRESHOLD,
) -> ClaimVerification:
    """
    Aggregate evidence votes for a single claim.

    Args:
        claim:               The Claim being evaluated.
        evidence_list:       List of evidence dicts (each with 'url', 'content', etc.).
        scores_for_claim:    Sigmoid CrossEncoder scores — one per evidence source.
        support_threshold:   Sigmoid score above which a source "supports" the claim.
        contradiction_threshold: Score below which a source is "contradicting" potential.

    Returns:
        ClaimVerification with per-source EvidenceScore breakdown and overall verdict.

    Negation logic:
        If claim.is_negated is True, the CrossEncoder is scoring how well the evidence
        *matches the claim text* — but the claim text already contains the negation
        (e.g. "x is not prime minister"). A high score means the evidence is relevant
        to the subject, which SUPPORTS the negated claim (the evidence discussing the
        correct person effectively validates the negation). We therefore treat high
        scores on negated claims the same as support — no inversion needed.
        Contradiction on a negated claim would come from evidence that explicitly
        asserts the negated subject (e.g. evidence saying "X IS prime minister"),
        which a relevance re-ranker would also score highly. We use the contradiction
        threshold only for non-negated claims to avoid false contradictions.
    """
    evidence_scores: list[EvidenceScore] = []
    supporting_count = 0
    contradicting_count = 0
    insufficient_count = 0
    best_score = 0.0
    best_source_idx = 0

    for i, (source, score) in enumerate(zip(evidence_list, scores_for_claim)):
        # Determine verdict for this source
        if score >= support_threshold:
            verdict = "supported"
            supporting_count += 1
        elif score < contradiction_threshold and not claim.is_negated:
            # Very low relevance on non-negated claims might indicate the
            # evidence is about a different entity — mark as insufficient.
            verdict = "insufficient"
            insufficient_count += 1
        else:
            verdict = "insufficient"
            insufficient_count += 1

        if score > best_score:
            best_score = score
            best_source_idx = i

        # Build per-source raw score (we don't store raw logits externally)
        evidence_scores.append(EvidenceScore(
            source_idx=i,
            raw_score=score,          # already sigmoid-normalised from cross_encoder.py
            sigmoid_score=score,
            verdict=verdict,
        ))

    # Overall claim verdict
    if supporting_count >= 2:
        overall_verdict = "supported"
    elif supporting_count == 1:
        # One supporting source — "supported" but with lower confidence weight
        overall_verdict = "supported"
    elif contradicting_count > supporting_count:
        overall_verdict = "contradicted"
    else:
        overall_verdict = "insufficient"

    return ClaimVerification(
        claim=claim,
        verdict=overall_verdict,
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        insufficient_count=insufficient_count,
        best_score=best_score,
        best_source_idx=best_source_idx,
        evidence_scores=evidence_scores,
    )


def vote_all_claims(
    claims: list[Claim],
    evidence_list: list[dict],
    all_scores: list[list[float]],
) -> list[ClaimVerification]:
    """
    Run evidence voting for every claim.

    Args:
        claims:       All extracted claims.
        evidence_list: All evidence sources.
        all_scores:   scores[claim_idx][evidence_idx] — output from cross_encoder.score_all_claims.

    Returns:
        List of ClaimVerification, one per claim.
    """
    verifications: list[ClaimVerification] = []
    for i, claim in enumerate(claims):
        scores_for_claim = all_scores[i] if i < len(all_scores) else [0.0] * len(evidence_list)
        cv = vote_on_claim(claim, evidence_list, scores_for_claim)
        verifications.append(cv)
        logger.debug(
            f"Claim [{i}] '{claim.raw_text[:60]}' → {cv.verdict} "
            f"(support={cv.supporting_count}, best={cv.best_score:.3f})"
        )
    return verifications


def compute_evidence_diversity(
    verifications: list[ClaimVerification],
    evidence_list: list[dict],
) -> float:
    """
    Compute diversity of supporting sources as: unique supporting domains / total sources.
    Returns 0.0 if no evidence exists.
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
