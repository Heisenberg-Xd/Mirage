"""
verification/models.py
======================
Dataclasses that represent every intermediate and final result
produced by the hybrid verification pipeline.

Using dataclasses (not dicts) gives us:
  - Type safety with type hints
  - IDE autocompletion
  - Readable repr() for debugging
  - Immutability via frozen=True where appropriate
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Claim:
    """
    A single atomic factual claim extracted from the LLM's answer.

    Attributes:
        text          The normalised text of the claim.
        raw_text      The original, un-normalised sentence from the answer.
        is_negated    True if the claim contains a syntactic negation
                      (e.g. "X is NOT Y", "X never did Z").
        key_entities  Named entities detected inside the claim (persons,
                      places, organisations, dates) — used to focus voting.
    """
    text: str
    raw_text: str
    is_negated: bool = False
    key_entities: list[str] = field(default_factory=list)


@dataclass
class EvidenceScore:
    """
    The raw CrossEncoder relevance score of one claim against one evidence source.

    Attributes:
        source_idx   Index into the evidence list.
        raw_score    Raw CrossEncoder logit (unbounded float).
        sigmoid_score Sigmoid of raw_score, normalised to [0, 1].
        verdict      "supported" | "insufficient" | "contradicted"
    """
    source_idx: int
    raw_score: float
    sigmoid_score: float
    verdict: str   # "supported" | "insufficient" | "contradicted"


@dataclass
class ClaimVerification:
    """
    The aggregated verification result for a single claim after voting
    across all evidence sources.

    Attributes:
        claim             The Claim that was verified.
        verdict           Overall verdict: "supported" | "contradicted" | "insufficient"
        supporting_count  Number of evidence sources that support the claim.
        contradicting_count Number of sources that potentially contradict it.
        insufficient_count  Number of sources with weak/no relevance.
        best_score        Highest sigmoid_score across all sources.
        best_source_idx   Index of the strongest supporting source.
        evidence_scores   Per-source EvidenceScore objects.
    """
    claim: Claim
    verdict: str
    supporting_count: int
    contradicting_count: int
    insufficient_count: int
    best_score: float
    best_source_idx: int
    evidence_scores: list[EvidenceScore] = field(default_factory=list)


@dataclass
class VerificationResult:
    """
    The complete output of the verification engine for one question.
    Consumed by both the UI rendering layer and the template engine.

    Attributes:
        label              Final human-readable label string.
        confidence_score   Final composite confidence value in [0, 1].
        confidence_pct     confidence_score expressed as a 0–100 integer.
        explanation        Template-generated explanation (no LLM involved).
        claims             All atomic claims extracted from the answer.
        claim_verifications Per-claim verification results.
        evidence           Raw evidence list from Tavily (list of dicts).
        authority_scores   Normalised authority score [0,1] per evidence source.
        semantic_score     Component score: weighted mean CrossEncoder scores.
        agreement_score    Component score: supported_claims / total_claims.
        authority_avg      Component score: mean authority of supporting sources.
        support_ratio      Component score: supported / (supported + contradicted).
        diversity_score    Component score: unique supporting domains / total.
        supported_count    Total claims with verdict "supported".
        contradicted_count Total claims with verdict "contradicted".
        insufficient_count Total claims with verdict "insufficient".
    """
    label: str
    confidence_score: float
    confidence_pct: int
    explanation: str
    claims: list[Claim]
    claim_verifications: list[ClaimVerification]
    evidence: list[dict]
    authority_scores: list[float]   # one per evidence source, normalised [0,1]
    # Component scores (for debugging / UI display)
    semantic_score: float = 0.0
    agreement_score: float = 0.0
    authority_avg: float = 0.0
    support_ratio: float = 0.0
    diversity_score: float = 0.0
    # Aggregated counts (convenience fields for the UI)
    supported_count: int = 0
    contradicted_count: int = 0
    insufficient_count: int = 0
