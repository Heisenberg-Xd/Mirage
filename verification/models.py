"""
verification/models.py
======================
Dataclasses that represent every intermediate and final result
produced by the hybrid verification pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Claim:
    """
    A single atomic factual claim extracted from the LLM's answer.
    """
    text: str
    raw_text: str
    is_negated: bool = False
    key_entities: list[str] = field(default_factory=list)
    is_relevant_to_question: bool = True  # Used to flag hallucinated extra info


@dataclass
class NLIScore:
    """
    NLI predictions for a (Claim, Evidence) pair.
    """
    entailment: float
    contradiction: float
    neutral: float


@dataclass
class EvidenceScore:
    """
    The relevance and NLI score of one claim against one evidence source.
    """
    source_idx: int
    nli_score: Optional[NLIScore] = None
    verdict: str = "insufficient"   # "supported" | "insufficient" | "contradicted"


@dataclass
class ClaimVerification:
    """
    The aggregated verification result for a single claim after voting.
    """
    claim: Claim
    verdict: str
    supporting_count: int
    contradicting_count: int
    insufficient_count: int
    best_nli_entailment: float
    best_source_idx: int
    evidence_scores: list[EvidenceScore] = field(default_factory=list)


@dataclass
class VerificationResult:
    """
    The complete hallucination detection output for one question.

    label is one of:
      - "Not Hallucinating"  — all relevant claims supported by evidence
      - "Cannot Verify"      — insufficient evidence to determine hallucination
      - "Hallucinating"      — one or more relevant claims contradicted by evidence

    confidence_score represents confidence IN the hallucination assessment (0–1).
    """
    label: str
    confidence_score: float
    confidence_pct: int
    explanation: str
    
    claims: list[Claim]
    claim_verifications: list[ClaimVerification]
    evidence: list[dict]
    authority_scores: list[float]
    
    # Entity Drift
    question_entities: list[str] = field(default_factory=list)
    answer_entities: list[str] = field(default_factory=list)
    primary_q_entity: str = "None"
    primary_a_entity: str = "None"
    entity_drift_detected: bool = False
    
    # Hallucination / Context Flags
    has_false_hallucination: bool = False
    has_unverified_context: bool = False
    # Component scores (for debugging / UI display)
    logic_trace: list[str] = field(default_factory=list)
    
    # Aggregated counts
    supported_count: int = 0
    contradicted_count: int = 0
    insufficient_count: int = 0
