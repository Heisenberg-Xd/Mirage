"""
verification/config.py
======================
Central configuration for the hybrid verification engine.
All weights, thresholds, and model names are defined here so they
can be tuned in one place without touching pipeline logic.
Includes support for evidence authority weighting and confidence margins.
"""

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

# CrossEncoder model for claim-vs-evidence relevance scoring.
# ms-marco-MiniLM-L-6-v2: fast (~22MB), optimised for retrieval re-ranking.
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# NLI model for Entailment / Contradiction / Neutral prediction.
# DeBERTa v3 zero-shot is a robust NLI classifier.
# Default voting thresholds: relevance >= 0.60, NLI confidence >= 0.70, margin = 0.15
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"

# spaCy model for dependency parsing and NER.
SPACY_MODEL: str = "en_core_web_sm"

# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------

# Number of Tavily evidence results to retrieve per query string.
# (We might expand to multiple query strings and deduplicate down to MAX_EVIDENCE_RESULTS)
MAX_EVIDENCE_RESULTS: int = 5

# ---------------------------------------------------------------------------
# Entity Alignment & Relevance parameters
# ---------------------------------------------------------------------------

# RapidFuzz similarity ratio [0, 100] above which two entities are considered matching.
ENTITY_MATCH_THRESHOLD: float = 80.0

# CrossEncoder relevance threshold above which a claim is considered "relevant"
# to the user's question (to filter out hallucinated extra information).
CLAIM_RELEVANCE_THRESHOLD: float = 0.50

# ---------------------------------------------------------------------------
# Composite confidence score weights
# Must sum to 1.0 (100%)
# ---------------------------------------------------------------------------
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "nli_entailment":  0.30,   # Strongest signal: NLI says it's true
    "ce_relevance":    0.20,   # CrossEncoder relevance of evidence to claim
    "entity_align":    0.15,   # Do answer entities match question entities?
    "q_relevance":     0.10,   # Are the claims actually answering the question?
    "authority":       0.10,   # Domain authority of supporting sources
    "diversity":       0.05,   # Number of unique supporting domains
    "support_ratio":   0.05,   # Ratio of supported vs contradicted claims
    "contradiction":   0.05,   # Penalty weight (subtracted or used as inverse signal)
}

# ---------------------------------------------------------------------------
# Hallucination Detection Label thresholds
# ---------------------------------------------------------------------------
# Confidence score represents confidence IN the hallucination assessment.
# Not Hallucinating → all relevant claims supported  → score 95–100%
# Cannot Verify     → insufficient evidence          → score 40–60%
# Hallucinating     → any relevant claim contradicted→ score 10–30%
LABEL_THRESHOLDS: dict[str, float] = {
    "not_hallucinating": 0.95,   # all relevant claims supported
    "cannot_verify":     0.40,   # insufficient evidence to confirm or deny
    "hallucinating":     0.20,   # relevant claim(s) contradicted by evidence
}

# Human-readable label strings
LABEL_STRINGS: dict[str, str] = {
    "not_hallucinating": "Not Hallucinating",
    "hallucinating":     "Hallucinating",
    "cannot_verify":     "Cannot Verify",
}
