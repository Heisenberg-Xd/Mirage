"""
verification/config.py
======================
Central configuration for the hybrid verification engine.
All weights, thresholds, and model names are defined here so they
can be tuned in one place without touching pipeline logic.
Includes support for evidence authority weighting and confidence margins.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Disable Flags for Low-Memory Environments
# ---------------------------------------------------------------------------
DISABLE_NLI           = os.getenv("DISABLE_NLI", "false").lower() == "true"
DISABLE_SPACY         = os.getenv("DISABLE_SPACY", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

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

# NLI combined entailment+contradiction threshold to consider a claim related to the question.
CLAIM_RELEVANCE_THRESHOLD: float = 0.15

# ---------------------------------------------------------------------------
# Composite confidence score weights
# Must sum to 1.0 (100%)
# ---------------------------------------------------------------------------
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "nli_entailment":  0.40,   # Strongest signal: NLI says it's true
    "entity_align":    0.15,   # Do answer entities match question entities?
    "q_relevance":     0.15,   # Are the claims actually answering the question?
    "authority":       0.15,   # Domain authority of supporting sources
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
