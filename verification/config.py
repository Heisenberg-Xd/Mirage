"""
verification/config.py
======================
Central configuration for the hybrid verification engine.
All weights, thresholds, and model names are defined here so they
can be tuned in one place without touching pipeline logic.
"""

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

# CrossEncoder model for claim-vs-evidence relevance scoring.
# ms-marco-MiniLM-L-6-v2: fast (~22MB), optimised for retrieval re-ranking.
# Swap for "cross-encoder/nli-deberta-v3-base" for full NLI-style scoring
# (3-way entail/neutral/contradict) at the cost of ~180MB and slower inference.
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# spaCy model for dependency parsing and NER during claim extraction.
SPACY_MODEL: str = "en_core_web_sm"

# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------

# Number of Tavily evidence results to retrieve per query.
MAX_EVIDENCE_RESULTS: int = 5

# ---------------------------------------------------------------------------
# Evidence voting thresholds
# ---------------------------------------------------------------------------

# CrossEncoder raw score (logit) above which an evidence item counts as
# "supporting" a claim.  ms-marco logits are unbounded; empirically
# scores > 5.0 indicate strong relevance.  We use a normalised [0,1]
# sigmoid form internally, so these are the sigmoid-space thresholds.
VOTE_SUPPORT_THRESHOLD: float = 0.65   # sigmoid(score) ≥ this → supported
VOTE_CONTRADICTION_THRESHOLD: float = 0.30  # sigmoid(score) < this → potential contradiction

# ---------------------------------------------------------------------------
# Composite confidence score weights
# Must sum to 1.0
# ---------------------------------------------------------------------------
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "semantic":      0.38,   # weighted-average best CrossEncoder score per claim
    "agreement":     0.22,   # fraction of claims that are supported
    "authority":     0.15,   # mean authority score of supporting sources
    "support_ratio": 0.15,   # supported / (supported + contradicted)
    "diversity":     0.10,   # unique supporting domains / total sources
}

# ---------------------------------------------------------------------------
# Label thresholds (applied to final 0–1 composite score)
# ---------------------------------------------------------------------------
LABEL_THRESHOLDS: dict[str, float] = {
    "certain":        0.88,
    "likely_certain": 0.68,
    "uncertain":      0.46,
    # below uncertain → "Needs Verification"
}

# Human-readable label strings
LABEL_STRINGS: dict[str, str] = {
    "certain":        "Certain",
    "likely_certain": "Likely Certain",
    "uncertain":      "Uncertain",
    "needs_verification": "Needs Verification",
}
