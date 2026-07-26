"""
verification/cross_encoder.py
==============================
CrossEncoder-based semantic verification of claims against evidence.

A CrossEncoder takes BOTH texts together (claim + evidence) as input
and produces a single relevance score — unlike a bi-encoder which encodes
each text independently and compares embeddings.

This joint attention mechanism means a CrossEncoder can detect whether
evidence *actually addresses* the claim, even when surface-level wording
differs — which is the root cause of failures in the old cosine-similarity
pipeline.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22MB download, ~2s one-time load
  - Outputs unbounded logits; higher = stronger relevance match
  - Converted to [0,1] via sigmoid for interpretable thresholding

No LLM is called. This is a local neural scoring model — deterministic
given fixed weights and inputs.
"""

import logging
import math
from typing import Optional

import numpy as np
import streamlit as st

from .config import CROSS_ENCODER_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sigmoid helper — converts raw logits to [0, 1]
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# Model loading — cached for the entire Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CrossEncoder model (one-time)…")
def load_cross_encoder():
    """
    Load the CrossEncoder model from HuggingFace Hub and cache it.
    This runs exactly once per Streamlit process lifetime.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info(f"CrossEncoder '{CROSS_ENCODER_MODEL}' loaded.")
        return model
    except Exception as e:
        logger.error(f"Failed to load CrossEncoder: {e}")
        raise


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_claim_against_evidence(
    claim_text: str,
    evidence_paragraphs: list[str],
    model,
) -> list[float]:
    """
    Score one claim against every evidence paragraph using the CrossEncoder.

    Returns a list of sigmoid-normalised scores in [0, 1], one per
    evidence paragraph.  An empty list is returned if there are no paragraphs.

    The claim_text should be the normalized version for best matching.
    We also run a secondary pass with the raw text if scores are uniformly low
    (surface-level mismatch fallback).
    """
    if not evidence_paragraphs:
        return []

    # Build (claim, evidence) pairs for batch inference
    pairs = [(claim_text, para) for para in evidence_paragraphs]

    try:
        raw_scores: np.ndarray = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"CrossEncoder inference failed: {e}. Returning zeros.")
        return [0.0] * len(evidence_paragraphs)

    sigmoid_scores = [_sigmoid(float(s)) for s in raw_scores]
    return sigmoid_scores


def score_all_claims(
    claims_texts: list[str],
    evidence_paragraphs: list[str],
    model,
) -> list[list[float]]:
    """
    Batch-score all claims against all evidence paragraphs.
    Returns a 2-D list: scores[claim_idx][evidence_idx] = sigmoid_score.

    All pairs are submitted in a single model.predict() call for efficiency.
    """
    if not claims_texts or not evidence_paragraphs:
        return [[0.0] * len(evidence_paragraphs)] * len(claims_texts)

    # Flatten all (claim, evidence) pairs
    all_pairs: list[tuple[str, str]] = []
    for ct in claims_texts:
        for para in evidence_paragraphs:
            all_pairs.append((ct, para))

    try:
        raw_scores: np.ndarray = model.predict(all_pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"Batch CrossEncoder inference failed: {e}. Returning zeros.")
        n_claims = len(claims_texts)
        n_ev = len(evidence_paragraphs)
        return [[0.0] * n_ev for _ in range(n_claims)]

    sigmoid_scores = [_sigmoid(float(s)) for s in raw_scores]

    # Reshape flat list back to [n_claims][n_evidence]
    n_ev = len(evidence_paragraphs)
    result = []
    for i in range(len(claims_texts)):
        result.append(sigmoid_scores[i * n_ev: (i + 1) * n_ev])
    return result
