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
import functools
from typing import Optional

import numpy as np

from .config import CROSS_ENCODER_MODEL, DISABLE_CROSS_ENCODER

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
# Model loading — cached for the entire process lifetime
# ---------------------------------------------------------------------------

_MODEL = None
_CROSS_ENCODER_FAILED = False

def load_cross_encoder():
    """
    Load the CrossEncoder model lazily on first request.
    This prevents memory spikes during server startup.
    """
    global _MODEL, _CROSS_ENCODER_FAILED
    
    if DISABLE_CROSS_ENCODER:
        logger.info("CrossEncoder is DISABLED via environment.")
        return None
        
    if _CROSS_ENCODER_FAILED:
        return None
        
    if _MODEL is not None:
        return _MODEL

    try:
        logger.info(f"Loading CrossEncoder '{CROSS_ENCODER_MODEL}' (CPU only)...")
        from sentence_transformers import CrossEncoder
        # Force CPU to prevent any CUDA memory allocation
        _MODEL = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
        _MODEL.model.eval()
        logger.info("CrossEncoder loaded successfully.")
        return _MODEL
    except Exception as e:
        logger.error(f"Failed to load CrossEncoder (Memory/Disk issue?). Disabling permanently. Error: {e}")
        _CROSS_ENCODER_FAILED = True
        return None


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

    import torch
    import gc

    try:
        model.model.eval()
        with torch.inference_mode():
            raw_scores: np.ndarray = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"CrossEncoder inference failed: {e}. Returning zeros.")
        return [0.0] * len(evidence_paragraphs)
    finally:
        gc.collect()

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
    n_claims = len(claims_texts)
    n_ev = len(evidence_paragraphs)
    
    if not claims_texts or not evidence_paragraphs:
        return [[0.0] * n_ev] * n_claims
        
    if model is None:
        # Fallback if CrossEncoder failed to load (e.g., OOM)
        # Return neutral 0.60 relevance so voting can proceed via lexical/authority
        return [[0.60] * n_ev for _ in range(n_claims)]

    # Flatten all (claim, evidence) pairs
    all_pairs: list[tuple[str, str]] = []
    for ct in claims_texts:
        for para in evidence_paragraphs:
            all_pairs.append((ct, para))

    import torch
    import gc

    try:
        model.model.eval()
        with torch.inference_mode():
            raw_scores: np.ndarray = model.predict(all_pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"Batch CrossEncoder inference failed: {e}. Returning zeros.")
        n_claims = len(claims_texts)
        n_ev = len(evidence_paragraphs)
        return [[0.0] * n_ev for _ in range(n_claims)]
    finally:
        gc.collect()

    sigmoid_scores = [_sigmoid(float(s)) for s in raw_scores]

    # Reshape flat list back to [n_claims][n_evidence]
    n_ev = len(evidence_paragraphs)
    result = []
    for i in range(len(claims_texts)):
        result.append(sigmoid_scores[i * n_ev: (i + 1) * n_ev])
    return result
