"""
verification/nli.py
===================
Natural Language Inference (NLI) scoring using cross-encoder/nli-deberta-v3-base.
Predicts Entailment, Contradiction, or Neutral for a (Claim, Evidence) pair.

Production optimizations:
- torch and gc imported at module level (not inside inference functions)
- torch.set_num_threads(2) limits CPU thread contention on Railway
- Model cached via module-level singleton — never loaded twice
- All inference runs under torch.inference_mode() to minimize memory
- gc.collect() called in finally blocks to release tensors promptly
"""

import gc
import logging

import numpy as np

from .config import NLI_MODEL, DISABLE_NLI
from .models import NLIScore

logger = logging.getLogger(__name__)

_NLI_MODEL_INSTANCE = None
_NLI_FAILED = False


def load_nli_model():
    """
    Load the NLI CrossEncoder model lazily on first request.
    Subsequent calls return the cached instance immediately.
    """
    global _NLI_MODEL_INSTANCE, _NLI_FAILED

    if DISABLE_NLI:
        logger.info("NLI Model is DISABLED via environment.")
        return None

    if _NLI_FAILED:
        return None

    if _NLI_MODEL_INSTANCE is not None:
        return _NLI_MODEL_INSTANCE

    try:
        logger.info("Loading NLI model '%s' (CPU only)...", NLI_MODEL)

        import torch
        # Limit CPU threads to prevent Railway container CPU contention.
        # 2 threads is optimal for single-user inference on constrained environments.
        torch.set_num_threads(2)

        from sentence_transformers import CrossEncoder

        # Force CPU to prevent any CUDA memory allocation
        _NLI_MODEL_INSTANCE = CrossEncoder(NLI_MODEL, device="cpu")
        _NLI_MODEL_INSTANCE.model.eval()
        logger.info("NLI model '%s' loaded successfully.", NLI_MODEL)
        return _NLI_MODEL_INSTANCE
    except Exception as e:
        logger.error(
            "Failed to load NLI model '%s' (Memory/Disk issue?). "
            "Disabling permanently. Error: %s",
            NLI_MODEL, e,
        )
        _NLI_FAILED = True
        return None


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def score_nli(claim_text: str, evidence_paragraphs: list[str], model) -> list[NLIScore]:
    """
    Score one claim against multiple evidence paragraphs.

    The cross-encoder/nli-deberta-v3-base model outputs logits:
      Index 0: Contradiction
      Index 1: Entailment
      Index 2: Neutral

    Returns a list of NLIScore objects.
    """
    if not evidence_paragraphs:
        return []

    pairs = [(para, claim_text) for para in evidence_paragraphs]

    try:
        import torch

        model.model.eval()
        with torch.inference_mode():
            raw_scores = model.predict(pairs, show_progress_bar=False)
        probs = softmax(raw_scores)

        nli_scores = []
        for prob in probs:
            # cross-encoder/nli-deberta-v3-base: 0=contradiction, 1=entailment, 2=neutral
            nli_scores.append(NLIScore(
                contradiction=float(prob[0]),
                entailment=float(prob[1]),
                neutral=float(prob[2]),
            ))
        return nli_scores
    except Exception as e:
        logger.warning("NLI inference failed: %s. Returning neutral.", e)
        return [NLIScore(entailment=0.0, contradiction=0.0, neutral=1.0) for _ in evidence_paragraphs]
    finally:
        gc.collect()


def score_all_nli(
    claims_texts: list[str],
    evidence_paragraphs: list[str],
    model,
) -> list[list[NLIScore]]:
    """
    Batch score all claims against all evidence using NLI.
    Runs a single batched forward pass — much faster than one claim at a time.
    """
    _neutral = [NLIScore(entailment=0.0, contradiction=0.0, neutral=1.0) for _ in evidence_paragraphs]

    if not claims_texts or not evidence_paragraphs:
        return [list(_neutral) for _ in claims_texts]

    if model is None:
        # NLI model failed/disabled. Return neutral — voting will treat all as insufficient.
        logger.warning("NLI model unavailable — all claims will score as 'insufficient'.")
        return [list(_neutral) for _ in claims_texts]

    # Build all (evidence, claim) pairs in one flat list for a single batched call
    all_pairs = []
    for ct in claims_texts:
        for para in evidence_paragraphs:
            all_pairs.append((para, ct))

    try:
        import torch

        model.model.eval()
        with torch.inference_mode():
            raw_scores = model.predict(all_pairs, show_progress_bar=False)
        probs = softmax(raw_scores)

        flat_nli_scores = [
            NLIScore(
                contradiction=float(prob[0]),
                entailment=float(prob[1]),
                neutral=float(prob[2]),
            )
            for prob in probs
        ]

        n_ev = len(evidence_paragraphs)
        return [flat_nli_scores[i * n_ev : (i + 1) * n_ev] for i in range(len(claims_texts))]

    except Exception as e:
        logger.warning("Batch NLI inference failed: %s — returning neutral for all pairs.", e)
        n_ev = len(evidence_paragraphs)
        return [[NLIScore(entailment=0.0, contradiction=0.0, neutral=1.0) for _ in range(n_ev)] for _ in claims_texts]
    finally:
        gc.collect()
