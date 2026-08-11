"""
verification/nli.py
===================
Natural Language Inference (NLI) scoring using cross-encoder/nli-deberta-v3-base.
Predicts Entailment, Contradiction, or Neutral for a (Claim, Evidence) pair.
"""

import logging
import functools
import numpy as np
from .config import NLI_MODEL
from .models import NLIScore

logger = logging.getLogger(__name__)

_NLI_MODEL_INSTANCE = None

def load_nli_model():
    """
    Load the NLI CrossEncoder model lazily on first request.
    """
    global _NLI_MODEL_INSTANCE
    if _NLI_MODEL_INSTANCE is not None:
        return _NLI_MODEL_INSTANCE

    try:
        logger.info(f"Loading NLI model '{NLI_MODEL}' (CPU only)...")
        from sentence_transformers import CrossEncoder
        # Force CPU to prevent any CUDA memory allocation
        _NLI_MODEL_INSTANCE = CrossEncoder(NLI_MODEL, device="cpu")
        _NLI_MODEL_INSTANCE.model.eval()
        logger.info(f"NLI model '{NLI_MODEL}' loaded.")
        return _NLI_MODEL_INSTANCE
    except Exception as e:
        logger.error(f"Failed to load NLI model: {e}")
        raise

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def score_nli(claim_text: str, evidence_paragraphs: list[str], model) -> list[NLIScore]:
    """
    Score one claim against multiple evidence paragraphs.
    The cross-encoder/nli-deberta-v3-base model outputs logits corresponding to:
    Index 0: Contradiction
    Index 1: Entailment
    Index 2: Neutral
    
    Returns a list of NLIScore objects.
    """
    if not evidence_paragraphs:
        return []
        
    pairs = [(para, claim_text) for para in evidence_paragraphs]
    
    import torch
    import gc

    try:
        model.model.eval()
        with torch.inference_mode():
            raw_scores = model.predict(pairs, show_progress_bar=False)
        probs = softmax(raw_scores)
        
        nli_scores = []
        for prob in probs:
            # According to the model card for cross-encoder/nli-deberta-v3-base:
            # 0: contradiction, 1: entailment, 2: neutral
            nli_scores.append(NLIScore(
                contradiction=float(prob[0]),
                entailment=float(prob[1]),
                neutral=float(prob[2])
            ))
        return nli_scores
    except Exception as e:
        logger.warning(f"NLI inference failed: {e}. Returning neutral.")
        return [NLIScore(0.0, 0.0, 1.0) for _ in evidence_paragraphs]
    finally:
        gc.collect()

def score_all_nli(claims_texts: list[str], evidence_paragraphs: list[str], model) -> list[list[NLIScore]]:
    """
    Batch score all claims against all evidence using NLI.
    """
    if not claims_texts or not evidence_paragraphs:
        return [[NLIScore(0.0, 0.0, 1.0) for _ in evidence_paragraphs] for _ in claims_texts]
        
    all_pairs = []
    for ct in claims_texts:
        for para in evidence_paragraphs:
            all_pairs.append((para, ct))
            
    import torch
    import gc

    try:
        model.model.eval()
        with torch.inference_mode():
            raw_scores = model.predict(all_pairs, show_progress_bar=False)
        probs = softmax(raw_scores)
        
        flat_nli_scores = []
        for prob in probs:
            flat_nli_scores.append(NLIScore(
                contradiction=float(prob[0]),
                entailment=float(prob[1]),
                neutral=float(prob[2])
            ))
            
        n_ev = len(evidence_paragraphs)
        result = []
        for i in range(len(claims_texts)):
            result.append(flat_nli_scores[i * n_ev : (i + 1) * n_ev])
        return result
        
    except Exception as e:
        logger.warning(f"Batch NLI inference failed: {e}")
        n_ev = len(evidence_paragraphs)
        return [[NLIScore(0.0, 0.0, 1.0) for _ in range(n_ev)] for _ in claims_texts]
    finally:
        gc.collect()
