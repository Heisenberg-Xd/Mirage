"""
verification/spacy_loader.py
============================
Single source of truth for the spaCy model instance.

Both claim_extractor.py and entity_extractor.py import from here,
ensuring the model is only loaded ONCE and never duplicated in RAM.

Usage:
    from verification.spacy_loader import get_spacy_model
    nlp = get_spacy_model()   # returns cached instance
"""

import functools
import logging

from .config import DISABLE_SPACY, SPACY_MODEL

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_spacy_model():
    """
    Load and cache the spaCy model exactly once.

    Raises:
        ImportError  — if spaCy is not installed.
        RuntimeError — if DISABLE_SPACY is set.
        OSError      — if the model package is not downloaded.
    """
    if DISABLE_SPACY:
        raise RuntimeError(
            "spaCy is DISABLED via DISABLE_SPACY environment variable. "
            "Set DISABLE_SPACY=false to enable."
        )

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is not installed. Add 'spacy' to requirements.txt "
            "and run: python -m spacy download en_core_web_sm"
        ) from exc

    try:
        nlp = spacy.load(SPACY_MODEL)
        logger.info("spaCy model '%s' loaded successfully.", SPACY_MODEL)
        return nlp
    except OSError as exc:
        raise OSError(
            f"spaCy model '{SPACY_MODEL}' not found. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from exc
