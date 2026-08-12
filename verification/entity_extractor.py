"""
verification/entity_extractor.py
================================
Wraps spaCy NER to extract entities from questions and answers.

Uses the shared spacy_loader module — the model is never loaded twice.
"""

import logging
import traceback

logger = logging.getLogger(__name__)


def extract_entities(text: str) -> list[str]:
    """
    Extract Named Entities from text using spaCy.
    Returns a deduplicated list of lowercase entity strings.

    Falls back to an empty list if spaCy is unavailable or DISABLED.
    """
    if not text:
        return []

    try:
        from .spacy_loader import get_spacy_model
        nlp = get_spacy_model()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("[entity_extractor] spaCy unavailable: %s — returning no entities.", e)
        return []

    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"):
                entities.append(ent.text.lower().strip())
        return list(set(entities))
    except Exception as exc:
        logger.error("[entity_extractor] extract_entities FAILED: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        return []
