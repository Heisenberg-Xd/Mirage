"""
verification/entity_extractor.py
================================
Wraps spaCy NER to extract entities from questions and answers.
"""

import logging
import traceback
import spacy
import functools

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=None)
def get_spacy_model():
    return spacy.load("en_core_web_sm")

def extract_entities(text: str) -> list[str]:
    """
    Extract Named Entities from text.
    Returns a list of lowercase entity strings.
    """
    if not text:
        return []
        
    try:
        nlp = get_spacy_model()
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

