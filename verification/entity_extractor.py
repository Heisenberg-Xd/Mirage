"""
verification/entity_extractor.py
================================
Wraps spaCy NER to extract entities from questions and answers.
"""

import spacy
import functools

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
    except Exception:
        return []
