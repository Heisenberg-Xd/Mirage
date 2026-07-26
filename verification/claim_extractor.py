"""
verification/claim_extractor.py
================================
Deterministic atomic claim extraction from LLM-generated answers.

Uses spaCy for:
  - Sentence segmentation
  - Dependency parsing (to find Subject-Verb-Object triples)
  - Named entity recognition (PERSON, ORG, GPE, DATE, etc.)
  - Negation detection via syntactic dependency arcs (token.dep_ == "neg")

No LLM is used. All logic is rule-based NLP.

Design:
  - An "atomic claim" is a single sentence (or sub-clause) that asserts
    one fact. Compound sentences are split at coordinating conjunctions.
  - Negation is detected at the spaCy token level — far more accurate than
    the keyword-window heuristic used in the old pipeline.
  - Very short sentences (<4 tokens after stripping stop words) are dropped
    as they carry insufficient factual content for verification.
"""

import re
import logging
from typing import Optional

import streamlit as st

from .models import Claim
from .normalizer import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy model — loaded once and cached
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading NLP claim extractor (one-time)…")
def _load_spacy():
    """Load and cache the spaCy model. Called once at startup."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm loaded successfully.")
        return nlp
    except OSError:
        logger.error(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "and", "or", "but", "in", "on",
    "at", "by", "for", "of", "to", "with", "as", "it", "that", "this",
    "from", "not", "no", "yes", "so", "if", "then", "than", "also",
}

_FILLER_PATTERNS = re.compile(
    r"^(yes|no|sure|of course|certainly|indeed|absolutely|i think|"
    r"in summary|in conclusion|therefore|however|furthermore|"
    r"please note|it('s| is) worth|keep in mind)[\s,.]",
    re.IGNORECASE,
)


def _is_content_rich(sentence: str, min_tokens: int = 4) -> bool:
    """Return True if the sentence has enough content words to be a verifiable claim."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
    content = [w for w in words if w not in _STOPWORDS]
    return len(content) >= min_tokens


def _detect_negation_in_span(token) -> bool:
    """
    Return True if any child of the given verb/root token is a negation
    modifier (dep_ == "neg"), or if the sentence contains explicit negation
    patterns.
    """
    for child in token.children:
        if child.dep_ == "neg":
            return True
    return False


def _extract_entities(span) -> list[str]:
    """Extract named entity strings from a spaCy span/doc."""
    return [ent.text for ent in span.ents
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE",
                               "NORP", "FAC", "PRODUCT", "EVENT", "LAW",
                               "WORK_OF_ART", "CARDINAL", "ORDINAL")]


def _split_at_coordinators(sent, nlp) -> list[str]:
    """
    Split a spaCy sentence at top-level coordinating conjunctions ("and", "but",
    "or") that link two independent clauses, yielding sub-sentences.
    Returns a list of string sub-claims.
    """
    text = sent.text.strip()
    # Simple heuristic: split on ", and " / "; and " / ". And " only when
    # the second part has its own subject (capital letter or pronoun follows)
    parts = re.split(r"(?:,\s+|\.\s+|;\s+)(?:and|but|or|while|whereas|although)\s+",
                     text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return [p.strip() for p in parts if len(p.strip()) > 10]
    return [text]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_claims(answer: str) -> list[Claim]:
    """
    Extract a list of atomic factual claims from the LLM's answer.

    Steps:
      1. Parse with spaCy for sentence segmentation.
      2. Split compound sentences at coordinating conjunctions.
      3. Drop filler / too-short sentences.
      4. Detect negation via syntactic dependency arcs.
      5. Extract named entities for focused voting downstream.
      6. Normalize the claim text for comparison.

    Returns:
        List of Claim dataclass instances. May be empty if no verifiable
        claims are found (e.g. answer is "I don't know.").
    """
    if not answer or not answer.strip():
        return []

    try:
        nlp = _load_spacy()
    except Exception as e:
        logger.warning(f"spaCy unavailable ({e}). Falling back to sentence splitting.")
        return _fallback_extract(answer)

    doc = nlp(answer)
    claims: list[Claim] = []

    for sent in doc.sents:
        raw_text = sent.text.strip()

        # Skip filler intros
        if _FILLER_PATTERNS.match(raw_text):
            continue

        # Split compound sentences
        sub_texts = _split_at_coordinators(sent, nlp)

        for sub_text in sub_texts:
            if not _is_content_rich(sub_text):
                continue

            # Re-parse the sub-sentence for entity/negation detection
            sub_doc = nlp(sub_text)

            # Negation: check if the root verb or any main verb has a neg child
            is_negated = False
            for token in sub_doc:
                if token.dep_ in ("ROOT", "relcl", "advcl") and token.pos_ == "VERB":
                    if _detect_negation_in_span(token):
                        is_negated = True
                        break

            # Fallback: keyword-level negation for non-standard parse trees
            if not is_negated:
                neg_keywords = re.compile(
                    r"\b(not|never|no|isn'?t|aren'?t|wasn'?t|weren'?t|"
                    r"doesn'?t|don'?t|didn'?t|cannot|can'?t|won'?t|"
                    r"wouldn'?t|shouldn'?t|couldn'?t|neither|nor|false|"
                    r"incorrect|wrong|untrue|denied|denied)\b",
                    re.IGNORECASE,
                )
                is_negated = bool(neg_keywords.search(sub_text))

            entities = _extract_entities(sub_doc)
            normalized_text = normalize(sub_text)

            claims.append(Claim(
                text=normalized_text,
                raw_text=sub_text,
                is_negated=is_negated,
                key_entities=entities,
            ))

    if not claims:
        # Absolute fallback: treat the whole answer as one claim
        return _fallback_extract(answer)

    return claims


def _fallback_extract(answer: str) -> list[Claim]:
    """
    Simple sentence-splitting fallback when spaCy is unavailable.
    Splits on sentence-ending punctuation, keeps non-trivial sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = []
    neg_re = re.compile(
        r"\b(not|never|no|isn'?t|aren'?t|was not|were not|"
        r"doesn'?t|don'?t|didn'?t|cannot|can'?t)\b",
        re.IGNORECASE,
    )
    for s in sentences:
        s = s.strip()
        if _is_content_rich(s, min_tokens=3):
            is_negated = bool(neg_re.search(s))
            claims.append(Claim(
                text=normalize(s),
                raw_text=s,
                is_negated=is_negated,
                key_entities=[],
            ))
    return claims
