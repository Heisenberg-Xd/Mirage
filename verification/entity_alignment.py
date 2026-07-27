"""
verification/entity_alignment.py
================================
Detects "Entity Drift" by comparing entities in the generated answer
against the entities in the user's question using RapidFuzz.
"""

from rapidfuzz import fuzz
from .config import ENTITY_MATCH_THRESHOLD

def check_entity_alignment(question_entities: list[str], answer_entities: list[str]) -> tuple[bool, float]:
    """
    Compare answer entities to question entities.
    If the question has entities, we expect the answer to mention them or very similar ones.
    
    Returns:
        (drift_detected, alignment_score)
        drift_detected: True if the answer introduces major entities not in the question,
                        or completely misses the question entities.
        alignment_score: 0.0 to 1.0 (1.0 = perfect alignment)
    """
    if not question_entities:
        # If question has no strict named entities (e.g. "Is it raining?"), we can't detect entity drift easily
        return False, 1.0
        
    if not answer_entities:
        # Answer has no entities but question did -> potential issue, but usually handled by relevance
        return False, 0.5

    # For each question entity, find the best matching answer entity
    match_scores = []
    for q_ent in question_entities:
        best_match = 0.0
        for a_ent in answer_entities:
            score = fuzz.ratio(q_ent, a_ent)
            # Also check substring match (e.g. "USA" in "United States of America" -> might fail ratio, 
            # but we assume normalization handles major synonyms before this step if possible).
            # We use token_sort_ratio for robust matching
            token_score = fuzz.token_sort_ratio(q_ent, a_ent)
            best_match = max(best_match, score, token_score)
            
        match_scores.append(best_match)
        
    # Average alignment of question entities found in the answer
    avg_alignment = sum(match_scores) / len(match_scores)
    
    # Are there entirely new entities in the answer that don't match the question?
    # (Hallucination of new entities)
    hallucinated_entities = 0
    for a_ent in answer_entities:
        best_match = max([fuzz.ratio(a_ent, q_ent) for q_ent in question_entities] + [0])
        if best_match < ENTITY_MATCH_THRESHOLD:
            hallucinated_entities += 1
            
    # If average alignment is low, we have drift.
    alignment_score = avg_alignment / 100.0
    
    # Drift is detected if alignment is poor OR too many unprompted entities are introduced
    # (In our target case, "Kartikesh" vs "Kartikeya" will have ratio ~ 88, which is high!
    # Wait, Kartikesh vs Kartikeya: fuzz.ratio("kartikesh", "kartikeya") = 77.7 
    # So if threshold is 80, it will be flagged as drift.)
    
    drift_detected = False
    if alignment_score < (ENTITY_MATCH_THRESHOLD / 100.0):
        drift_detected = True
        
    # If the answer introduces entirely new entities, it might be an expansion hallucination.
    # We will flag it if the ratio of hallucinated entities is very high.
    if hallucinated_entities > len(question_entities) * 2:
        drift_detected = True
        
    return drift_detected, alignment_score
