"""
verification/entity_alignment.py
================================
Detects "Entity Drift" by comparing entities in the generated answer
against the entities in the user's question using RapidFuzz.
"""

from rapidfuzz import fuzz
from .config import ENTITY_MATCH_THRESHOLD

def check_entity_alignment(question_entities: list[str], answer_entities: list[str]) -> tuple[bool, float, str, str]:
    """
    Compare answer entities to question entities.
    
    Returns:
        (drift_detected, alignment_score, primary_q_entity, primary_a_entity)
    """
    if not question_entities:
        # If question has no strict named entities (e.g. "Is it raining?"), we can't detect entity drift easily
        primary_a = answer_entities[0] if answer_entities else "None"
        return False, 1.0, "None", primary_a
        
    primary_q = question_entities[0]
    primary_a = answer_entities[0] if answer_entities else "None"
        
    if not answer_entities:
        return True, 0.0, primary_q, "None"

    # For each question entity, find the best matching answer entity
    match_scores = []
    
    # We will identify the lowest matching question entity as the "drifted" one
    lowest_match_score = 100.0
    drifted_q = primary_q
    drifted_a = primary_a

    for q_ent in question_entities:
        best_match = 0.0
        best_a = "None"
        for a_ent in answer_entities:
            score = fuzz.ratio(q_ent, a_ent)
            token_score = fuzz.token_sort_ratio(q_ent, a_ent)
            current_best = max(score, token_score)
            if current_best > best_match:
                best_match = current_best
                best_a = a_ent
                
        match_scores.append(best_match)
        
        if best_match < lowest_match_score:
            lowest_match_score = best_match
            drifted_q = q_ent
            drifted_a = best_a
            
    # Average alignment of question entities found in the answer
    avg_alignment = sum(match_scores) / len(match_scores)
    alignment_score = avg_alignment / 100.0
    
    drift_detected = False
    
    if alignment_score < (ENTITY_MATCH_THRESHOLD / 100.0):
        drift_detected = True
        
    if drift_detected:
        # If drift is detected, we return the entity that drifted
        return True, alignment_score, drifted_q, drifted_a
        
    return False, alignment_score, primary_q, primary_a
