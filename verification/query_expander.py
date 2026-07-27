"""
verification/query_expander.py
==============================
Deterministic query expansion for better evidence retrieval.
"""

import re
from typing import List
from .entity_extractor import extract_entities
from .question_parser import parse_question

def expand_query(question: str) -> List[str]:
    """
    Generate multiple deterministic search queries from the user's question.
    """
    queries = [question] # Always include the raw question
    
    # Clean the question (remove punctuation, lower)
    clean_q = re.sub(r'[^\w\s]', '', question.lower())
    
    entities = extract_entities(question)
    
    # Add a query just focused on entities if there are any
    if entities:
        queries.append(" ".join(entities))
        
    intent_data = parse_question(question)
    intent = intent_data["intent"]
    
    # Deterministic expansions
    if intent == "boolean" and entities:
        # e.g., "Is Kartikesh Prime Minister?" -> "Kartikesh Prime Minister official"
        queries.append(" ".join(entities) + " official news")
    
    if intent == "who" and entities:
        queries.append(" ".join(entities) + " biography profile")
        
    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen and len(q.strip()) > 0:
            seen.add(q)
            unique_queries.append(q)
            
    return unique_queries
