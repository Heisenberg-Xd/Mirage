"""
verification/question_parser.py
===============================
Extracts intent and expected answer type from the user's question.
"""

import re
from typing import Dict, Any

def parse_question(question: str) -> Dict[str, Any]:
    """
    Parse the user's question to determine its intent and expected answer type.
    """
    question_lower = question.lower().strip()
    
    intent = "unknown"
    expected_type = "unknown"
    
    if question_lower.startswith(("who", "whose", "whom")):
        intent = "who"
        expected_type = "person_or_org"
    elif question_lower.startswith("where"):
        intent = "where"
        expected_type = "location"
    elif question_lower.startswith("when"):
        intent = "when"
        expected_type = "date_or_time"
    elif question_lower.startswith(("is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would")):
        intent = "boolean"
        expected_type = "boolean"
    elif question_lower.startswith(("how many", "how much")):
        intent = "quantity"
        expected_type = "number"
    elif question_lower.startswith(("how", "why")):
        intent = "explanation"
        expected_type = "text"
    elif question_lower.startswith("what"):
        intent = "what"
        expected_type = "entity_or_concept"
        
    return {
        "intent": intent,
        "expected_type": expected_type
    }
