"""
verification/search.py
======================
Orchestrates Tavily search across expanded queries and deduplicates results.
"""

import os
import logging
import traceback
from tavily import TavilyClient
from typing import List, Dict, Any
from .config import MAX_EVIDENCE_RESULTS
from .query_expander import expand_query

logger = logging.getLogger(__name__)

def search_evidence_expanded(question: str, max_results: int = MAX_EVIDENCE_RESULTS) -> List[Dict[str, Any]]:
    """
    Run search on multiple expanded queries and deduplicate by URL.
    Returns up to `max_results` top evidence items.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY missing.")
        return []
        
    client = TavilyClient(api_key=tavily_api_key)
    
    queries = expand_query(question)
    logger.info(f"Expanded queries: {queries}")
    
    all_results = []
    seen_urls = set()
    
    # To save API calls and time, we only run the top 2 queries
    # and request max_results for each.
    for q in queries[:2]:
        try:
            response = client.search(query=q, max_results=max_results)
            for r in response.get("results", []):
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "content": r.get("content", ""),
                        "url": url,
                        "title": r.get("title", "")
                    })
        except Exception as e:
            logger.warning(f"Tavily search failed for query '{q}': {type(e).__name__} - {e}")
            traceback.print_exc()
            
    # Return the top N results
    return all_results[:max_results]
