from __future__ import annotations

from typing import List, Dict


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Simple web search using DuckDuckGo.
    Falls back to empty results if search fails.
    
    Note: DuckDuckGo works better with shorter, keyword-focused queries.
    We automatically simplify very long queries by removing question words.
    """
    try:
        from ddgs import DDGS
        
        # Simplify query if it's too long or question-like
        simplified_query = query
        if len(query) > 50 or any(query.lower().startswith(w) for w in ['what ', 'how ', 'why ', 'when ', 'where ', 'who ', 'which ']):
            # Remove common question words and filler
            stop_words = {'what', 'are', 'is', 'was', 'were', 'the', 'a', 'an', 'of', 'for', 'to', 'in', 'on', 'at', 'by', 'with', 'from'}
            # Remove words like "latest", "recent" and keep core technical terms
            temporal_words = {'latest', 'recent', 'new', 'current', 'findings', 'results', 'updates'}
            
            words = query.replace('?', '').split()
            keywords = [w for w in words if w.lower() not in stop_words and w.lower() not in temporal_words]
            
            simplified_query = ' '.join(keywords[:5])  # Keep top 5 core keywords
            print(f"[WebSearch] Simplified: '{query[:60]}...' -> '{simplified_query}'")
        
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(simplified_query, max_results=max_results)
            # Convert generator to list and iterate
            try:
                for r in search_results:
                    if r and isinstance(r, dict):
                        results.append({
                            'title': r.get('title', 'No title'),
                            'href': r.get('href', ''),
                            'body': r.get('body', 'No description available')
                        })
                    if len(results) >= max_results:
                        break
            except StopIteration:
                pass
        
        print(f"[WebSearch] Found {len(results)} results")
        return results
        
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return []
