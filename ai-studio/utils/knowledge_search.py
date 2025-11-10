"""
Filesystem-based knowledge search for when vector database is not available.
Uses simple text matching and keyword search.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from utils.logger_util import get_logger
from utils.ocr_util import extract_text_from_pdf, OCR_AVAILABLE

logger = get_logger("utils.knowledge_search")


def search_knowledge_files_filesystem(
    knowledge_dir: Path, 
    query: str, 
    limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Search knowledge files using simple text matching when vector DB is not available.
    Checks for cached .txt files first, then extracts on-the-fly if needed.
    
    Args:
        knowledge_dir: Directory containing knowledge files
        query: Search query text
        limit: Maximum number of results
        
    Returns:
        List of matching files with content snippets
    """
    if not knowledge_dir.exists():
        return []
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Remove very common words to improve matching
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'where', 'when', 'why', 'how'}
    query_words = {w for w in query_words if w not in stop_words and len(w) > 2}
    
    if not query_words:
        # If all words filtered out, use original query
        query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Require at least 2 meaningful words for search (reduces unrelated results)
    # But allow single-word queries if they're substantial (longer than 4 chars)
    if len(query_words) < 2:
        # Check if we have at least one substantial word
        substantial_words = [w for w in query_words if len(w) > 4]
        if not substantial_words:
            logger.debug("Query too short or only stop words, skipping search")
            return []
        # Use the substantial words for search
        query_words = set(substantial_words)
    
    results = []
    
    for file_path in knowledge_dir.iterdir():
        if not file_path.is_file() or file_path.name.endswith('.txt'):
            continue  # Skip .txt cache files themselves
        
        try:
            # Check for cached content first
            cache_path = knowledge_dir / f"{file_path.name}.txt"
            content = ""
            
            if cache_path.exists():
                # Use cached extracted content
                content = cache_path.read_text(encoding='utf-8', errors='ignore')
            elif file_path.suffix.lower() in ['.txt', '.md']:
                # Read text files directly
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            elif file_path.suffix.lower() == '.pdf':
                # Try OCR extraction (may be slow)
                if OCR_AVAILABLE:
                    content = extract_text_from_pdf(file_path)
                    # Cache it for next time
                    if content:
                        try:
                            cache_path.write_text(content, encoding='utf-8')
                        except Exception:
                            pass
                else:
                    logger.debug("OCR not available, skipping PDF: %s", file_path.name)
                    continue
            else:
                continue
            
            if not content:
                continue
            
            content_lower = content.lower()
            
            # Calculate relevance score based on keyword matches
            matches = sum(1 for word in query_words if word in content_lower)
            if matches == 0:
                continue
            
            # Require at least 20% of query words to match (reduced from 30% to catch more relevant docs)
            match_ratio = matches / len(query_words)
            if match_ratio < 0.2:
                continue
            
            # Find a relevant snippet (first occurrence of query terms)
            snippet_start = 0
            best_pos = len(content)
            for word in query_words:
                pos = content_lower.find(word)
                if pos != -1 and pos < best_pos:
                    best_pos = pos
                    snippet_start = max(0, pos - 150)
            
            snippet = content[snippet_start:snippet_start + 1000]
            if len(content) > snippet_start + 1000:
                snippet += "..."
            
            # Calculate score (matches / query_words + length bonus + frequency bonus)
            score = matches / max(len(query_words), 1)
            if len(content) > 1000:  # Prefer longer documents
                score += 0.1
            # Bonus for multiple occurrences
            total_occurrences = sum(content_lower.count(word) for word in query_words)
            score += min(total_occurrences / 10, 0.5)  # Cap bonus
            
            results.append({
                "file_name": file_path.name,
                "file_path": str(file_path),
                "content": snippet,
                "full_content": content,
                "score": score,
                "metadata": {"file_type": file_path.suffix}
            })
            
        except Exception as e:
            logger.warning("Failed to search file %s: %s", file_path.name, e)
            continue
    
    # Sort by score and return top results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

