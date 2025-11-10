"""
Script to reindex all existing knowledge files into the vector database.
This ensures all files are searchable even if they were uploaded before vector DB was set up.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ocr_util import extract_text_from_pdf
from utils.logger_util import get_logger

logger = get_logger("scripts.reindex_knowledge")

# Import vector DB and embedder
try:
    from core.vector_db import VectorDB
    from core.embedder import Embedder
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logger.error("Vector DB dependencies not available")


def reindex_knowledge_files(knowledge_dir: Path) -> None:
    """
    Reindex all knowledge files into the vector database.
    
    Args:
        knowledge_dir: Directory containing knowledge files
    """
    if not VECTOR_DB_AVAILABLE:
        logger.error("Vector DB not available. Cannot reindex files.")
        return
    
    if not knowledge_dir.exists():
        logger.error("Knowledge directory does not exist: %s", knowledge_dir)
        return
    
    try:
        vector_db = VectorDB()
        vector_db.connect()
        embedder = Embedder()
        embedder.load()
        logger.info("Vector DB and embedder initialized")
    except Exception as e:
        logger.error("Failed to initialize vector DB: %s", e)
        return
    
    processed = 0
    skipped = 0
    errors = 0
    
    for file_path in knowledge_dir.iterdir():
        if not file_path.is_file():
            continue
        
        # Skip cache files
        if file_path.name.endswith('.txt') and file_path.suffix == '.txt':
            # Check if this is a cache file (has corresponding PDF)
            pdf_path = knowledge_dir / file_path.name.replace('.txt', '')
            if pdf_path.exists() and pdf_path.suffix == '.pdf':
                continue  # This is a cache file, skip it
        
        try:
            # Check for cached content first
            cache_path = knowledge_dir / f"{file_path.name}.txt"
            content = ""
            
            if cache_path.exists():
                # Use cached extracted content
                content = cache_path.read_text(encoding='utf-8', errors='ignore')
                logger.debug("Using cached content for: %s", file_path.name)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            elif file_path.suffix.lower() == '.pdf':
                logger.info("Extracting text from PDF: %s", file_path.name)
                content = extract_text_from_pdf(file_path)
                if content:
                    # Cache it
                    try:
                        cache_path.write_text(content, encoding='utf-8')
                    except Exception:
                        pass
            else:
                logger.debug("Skipping unsupported file type: %s", file_path.name)
                skipped += 1
                continue
            
            if not content:
                logger.warning("No content extracted from: %s", file_path.name)
                skipped += 1
                continue
            
            # Generate embedding and store in vector DB
            try:
                embedding = embedder.embed(content)
                vector_db.add_knowledge_file(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    content=content,
                    embedding=embedding
                )
                logger.info("Reindexed: %s (%s chars)", file_path.name, len(content))
                processed += 1
            except Exception as e:
                logger.error("Failed to index %s: %s", file_path.name, e)
                errors += 1
                
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path.name, e, exc_info=True)
            errors += 1
    
    logger.info("Reindexing complete: %s processed, %s skipped, %s errors", processed, skipped, errors)


if __name__ == "__main__":
    knowledge_dir = Path(__file__).parent.parent / "data" / "knowledge"
    reindex_knowledge_files(knowledge_dir)

