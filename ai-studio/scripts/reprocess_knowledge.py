"""
Script to reprocess existing knowledge files and extract/cache their text content.
This is useful when files were uploaded before OCR caching was implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ocr_util import extract_text_from_pdf
from utils.logger_util import get_logger

logger = get_logger("scripts.reprocess_knowledge")


def reprocess_knowledge_files(knowledge_dir: Path) -> None:
    """
    Reprocess all knowledge files and cache their extracted text.
    
    Args:
        knowledge_dir: Directory containing knowledge files
    """
    if not knowledge_dir.exists():
        logger.error("Knowledge directory does not exist: %s", knowledge_dir)
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
        
        cache_path = knowledge_dir / f"{file_path.name}.txt"
        
        # Skip if already cached
        if cache_path.exists():
            logger.debug("Already cached: %s", file_path.name)
            skipped += 1
            continue
        
        try:
            content = ""
            
            if file_path.suffix.lower() in ['.txt', '.md']:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                logger.info("Extracted text from %s (%s chars)", file_path.name, len(content))
            elif file_path.suffix.lower() == '.pdf':
                logger.info("Processing PDF: %s (this may take a while)...", file_path.name)
                content = extract_text_from_pdf(file_path)
                if content:
                    logger.info("Extracted %s characters from %s", len(content), file_path.name)
                else:
                    logger.warning("No content extracted from %s", file_path.name)
            else:
                logger.debug("Skipping unsupported file type: %s", file_path.name)
                skipped += 1
                continue
            
            if content:
                # Cache the content
                cache_path.write_text(content, encoding='utf-8')
                logger.info("Cached content for: %s", file_path.name)
                processed += 1
            else:
                logger.warning("No content to cache for: %s", file_path.name)
                skipped += 1
                
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path.name, e, exc_info=True)
            errors += 1
    
    logger.info("Reprocessing complete: %s processed, %s skipped, %s errors", processed, skipped, errors)


if __name__ == "__main__":
    knowledge_dir = Path(__file__).parent.parent / "data" / "knowledge"
    reprocess_knowledge_files(knowledge_dir)

