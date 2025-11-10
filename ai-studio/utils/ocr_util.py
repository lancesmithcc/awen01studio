"""
OCR utility for extracting text from PDF files using Tesseract.
Also supports text-based PDFs using PyPDF2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None  # type: ignore
    convert_from_path = None  # type: ignore
    Image = None  # type: ignore

try:
    import PyPDF2
    PDF_TEXT_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_TEXT_EXTRACTION_AVAILABLE = False
    PyPDF2 = None  # type: ignore

from utils.logger_util import get_logger

logger = get_logger("utils.ocr")


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from a PDF file.
    First tries text extraction (for text-based PDFs), then falls back to OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string, or None if extraction fails
    """
    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        return None
    
    # First, try text extraction (for text-based PDFs)
    if PDF_TEXT_EXTRACTION_AVAILABLE:
        try:
            logger.debug("Attempting text extraction from PDF: %s", pdf_path.name)
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_parts.append(text.strip())
                    except Exception as e:
                        logger.debug("Failed to extract text from page %s: %s", page_num + 1, e)
                        continue
                
                if text_parts:
                    full_text = "\n\n".join(text_parts)
                    logger.info("Extracted %s characters from %s pages using text extraction", len(full_text), len(text_parts))
                    return full_text
        except Exception as e:
            logger.debug("Text extraction failed, trying OCR: %s", e)
    
    # Fallback to OCR if text extraction didn't work
    if not OCR_AVAILABLE:
        logger.warning("OCR not available and text extraction failed for: %s", pdf_path.name)
        return None
    
    try:
        logger.info("Extracting text from PDF using OCR: %s", pdf_path.name)
        
        # Convert PDF to images (one per page)
        images = convert_from_path(str(pdf_path), dpi=300)
        
        # Extract text from each page
        extracted_texts = []
        for i, image in enumerate(images):
            logger.debug("Processing page %s of %s", i + 1, len(images))
            # Use OCR to extract text
            text = pytesseract.image_to_string(image, lang='eng')
            if text.strip():
                extracted_texts.append(text.strip())
        
        full_text = "\n\n".join(extracted_texts)
        logger.info("Extracted %s characters from %s pages using OCR", len(full_text), len(images))
        
        return full_text if full_text.strip() else None
        
    except Exception as e:
        logger.error("Failed to extract text from PDF %s: %s", pdf_path, e, exc_info=True)
        return None


def extract_text_from_image(image_path: Path) -> Optional[str]:
    """
    Extract text from an image file using OCR (Tesseract).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text as a string, or None if extraction fails
    """
    if not OCR_AVAILABLE:
        logger.warning("OCR dependencies not available. Install pytesseract.")
        return None
    
    if not image_path.exists():
        logger.error("Image file not found: %s", image_path)
        return None
    
    try:
        logger.info("Extracting text from image: %s", image_path.name)
        image = Image.open(str(image_path))
        text = pytesseract.image_to_string(image, lang='eng')
        logger.info("Extracted %s characters from image", len(text))
        return text.strip() if text.strip() else None
    except Exception as e:
        logger.error("Failed to extract text from image %s: %s", image_path, e, exc_info=True)
        return None

