import logging
import os
from io import BytesIO

import pypandoc
from docx import Document
from docx.shared import Pt

logger = logging.getLogger(__name__)

def create_meeting_minutes_doc_buffer(markdown_content: str) -> BytesIO:
    """
    Converts a markdown string into an in-memory DOCX file buffer.
    """
    try:
        logger.info("Attempting DOCX conversion with Pandoc...")
        docx_bytes = pypandoc.convert_text(
            markdown_content,
            to='docx',
            format='md',
            outputfile=None  
        )
        buffer = BytesIO(docx_bytes)
        logger.info("Pandoc conversion successful!")
        return buffer
    except Exception as e:
        logger.error(f"Pandoc conversion failed: {e}. Falling back to basic conversion.")
        try:
            doc = Document()
            doc.add_paragraph(markdown_content)
            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return buffer
        except Exception as fallback_e:
            logger.error(f"Basic DOCX conversion also failed: {fallback_e}", exc_info=True)
            raise RuntimeError("Failed to generate DOCX document.")
