"""
chunker.py — Splits large documents into smaller overlapping chunks.

WHY DO WE CHUNK?
  Language models have a context window limit — you can't feed them
  an entire 50-page PDF. Chunking breaks the document into pieces
  small enough to embed and retrieve individually.

WHY OVERLAP?
  If we split cleanly at 500 characters, an answer that spans the
  boundary of two chunks would be lost. Overlap (e.g. 50 chars)
  means each chunk shares context with its neighbours — no answers
  fall through the cracks.

ANALOGY:
  Imagine highlighting a textbook. You don't highlight the whole page —
  you highlight the key paragraphs. But you make sure each highlight
  starts a few words before the last one ended, so nothing is missed.
"""

import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split a string into overlapping chunks of a fixed character size.

    Args:
        text:          The full document text to split.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: How many characters each chunk shares with the previous one.

    Returns:
        A list of text chunks. Empty input returns an empty list.

    Example:
        chunk_text("abcdefghij", chunk_size=5, chunk_overlap=2)
        → ["abcde", "defgh", "ghij"]
    """
    if not text or not text.strip():
        logger.warning("chunk_text received empty or whitespace-only text")
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Don't cut words in half — walk back to the last space
        # so chunks end on a word boundary (better for embeddings)
        if end < text_length:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:  # Skip chunks that are pure whitespace
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so the next chunk
        # starts inside the current one — creating the overlap
        start += chunk_size - chunk_overlap

    logger.debug(f"Chunked text into {len(chunks)} chunks "
                 f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def extract_text_from_file(content: bytes, file_type: str) -> str:
    """
    Extract plain text from uploaded file bytes.

    Supports:
      - "text" : UTF-8 text files (.txt, .md, etc.)
      - "pdf"  : PDF files parsed with pypdf

    Args:
        content:   Raw file bytes from the upload.
        file_type: Either "text" or "pdf".

    Returns:
        Extracted plain text string.

    Raises:
        ValueError: If the file type is unsupported or extraction fails.
    """
    if file_type == "text":
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"File is not valid UTF-8 text: {e}") from e

    elif file_type == "pdf":
        try:
            import io
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages_text = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
                else:
                    logger.warning(f"PDF page {page_num + 1} yielded no text "
                                   "(may be scanned/image-based)")

            full_text = "\n\n".join(pages_text)

            if not full_text.strip():
                raise ValueError(
                    "PDF appears to contain no extractable text. "
                    "Scanned PDFs (images) are not supported."
                )

            return full_text

        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}") from e

    else:
        raise ValueError(
            f"Unsupported file type: '{file_type}'. "
            "Supported types: 'text', 'pdf'"
        )