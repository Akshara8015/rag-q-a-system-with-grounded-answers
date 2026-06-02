import io
import logging
import os
from PyPDF2 import PageObject, PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import clean_text

logger = logging.getLogger(__name__)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)


def extract_text_from_pdf(file_path: str):
    """
    Extracts text from a PDF file, cleans the content, and returns chunked data with metadata.
    Args:
        file_path (str): Path to the PDF file.
    Returns:
        List[dict]: A list of chunks with text and metadata.
    """
    final_chunks = []
    source = os.path.basename(file_path)
    global_chunk_id = 0

    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        logger.info(f"Opened PDF file for text extraction: {file_path}")

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(f"Could not extract page text for page {page_number}: {exc}")
                page_text = ""

            if not page_text:
                logger.info(f"No direct text on page {page_number}, trying OCR.")
                page_text = extract_text_from_images(page)
            else:
                logger.info(f"Extracted text from page {page_number} without OCR.")

            cleaned_text = clean_text(page_text)
            page_chunks = splitter.split_text(cleaned_text)

            for chunk in page_chunks:
                final_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "page": page_number,
                        "source": source,
                        "chunk_id": global_chunk_id,
                    },
                })
                global_chunk_id += 1

    logger.info(f"Completed text extraction for {file_path} with {len(final_chunks)} chunks")
    return final_chunks


def extract_text_from_images(page: PageObject) -> str:
    
    """
    Extracts text from images on a page using OCR.

    Args:
        page (PageObject): The PDF page object containing images.

    Returns:
        str: Extracted text from images using OCR.
    """
    text = ""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        logger.warning("OCR dependencies are not installed, skipping image OCR: %s", exc)
        return text

    for image_file_object in getattr(page, "images", []):
        try:
            image = Image.open(io.BytesIO(image_file_object.data))
            ocr_text = pytesseract.image_to_string(image)
            text += ocr_text
            logger.info("Extracted text from image using OCR.")
        except Exception as e:
            logger.error(f"Error processing image for OCR: {e}")
    return text



