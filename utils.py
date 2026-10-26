import re
import fitz
import pytesseract

from PIL import Image
from pdf2image import convert_from_path

# TESSERACT PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# CLEAN TEXT
def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[^a-zA-Z0-9.,; ]', '', text)

    return text.strip()

# PDF TEXT EXTRACTION
def extract_text_from_pdf(pdf_path):

    text = ""

    pdf = fitz.open(pdf_path)

    for page in pdf:

        text += page.get_text()

    return text

# OCR
def extract_text_using_ocr(file_path):

    text = ""

    # IMAGE
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):

        image = Image.open(file_path)

        text = pytesseract.image_to_string(image)

    # PDF
    elif file_path.lower().endswith('.pdf'):

        pages = convert_from_path(
            file_path,
            poppler_path=r"C:\Users\itsme\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"
        )

        for page in pages:

            text += pytesseract.image_to_string(page)

    return text