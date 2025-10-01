import streamlit as st
from PyPDF2 import PdfReader
import difflib
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(
    page_title="PDF_DIFF TOOL",
    page_icon="📑",
    layout="wide"
)

# Title
st.title("PDF_DIFF TOOL")

# Sidebar options
st.sidebar.header("Options")
opacity = st.sidebar.slider("Highlight opacity (%)", 10, 100, 50)
debug_mode = st.sidebar.checkbox("Show debug state")

# Function: Extract text from PDF using PyPDF2
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return text

# Function: OCR fallback using pytesseract (for scanned PDFs)
def extract_text_with_ocr(file):
    text = []
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img)
        text.append(ocr_text)
    return text

# Compare PDFs function
def compare_pdfs(old_pdf, new_pdf):
    old_text = extract_text_from_pdf(old_pdf)
    new_text = extract_text_from_pdf(new_pdf)

    # If no text extracted, fallback to OCR
    if not any(old_text):
        old_pdf.seek(0)
        old_text = extract_text_with_ocr(old_pdf)

    if not any(new_text):
        new_pdf.seek(0)
        new_text = extract_text_with_ocr(new_pdf)

    diff_results = []
    for i, (old_page, new_page) in enumerate(zip(old_text, new_text)):
        d = difflib.Differ()
        diff = list(d.compare(old_page.splitlines(), new_page.splitlines()))
        diff_results.append((i + 1, diff))
    return diff_results

# Upload widgets
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload OLD PDF")
    old_file = st.file_uploader("Drag and drop file here", type="pdf", key="old")

with col2:
    st.subheader("Upload NEW PDF")
    new_file = st.file_uploader("Drag and drop file here", type="pdf", key="new")

# Process comparison
if old_file and new_file:
    st.info("Comparing PDFs... please wait.")
    diffs = compare_pdfs(old_file, new_file)

    if diffs:
        st.success("Comparison complete!")
        for page_num, diff in diffs:
            st.write(f"### Page {page_num}")
            diff_html = "<pre style='font-size:13px;'>"
            for line in diff:
                if line.startswith("+"):
                    diff_html += f"<span style='background-color:rgba(0,255,0,{opacity/100});'>{line}</span>\n"
                elif line.startswith("-"):
                    diff_html += f"<span style='background-color:rgba(255,0,0,{opacity/100});'>{line}</span>\n"
                else:
                    diff_html += line + "\n"
            diff_html += "</pre>"
            st.markdown(diff_html, unsafe_allow_html=True)
    else:
        st.warning("No differences found.")
else:
    st.info("Upload both OLD and NEW PDFs to begin comparison.")

# Debug info
if debug_mode:
    st.subheader("Debug Information")
    st.write("Old file:", old_file)
    st.write("New file:", new_file)

# ✅ Footer (kept)
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>&copy; 2025 Created by Ashutosh Nanaware. All rights reserved.</p>",
    unsafe_allow_html=True
)






