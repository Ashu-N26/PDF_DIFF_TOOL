import streamlit as st
import fitz  # PyMuPDF
from difflib import SequenceMatcher
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfMerger
import io
import os

# -----------------------------
# Utility functions
# -----------------------------

def extract_text_from_pdf(file):
    """Extract text from all pages of a PDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = [page.get_text("text") for page in doc]
    return text, doc

def compare_texts(old_text, new_text):
    """Compare two texts and return differences with labels."""
    matcher = SequenceMatcher(None, old_text, new_text)
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            changes.append(("changed", old_text[i1:i2], new_text[j1:j2]))
        elif tag == "insert":
            changes.append(("inserted", "", new_text[j1:j2]))
        elif tag == "delete":
            changes.append(("removed", old_text[i1:i2], ""))
        else:
            changes.append(("unchanged", old_text[i1:i2], new_text[j1:j2]))
    return changes

def annotate_pdf(doc, changes):
    """Annotate PDF with highlights (red for changed, green for inserted)."""
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        for change_type, old, new in changes:
            if change_type == "changed" and new.strip():
                areas = page.search_for(new.strip())
                for area in areas:
                    highlight = page.add_highlight_annot(area)
                    highlight.set_colors(stroke=(1, 0, 0))  # Red
                    highlight.update()
            elif change_type == "inserted" and new.strip():
                areas = page.search_for(new.strip())
                for area in areas:
                    highlight = page.add_highlight_annot(area)
                    highlight.set_colors(stroke=(0, 1, 0))  # Green
                    highlight.update()
    return doc

def build_summary_pdf(changes):
    """Generate a summary panel PDF with comparison table."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    data = [["Type", "Old Text", "New Text"]]
    for change_type, old, new in changes:
        if change_type == "changed":
            data.append(
                [Paragraph("<font color='red'>Changed</font>", styles["Normal"]),
                 old, new]
            )
        elif change_type == "inserted":
            data.append(
                [Paragraph("<font color='green'>Inserted</font>", styles["Normal"]),
                 "-", new]
            )
        elif change_type == "removed":
            data.append(
                [Paragraph("<font color='red'>Removed</font>", styles["Normal"]),
                 old, "-"]
            )
        elif change_type == "unchanged":
            data.append(["Unchanged", old, new])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    doc.build([Paragraph("Summary Panel", styles["Title"]), table])
    buffer.seek(0)
    return buffer

def merge_pdfs(main_pdf, summary_pdf):
    """Append summary PDF to the end of main PDF."""
    merger = PdfMerger()
    merger.append(main_pdf)
    merger.append(summary_pdf)
    output = io.BytesIO()
    merger.write(output)
    merger.close()
    output.seek(0)
    return output

# -----------------------------
# Streamlit Interface
# -----------------------------
st.set_page_config(page_title="PDF Compare Tool", layout="wide")
st.title("📄 Advanced PDF Compare Tool")
st.markdown("<p style='text-align:center;'>Created by Ashutosh Nanaware</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    old_pdf = st.file_uploader("Upload Old PDF", type="pdf")
with col2:
    new_pdf = st.file_uploader("Upload New PDF", type="pdf")

if old_pdf and new_pdf:
    if st.button("🔍 Compare PDFs"):
        old_text, old_doc = extract_text_from_pdf(old_pdf)
        new_text, new_doc = extract_text_from_pdf(new_pdf)

        # Compare page by page
        changes = []
        for i in range(max(len(old_text), len(new_text))):
            old_pg = old_text[i] if i < len(old_text) else ""
            new_pg = new_text[i] if i < len(new_text) else ""
            changes.extend(compare_texts(old_pg, new_pg))

        # Annotate new PDF
        annotated_doc = annotate_pdf(new_doc, changes)
        annotated_output = io.BytesIO()
        annotated_doc.save(annotated_output)
        annotated_output.seek(0)

        # Summary panel
        summary_pdf = build_summary_pdf(changes)
        final_pdf = merge_pdfs(annotated_output, summary_pdf)

        # Side by Side (basic version - showing extracted text)
        st.subheader("📑 Side-by-Side View")
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Old PDF", "\n".join(old_text), height=400)
        with col2:
            st.text_area("New PDF (Highlighted)", "\n".join(new_text), height=400)

        # Downloads
        st.download_button("⬇️ Download Annotated PDF (with Summary)", final_pdf,
                           file_name="Annotated_Comparison.pdf", mime="application/pdf")
        st.download_button("⬇️ Download Side-by-Side Text", "\n".join(new_text),
                           file_name="SideBySide.txt")

