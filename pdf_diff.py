import difflib
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    return text

def generate_annotated_pdf(old_pdf, new_pdf, output_path):
    old_text = extract_text_from_pdf(old_pdf)
    new_text = extract_text_from_pdf(new_pdf)

    # Create new PDF
    doc = fitz.open()

    removed_data = []

    for page_num, (old, new) in enumerate(zip(old_text, new_text), start=1):
        diff = difflib.ndiff(old.split(), new.split())

        page = doc.new_page()
        cursor = 50
        for token in diff:
            if token.startswith("- "):  # Removed
                removed_data.append(token[2:])
            elif token.startswith("+ "):  # Inserted
                page.insert_text((50, cursor), token[2:], fontsize=12, color=(0, 1, 0), fill_opacity=0.5)
                cursor += 15
            elif token.startswith("? "):  # Ignore ndiff markers
                continue
            else:  # Unchanged
                page.insert_text((50, cursor), token[2:], fontsize=12, color=(0, 0, 0))
                cursor += 15

    # Add summary page
    summary = doc.new_page()
    summary.insert_text((50, 50), "Summary of Changes", fontsize=14, color=(0, 0, 0))
    y = 80
    for item in removed_data:
        summary.insert_text((50, y), f"Removed: {item}", fontsize=12, color=(1, 0, 0))
        y += 15

    doc.save(output_path)

def generate_side_by_side_pdf(old_pdf, new_pdf, output_path):
    old_text = extract_text_from_pdf(old_pdf)
    new_text = extract_text_from_pdf(new_pdf)

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)

    y = 750
    for old_line, new_line in zip(old_text, new_text):
        c.drawString(50, y, f"OLD: {old_line.strip()}")
        c.drawString(300, y, f"NEW: {new_line.strip()}")
        y -= 15

    c.save()

    with open(output_path, "wb") as f:
        f.write(packet.getvalue())






