import difflib
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io


# ----------------------------
# Extract text from PDF
# ----------------------------
def extract_text_from_pdf(pdf_file):
    """Extracts text page by page from PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text.splitlines())
    return pages


# ----------------------------
# Token level diff
# ----------------------------
def diff_tokens(old_line, new_line):
    """
    Compare two lines at the word level.
    Returns a list of (token, status) tuples:
      - status = "insert", "delete", "replace", or "equal"
    """
    diff = difflib.SequenceMatcher(None, old_line.split(), new_line.split())
    result = []

    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        if tag == "equal":
            for word in old_line.split()[i1:i2]:
                result.append((word, "equal"))
        elif tag == "replace":
            for word in old_line.split()[i1:i2]:
                result.append((word, "delete"))
            for word in new_line.split()[j1:j2]:
                result.append((word, "insert"))
        elif tag == "delete":
            for word in old_line.split()[i1:i2]:
                result.append((word, "delete"))
        elif tag == "insert":
            for word in new_line.split()[j1:j2]:
                result.append((word, "insert"))

    return result


# ----------------------------
# Generate Annotated PDF
# ----------------------------
def generate_annotated_pdf(old_pdf, new_pdf, output_path):
    old_pages = extract_text_from_pdf(old_pdf)
    new_pages = extract_text_from_pdf(new_pdf)

    doc = fitz.open()
    removed_data = []

    num_pages = max(len(old_pages), len(new_pages))

    for page_num in range(num_pages):
        page = doc.new_page()
        cursor_y = 50

        old_lines = old_pages[page_num] if page_num < len(old_pages) else []
        new_lines = new_pages[page_num] if page_num < len(new_pages) else []

        max_lines = max(len(old_lines), len(new_lines))

        for line_idx in range(max_lines):
            old_line = old_lines[line_idx] if line_idx < len(old_lines) else ""
            new_line = new_lines[line_idx] if line_idx < len(new_lines) else ""

            tokens = diff_tokens(old_line, new_line)

            cursor_x = 50
            for token, status in tokens:
                if status == "equal":
                    page.insert_text((cursor_x, cursor_y), token + " ", fontsize=11, color=(0, 0, 0))
                elif status == "insert":
                    page.insert_text(
                        (cursor_x, cursor_y),
                        token + " ",
                        fontsize=11,
                        color=(0, 1, 0),  # Green
                        fill_opacity=0.5,
                    )
                elif status == "delete":
                    removed_data.append(token)
                cursor_x += len(token) * 4  # Rough spacing

            cursor_y += 15

    # Add summary page
    summary = doc.new_page()
    summary.insert_text((50, 50), "Summary of Changes", fontsize=14, color=(0, 0, 0))
    y = 80
    for item in removed_data:
        summary.insert_text((50, y), f"Removed: {item}", fontsize=11, color=(1, 0, 0))  # Red
        y += 15

    doc.save(output_path)


# ----------------------------
# Generate Side by Side PDF
# ----------------------------
def generate_side_by_side_pdf(old_pdf, new_pdf, output_path):
    old_pages = extract_text_from_pdf(old_pdf)
    new_pages = extract_text_from_pdf(new_pdf)

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)

    width, height = letter
    left_margin = 0.5 * inch
    right_margin = width / 2 + 0.5 * inch
    line_height = 12
    y_start = height - 50

    num_pages = max(len(old_pages), len(new_pages))

    for page_num in range(num_pages):
        old_lines = old_pages[page_num] if page_num < len(old_pages) else []
        new_lines = new_pages[page_num] if page_num < len(new_pages) else []

        max_lines = max(len(old_lines), len(new_lines))
        y = y_start

        c.setFont("Helvetica", 10)
        c.drawString(left_margin, height - 30, f"OLD PDF - Page {page_num + 1}")
        c.drawString(right_margin, height - 30, f"NEW PDF - Page {page_num + 1}")

        for line_idx in range(max_lines):
            old_line = old_lines[line_idx] if line_idx < len(old_lines) else ""
            new_line = new_lines[line_idx] if line_idx < len(new_lines) else ""

            if y < 50:
                c.showPage()
                y = y_start
                c.setFont("Helvetica", 10)

            c.drawString(left_margin, y, old_line.strip())
            c.drawString(right_margin, y, new_line.strip())
            y -= line_height

        c.showPage()

    c.save()

    with open(output_path, "wb") as f:
        f.write(packet.getvalue())







