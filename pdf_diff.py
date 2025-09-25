import fitz  # PyMuPDF
import difflib
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


def extract_text_by_page(pdf_file):
    """Extract text page by page from a PDF file-like object."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()
    pdf_file.seek(0)  # Reset pointer
    return pages_text


def highlight_changes(doc, diffs, page_num):
    """Apply highlights for inserted/changed text on a page."""
    page = doc[page_num]
    for diff in diffs:
        code, text = diff
        if code == "insert":
            color = (0, 1, 0)  # green
            opacity = 0.5
        elif code == "replace":
            color = (1, 0, 0)  # red
            opacity = 0.5
        else:
            continue  # skip deletes

        areas = page.search_for(text)
        for area in areas:
            annot = page.add_highlight_annot(area)
            annot.set_colors(stroke=color, fill=color)
            annot.set_opacity(opacity)
            annot.update()


def compare_texts(old_text, new_text):
    """Return list of diffs and removed text summary."""
    seqm = difflib.SequenceMatcher(None, old_text.split(), new_text.split())
    diffs = []
    removed = []

    for tag, i1, i2, j1, j2 in seqm.get_opcodes():
        if tag == "insert":
            diffs.append(("insert", " ".join(new_text.split()[j1:j2])))
        elif tag == "replace":
            diffs.append(("replace", " ".join(new_text.split()[j1:j2])))
            removed.append(" ".join(old_text.split()[i1:i2]))
        elif tag == "delete":
            removed.append(" ".join(old_text.split()[i1:i2]))

    return diffs, removed


def generate_annotated_pdf(old_pdf, new_pdf):
    """Generate annotated PDF (with highlights + summary panel)."""
    old_pages = extract_text_by_page(old_pdf)
    new_pages = extract_text_by_page(new_pdf)

    doc = fitz.open(stream=new_pdf.read(), filetype="pdf")
    new_pdf.seek(0)

    removed_summary = []

    for i, page in enumerate(doc):
        if i < len(old_pages):
            diffs, removed = compare_texts(old_pages[i], new_pages[i])
            highlight_changes(doc, diffs, i)
            removed_summary.extend(removed)

    # Save to buffer
    buf = BytesIO()
    doc.save(buf)
    doc.close()

    # Append summary panel
    final_buf = BytesIO()
    summary_doc = SimpleDocTemplate(final_buf, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = [Paragraph("PDF Comparison Summary", styles["Heading1"]), Spacer(1, 12)]

    if removed_summary:
        data = [["Removed Text"]]
        for r in removed_summary:
            data.append([r])
        table = Table(data, colWidths=[450])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.red),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No removed text detected.", styles["Normal"]))

    summary_doc.build(elements)

    # Merge highlights + summary
    final_buf.seek(0)
    annotated_pdf = buf.getvalue() + final_buf.getvalue()
    return annotated_pdf, "\n".join(removed_summary)


def generate_side_by_side_pdf(old_pdf, new_pdf):
    """Generate side-by-side PDF for preview (no highlights)."""
    old_doc = fitz.open(stream=old_pdf.read(), filetype="pdf")
    new_doc = fitz.open(stream=new_pdf.read(), filetype="pdf")
    old_pdf.seek(0)
    new_pdf.seek(0)

    result = fitz.open()

    for i in range(max(len(old_doc), len(new_doc))):
        old_page = old_doc[i] if i < len(old_doc) else None
        new_page = new_doc[i] if i < len(new_doc) else None

        rect = fitz.Rect(0, 0, 595 * 2, 842)  # A4 width * 2
        result_page = result.new_page(width=rect.width, height=rect.height)

        if old_page:
            pix = old_page.get_pixmap(matrix=fitz.Matrix(1, 1))
            result_page.insert_image(fitz.Rect(0, 0, 595, 842), pixmap=pix)

        if new_page:
            pix = new_page.get_pixmap(matrix=fitz.Matrix(1, 1))
            result_page.insert_image(fitz.Rect(595, 0, 1190, 842), pixmap=pix)

    buf = BytesIO()
    result.save(buf)
    result.close()
    return buf.getvalue()





