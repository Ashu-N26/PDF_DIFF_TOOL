from difflib import ndiff
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def compare_pdfs(old_pdf, new_pdf, output_folder):
    # Dummy comparison logic (to be replaced with full PDF parsing + diff)
    # For now: just text diff simulation
    with open(old_pdf, "rb") as f1, open(new_pdf, "rb") as f2:
        old_text = f1.read().decode(errors="ignore")
        new_text = f2.read().decode(errors="ignore")

    diff = list(ndiff(old_text.split(), new_text.split()))

    annotated_path = os.path.join(output_folder, "annotated.pdf")
    c = canvas.Canvas(annotated_path, pagesize=letter)
    y = 750
    for line in diff:
        if line.startswith("-"):
            c.setFillColorRGB(1, 0, 0)  # Red for removed
        elif line.startswith("+"):
            c.setFillColorRGB(0, 0.6, 0)  # Green for added
        else:
            c.setFillColorRGB(0, 0, 0)
        c.drawString(30, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = 750
    c.save()

    side_by_side_path = os.path.join(output_folder, "side_by_side.pdf")
    c = canvas.Canvas(side_by_side_path, pagesize=letter)
    c.drawString(100, 750, "Old PDF (raw view)")
    c.drawString(350, 750, "New PDF (with highlights)")
    c.save()

    summary_path = os.path.join(output_folder, "summary.pdf")
    c = canvas.Canvas(summary_path, pagesize=letter)
    c.drawString(200, 750, "Summary Panel: DA/MDA, VIS, RVR changes")
    c.save()

    return "annotated.pdf", "side_by_side.pdf", "summary.pdf"
