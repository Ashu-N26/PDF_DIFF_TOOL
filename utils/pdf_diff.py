# utils/pdf_diff.py
import os
import io
import tempfile
import shutil
import logging
from difflib import SequenceMatcher, HtmlDiff, unified_diff
from typing import List, Dict, Tuple
import pdfplumber
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from PIL import Image, ImageDraw
import pytesseract

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def ensure_dirs(paths):
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        os.makedirs(p, exist_ok=True)

# -------------------------
# Text Extraction helpers
# -------------------------
def extract_text_pdfplumber(path: str) -> str:
    try:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text()
                if ptext:
                    text_parts.append(ptext)
        return "\n".join(text_parts).strip()
    except Exception as e:
        logger.warning("pdfplumber failed: %s", e)
        return ""

def extract_text_pymupdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        parts = []
        for p in doc:
            try:
                parts.append(p.get_text("text") or "")
            except Exception:
                parts.append("")
        doc.close()
        return "\n".join(parts).strip()
    except Exception as e:
        logger.warning("PyMuPDF extract failed: %s", e)
        return ""

def extract_text(path: str) -> str:
    text = extract_text_pdfplumber(path)
    if text:
        return text
    return extract_text_pymupdf(path)

# -------------------------
# OCR (scanned PDFs)
# -------------------------
def ocr_pdf_to_text(path: str, dpi: int = 200) -> str:
    """
    Convert PDF pages to images, OCR each and return joined text.
    Requires poppler (pdf2image) and tesseract installed.
    """
    texts = []
    try:
        images = convert_from_path(path, dpi=dpi)
        for img in images:
            txt = pytesseract.image_to_string(img)
            texts.append(txt)
        return "\n".join(texts).strip()
    except Exception as e:
        logger.exception("OCR failed: %s", e)
        return ""

# -------------------------
# Diffs
# -------------------------
def compute_line_diffs(old_text: str, new_text: str) -> Dict[str, List[str]]:
    """
    Returns dict with keys 'removed', 'inserted', 'changed' (lines)
    We derive inserted/removed lines from unified_diff
    """
    removed, inserted = [], []
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    for line in unified_diff(old_lines, new_lines, lineterm=""):
        if line.startswith("- ") or line.startswith("---"):
            # skip headers
            if line.startswith("- ") and not line.startswith("---"):
                removed.append(line[2:])
        elif line.startswith("+ ") or line.startswith("+++"):
            if line.startswith("+ ") and not line.startswith("+++"):
                inserted.append(line[2:])
    return {"removed": removed, "inserted": inserted}

def generate_text_diff_html(old_text: str, new_text: str) -> str:
    diff = HtmlDiff(wrapcolumn=100)
    return diff.make_table(old_text.splitlines(), new_text.splitlines(), fromdesc="Old", todesc="New")

# -------------------------
# PDF Generators
# -------------------------
def generate_colored_text_pdf(changes: Dict[str, List[str]], out_path: str, title: str = "Changes Summary"):
    """
    Create a simple PDF listing removed (red) and inserted (green) lines.
    This is a safe lightweight annotated textual report.
    """
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 24
    c.setFont("Helvetica", 10)

    # Removed (red)
    if changes.get("removed"):
        c.setFillColorRGB(0.9, 0.2, 0.2)
        c.drawString(margin, y, "Removed / Changed (in old -> missing in new):")
        y -= 14
        c.setFillColorRGB(0, 0, 0)
        for line in changes["removed"]:
            if y < 60:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            # red background box
            c.setFillColorRGB(1, 0.9, 0.9)
            c.rect(margin - 2, y - 2, width - 2 * margin + 4, 12, stroke=0, fill=1)
            c.setFillColorRGB(0.8, 0.0, 0.0)
            c.drawString(margin, y, (line[:120]))
            y -= 14
            c.setFillColorRGB(0, 0, 0)

    # Inserted (green)
    if changes.get("inserted"):
        if y < 60:
            c.showPage()
            y = height - margin
        c.setFillColorRGB(0.2, 0.65, 0.2)
        c.drawString(margin, y, "Inserted / New (present in new PDF):")
        y -= 14
        c.setFillColorRGB(0, 0, 0)
        for line in changes["inserted"]:
            if y < 60:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            c.setFillColorRGB(0.95, 1.0, 0.95)
            c.rect(margin - 2, y - 2, width - 2 * margin + 4, 12, stroke=0, fill=1)
            c.setFillColorRGB(0.03, 0.55, 0.03)
            c.drawString(margin, y, (line[:120]))
            y -= 14
            c.setFillColorRGB(0, 0, 0)

    c.showPage()
    c.save()

def make_side_by_side_pdf(old_pdf: str, new_pdf: str, out_path: str, max_pages: int = None, dpi: int = 150):
    """
    Convert each PDF page to images and place side-by-side.
    This is robust and works for scanned and normal PDFs.
    """
    # convert pages
    old_images = convert_from_path(old_pdf, dpi=dpi)
    new_images = convert_from_path(new_pdf, dpi=dpi)

    if max_pages:
        old_images = old_images[:max_pages]
        new_images = new_images[:max_pages]

    # pad one list to the other in case of different page counts
    n = max(len(old_images), len(new_images))
    while len(old_images) < n:
        old_images.append(Image.new("RGB", new_images[0].size, (255, 255, 255)))
    while len(new_images) < n:
        new_images.append(Image.new("RGB", old_images[0].size, (255, 255, 255)))

    # Create PDF with side-by-side images using reportlab canvas
    c = canvas.Canvas(out_path, pagesize=letter)
    page_w, page_h = letter
    margin = 20

    for oimg, nimg in zip(old_images, new_images):
        # scale images to fit half page
        half_w = (page_w - 3 * margin) / 2
        max_h = page_h - 2 * margin

        # compute scaling
        o_w, o_h = oimg.size
        scale_o = min(half_w / o_w, max_h / o_h)
        n_w, n_h = nimg.size
        scale_n = min(half_w / n_w, max_h / n_h)

        # resize for embedding
        o_resized = oimg.resize((int(o_w * scale_o), int(o_h * scale_o)))
        n_resized = nimg.resize((int(n_w * scale_n), int(n_h * scale_n)))

        # temp files
        tmp_o = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_n = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            o_resized.save(tmp_o.name, format="PNG")
            n_resized.save(tmp_n.name, format="PNG")

            # left image
            left_x = margin
            left_y = page_h - margin - o_resized.size[1]
            c.drawImage(tmp_o.name, left_x, left_y, width=o_resized.size[0], height=o_resized.size[1])

            # right image
            right_x = margin * 2 + half_w
            right_y = page_h - margin - n_resized.size[1]
            c.drawImage(tmp_n.name, right_x, right_y, width=n_resized.size[0], height=n_resized.size[1])

            # footer labels
            c.setFont("Helvetica", 8)
            c.drawString(left_x, margin / 2 + 5, "Old PDF")
            c.drawString(right_x, margin / 2 + 5, "New PDF")

            c.showPage()
        finally:
            tmp_o.close()
            tmp_n.close()
            try:
                os.unlink(tmp_o.name)
                os.unlink(tmp_n.name)
            except Exception:
                pass

    c.save()

# -------------------------
# Top-level orchestrator
# -------------------------
def compare_pdfs(
    old_pdf: str,
    new_pdf: str,
    output_folder: str,
    do_ocr: bool = False,
    enable_page_annotations: bool = False,
    prefix: str = ""
) -> Dict[str, str]:
    """
    Compare two PDFs and generate:
      - annotated_text_report.pdf  (colored removed/inserted lines)
      - side_by_side.pdf
      - merged_report.pdf (summary + side_by_side)
    Returns dict with output file paths.
    """

    ensure_dirs([output_folder])
    # 1) Extract text
    old_text = extract_text(old_pdf)
    new_text = extract_text(new_pdf)

    # If both empty and OCR allowed -> run OCR
    if not old_text and not new_text and do_ocr:
        # OCR both
        old_text = ocr_pdf_to_text(old_pdf) or ""
        new_text = ocr_pdf_to_text(new_pdf) or ""

    # If still empty: warn and proceed with image-only side-by-side
    if not old_text and not new_text:
        logger.info("No extractable text found; generating side-by-side only")
        sb_path = os.path.join(output_folder, f"{prefix}side_by_side.pdf")
        make_side_by_side_pdf(old_pdf, new_pdf, sb_path)
        return {
            "side_by_side": sb_path,
            "annotated_text_report": "",
            "merged_report": sb_path
        }

    # 2) Compute diffs
    diffs = compute_line_diffs(old_text, new_text)

    # 3) Create annotated textual PDF summary
    annotated_text_path = os.path.join(output_folder, f"{prefix}annotated_text_report.pdf")
    generate_colored_text_pdf(diffs, annotated_text_path, title="PDF Diff - Textual Summary")

    # 4) Create side-by-side PDF (images)
    side_by_side_path = os.path.join(output_folder, f"{prefix}side_by_side.pdf")
    make_side_by_side_pdf(old_pdf, new_pdf, side_by_side_path)

    # 5) Create merged report: first page embed annotated_text_report as page, then side-by-side pages
    merged_path = os.path.join(output_folder, f"{prefix}merged_report.pdf")
    try:
        # We'll append pages by converting annotated_text_report to an image and then adding
        # (Simpler: create a new PDF that first draws annotated text report as one page (embedding PDF as image)
        # then embed each side_by_side page image. Here we just create a simple merged PDF:
        c = canvas.Canvas(merged_path, pagesize=letter)
        # Summary page (draw a title and note)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, 750, "PDF DIFF - Merged Report")
        c.setFont("Helvetica", 10)
        c.drawString(40, 730, f"Generated summary (text diff) and side-by-side pages.")
        c.showPage()

        # Then insert the annotated_text_report as a single page image snapshot
        # fallback: draw text lines from annotated_text_report (since reading PDF pages requires PyPDF2)
        # Read annotated_text_report page images via pdf2image and place them
        preview_images = convert_from_path(annotated_text_path, dpi=150)
        for img in preview_images:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            try:
                img.save(tmp.name, format="PNG")
                page_w, page_h = letter
                img_w, img_h = img.size
                scale = min((page_w - 80) / img_w, (page_h - 80) / img_h)
                w = img_w * scale
                h = img_h * scale
                x = (page_w - w) / 2
                y = (page_h - h) / 2
                c.drawImage(tmp.name, x, y, width=w, height=h)
                c.showPage()
            finally:
                tmp.close()
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        # Append side-by-side pages too (as images)
        sb_images = convert_from_path(side_by_side_path, dpi=150)
        for img in sb_images:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            try:
                img.save(tmp.name, format="PNG")
                page_w, page_h = letter
                img_w, img_h = img.size
                scale = min((page_w - 80) / img_w, (page_h - 80) / img_h)
                w = img_w * scale
                h = img_h * scale
                x = (page_w - w) / 2
                y = (page_h - h) / 2
                c.drawImage(tmp.name, x, y, width=w, height=h)
                c.showPage()
            finally:
                tmp.close()
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        c.save()
    except Exception as e:
        logger.exception("Failed building merged report: %s", e)
        # fallback to side_by_side as merged
        merged_path = side_by_side_path

    results = {
        "annotated_text_report": annotated_text_path,
        "side_by_side": side_by_side_path,
        "merged_report": merged_path
    }

    # Optional: attempt to annotate actual new PDF pages (heavy)
    if enable_page_annotations:
        try:
            annotated_new_pdf = os.path.join(output_folder, f"{prefix}annotated_new_pdf_with_boxes.pdf")
            _annotate_new_pdf_as_images(new_pdf, diffs, annotated_new_pdf)
            results["annotated_new_pdf"] = annotated_new_pdf
        except Exception:
            logger.exception("Page-level annotation failed; skipped.")
    return results

# -------------------------
# Optional heavy function:
# convert pages to images and draw boxes near text lines in new PDF (if run on a powerful host)
# -------------------------
def _annotate_new_pdf_as_images(pdf_path: str, diffs: dict, out_pdf: str, dpi: int = 150):
    """
    Convert pages into images and overlay bounding boxes near lines found in diffs.
    This function is optional and heavy. Use enable_page_annotations only if you have good RAM.
    """
    ensure_dirs([os.path.dirname(out_pdf) or "."])
    # get page images
    images = convert_from_path(pdf_path, dpi=dpi)
    # use PyMuPDF to search for text positions
    doc = fitz.open(pdf_path)
    out_images = []
    for pnum, img in enumerate(images):
        draw = ImageDraw.Draw(img, "RGBA")
        if pnum < len(doc):
            page = doc[pnum]
            page_w = page.rect.width
            page_h = page.rect.height
            # for each inserted/removed line try to find rects and overlay
            for line in diffs.get("inserted", []) + diffs.get("removed", []):
                if not line.strip():
                    continue
                try:
                    rects = page.search_for(line, hit_max=50)
                except Exception:
                    rects = []
                for r in rects:
                    # map page coordinates to image coords
                    # fitz uses points; image width maps to page width
                    img_w, img_h = img.size
                    fx = img_w / page_w
                    fy = img_h / page_h
                    x0 = r.x0 * fx
                    y0 = r.y0 * fy
                    x1 = r.x1 * fx
                    y1 = r.y1 * fy
                    # PyMuPDF y origin might be top-left or bottom-left — if boxes look inverted, swap
                    # Use semi-transparent fill
                    color = (0, 255, 0, 120) if line in diffs.get("inserted", []) else (255, 0, 0, 120)
                    draw.rectangle([x0, y0, x1, y1], fill=color)
        out_images.append(img)

    # save images into a PDF
    out_images[0].save(out_pdf, save_all=True, append_images=out_images[1:])






