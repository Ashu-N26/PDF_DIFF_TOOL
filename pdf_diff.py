# pdf_diff.py
"""
PDF diff utilities:
- Token-level comparison with OCR fallback
- Annotated NEW PDF generation (red for changed, green for inserted, removed only in summary)
- Side-by-side PDF generation (visual preview)
"""

from typing import List, Dict, Tuple, Any
import fitz  # PyMuPDF
from difflib import SequenceMatcher
import io
import os
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import pytesseract
import math
import logging

logging.basicConfig(level=logging.INFO)
GREEN = (60/255.0, 170/255.0, 60/255.0)
RED = (1.0, 80/255.0, 80/255.0)

number_regex = r"[-+]?\d[\d,]*\.?\d*"

def _save_input_to_temp(input_obj: Any, out_path: str):
    """Accept file-like or path; write to out_path and return path."""
    if isinstance(input_obj, str):
        if not os.path.exists(input_obj):
            raise FileNotFoundError(input_obj)
        return input_obj
    # file-like (e.g., streamlit uploaded file)
    data = input_obj.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(data)
    try:
        input_obj.seek(0)
    except Exception:
        pass
    return out_path

def extract_tokens_from_page(page: fitz.Page, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Return list of tokens with bounding boxes on given page.
    Attempt to use page.get_text('words') first. If no words or use_ocr True, fallback to OCR.
    Each token: {"text": str, "bbox": (x0,y0,x1,y1)}
    """
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,wordno
    tokens = []
    if words and not use_ocr:
        words.sort(key=lambda w: (round(w[1], 1), w[0]))
        for w in words:
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            txt = str(word).strip()
            if txt:
                tokens.append({"text": txt, "bbox": (x0, y0, x1, y1)})
        if tokens:
            return tokens

    # OCR fallback: render page to image and run pytesseract
    mat = fitz.Matrix(2, 2)  # render at 2x for better OCR
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(ocr_data["text"])
    W_img, H_img = img.size
    # Map image pixels to PDF coordinates: image mapped to page.rect (width, height)
    page_rect = page.rect
    pw, ph = page_rect.width, page_rect.height
    scale_x = pw / W_img
    scale_y = ph / H_img

    for i in range(n):
        txt = ocr_data["text"][i].strip()
        if not txt:
            continue
        x = ocr_data["left"][i]
        y = ocr_data["top"][i]
        w = ocr_data["width"][i]
        h = ocr_data["height"][i]
        # caret: pytesseract coords origin (0,0) at top-left of image; PDF origin top-left as well in PyMuPDF
        x0 = x * scale_x
        y0 = y * scale_y
        x1 = (x + w) * scale_x
        y1 = (y + h) * scale_y
        # clamp
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(pw, x1), min(ph, y1)
        tokens.append({"text": txt, "bbox": (x0, y0, x1, y1)})
    return tokens

def extract_pdf_tokens(pdf_path: str) -> List[List[Dict[str, Any]]]:
    """Return pages -> list of tokens for each page."""
    doc = fitz.open(pdf_path)
    pages_tokens = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        tokens = extract_tokens_from_page(page, use_ocr=False)
        # If empty, try OCR explicitly
        if not tokens:
            tokens = extract_tokens_from_page(page, use_ocr=True)
        pages_tokens.append(tokens)
    doc.close()
    return pages_tokens

def _group_snippet(tokens: List[str], start:int, end:int) -> str:
    return " ".join(tokens[start:end])

def token_level_diff_and_annotate(old_pdf_path: str, new_pdf_path: str,
                                  out_dir: str, highlight_opacity: float = 0.5) -> Tuple[str, List[Dict[str,Any]]]:
    """
    Compare token lists and annotate NEW PDF:
    - Insert -> green highlight (new token bbox)
    - Replace -> red highlight (new token bbox)
    - Delete -> recorded in summary (no highlight)
    Returns annotated_new_path, summary_rows
    """
    # Ensure inputs exist on disk
    old_tokens_pages = extract_pdf_tokens(old_pdf_path)
    new_tokens_pages = extract_pdf_tokens(new_pdf_path)

    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)

    summary_rows = []  # {page, change_type, old_snippet, new_snippet}

    max_pages = max(len(old_tokens_pages), len(new_tokens_pages))
    for p in range(max_pages):
        old_tokens = old_tokens_pages[p] if p < len(old_tokens_pages) else []
        new_tokens = new_tokens_pages[p] if p < len(new_tokens_pages) else []

        old_texts = [t["text"] for t in old_tokens]
        new_texts = [t["text"] for t in new_tokens]

        sm = SequenceMatcher(None, old_texts, new_texts, autojunk=False)
        ops = sm.get_opcodes()

        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                continue
            if tag == "insert":
                # annotate new tokens j1..j2-1 (green)
                if p < new_doc.page_count:
                    page = new_doc.load_page(p)
                    for nj in range(j1, j2):
                        tok = new_tokens[nj]
                        rect = fitz.Rect(tok["bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=GREEN, fill=GREEN)
                        annot.set_opacity(float(highlight_opacity))
                        annot.update()
                summary_rows.append({
                    "page": p+1,
                    "change_type": "insert",
                    "old_snippet": "",
                    "new_snippet": " ".join(new_texts[j1:j2])
                })
            elif tag == "delete":
                # record deletion
                summary_rows.append({
                    "page": p+1,
                    "change_type": "delete",
                    "old_snippet": " ".join(old_texts[i1:i2]),
                    "new_snippet": ""
                })
            elif tag == "replace":
                # annotate new tokens j1..j2-1 (red)
                if p < new_doc.page_count:
                    page = new_doc.load_page(p)
                    for nj in range(j1, j2):
                        tok = new_tokens[nj]
                        rect = fitz.Rect(tok["bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=RED, fill=RED)
                        annot.set_opacity(float(highlight_opacity))
                        annot.update()
                summary_rows.append({
                    "page": p+1,
                    "change_type": "replace",
                    "old_snippet": " ".join(old_texts[i1:i2]),
                    "new_snippet": " ".join(new_texts[j1:j2])
                })
            else:
                # fallback
                summary_rows.append({
                    "page": p+1,
                    "change_type": tag,
                    "old_snippet": " ".join(old_texts[i1:i2]) if i1 is not None else "",
                    "new_snippet": " ".join(new_texts[j1:j2]) if j1 is not None else ""
                })

    # Save annotated new PDF
    annotated_new_path = os.path.join(out_dir, "annotated_new_highlights.pdf")
    new_doc.save(annotated_new_path, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()
    return annotated_new_path, summary_rows

def build_summary_pdf(summary_rows: List[Dict[str,Any]], out_path: str, created_by: str = "Ashutosh Nanaware"):
    """Create summary panel PDF and write to out_path."""
    headers = ["Page", "Type", "Old (snippet)", "New (snippet)"]
    table_rows = [headers]
    for r in summary_rows:
        ctype = r.get("change_type", "")
        tdisplay = "Other"
        if ctype == "insert":
            tdisplay = "Inserted"
        elif ctype == "replace":
            tdisplay = "Changed"
        elif ctype == "delete":
            tdisplay = "Removed"
        table_rows.append([r.get("page", ""), tdisplay, r.get("old_snippet", ""), r.get("new_snippet", "")])

    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=12, leftMargin=12, topMargin=12, bottomMargin=12)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("PDF Comparison Summary", styles["Heading1"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Created by: {created_by}", styles["Normal"]))
    elements.append(Spacer(1, 8))

    t = Table(table_rows, repeatRows=1, colWidths=[40, 80, 240, 240])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DDDDDD")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    elements.append(t)
    doc.build(elements)

def append_pdf(base_pdf: str, to_append_pdf: str, out_path: str):
    """Append pages of to_append_pdf to base_pdf and save to out_path"""
    base = fitz.open(base_pdf)
    app = fitz.open(to_append_pdf)
    base.insert_pdf(app)
    base.save(out_path, garbage=4, deflate=True)
    base.close()
    app.close()

def generate_side_by_side_pdf(old_pdf_path: str, new_pdf_path: str, output_path: str, zoom: float = 1.5, gap: int = 12):
    """Render each page from old/new and place images side-by-side into a PDF."""
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)
    out_doc = fitz.open()

    try:
        max_pages = max(old_doc.page_count, new_doc.page_count)
        for i in range(max_pages):
            if i < old_doc.page_count:
                op = old_doc.load_page(i)
                op_pix = op.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                o_img = Image.open(io.BytesIO(op_pix.tobytes("png")))
            else:
                o_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255,255,255))

            if i < new_doc.page_count:
                npg = new_doc.load_page(i)
                np_pix = npg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                n_img = Image.open(io.BytesIO(np_pix.tobytes("png")))
            else:
                n_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255,255,255))

            target_h = max(o_img.height, n_img.height)
            def scale_to_height(img, h):
                w, hh = img.size
                new_w = int(w * (h / hh))
                return img.resize((new_w, h), Image.LANCZOS)
            o_resized = scale_to_height(o_img, target_h)
            n_resized = scale_to_height(n_img, target_h)

            o_w, _ = o_resized.size
            n_w, _ = n_resized.size

            page_w = o_w + gap + n_w
            page_h = target_h
            page = out_doc.new_page(width=page_w, height=page_h)

            buf = io.BytesIO()
            o_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(0, 0, o_w, page_h), stream=buf.getvalue())

            buf = io.BytesIO()
            n_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(o_w + gap, 0, o_w + gap + n_w, page_h), stream=buf.getvalue())

        out_doc.save(output_path, garbage=4, deflate=True)
    finally:
        old_doc.close()
        new_doc.close()
        out_doc.close()

    return output_path

def process_and_generate(old_input: Any, new_input: Any, workdir: str = None, highlight_opacity: float = 0.5):
    """
    Full pipeline:
      - write inputs to disk
      - token diff + annotate new pdf
      - build summary pdf
      - append summary to annotated pdf (final)
      - generate side-by-side pdf
    Returns: annotated_with_summary_path, side_by_side_path, summary_rows
    """
    tmpdir = workdir or tempfile.mkdtemp(prefix="pdfdiff_")
    old_path = _save_input_to_temp(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_temp(new_input, os.path.join(tmpdir, "new.pdf"))

    annotated_temp, summary_rows = token_level_diff_and_annotate(old_path, new_path, tmpdir, highlight_opacity=highlight_opacity)
    summary_pdf = os.path.join(tmpdir, "summary_panel.pdf")
    build_summary_pdf(summary_rows, summary_pdf)

    final_annotated = os.path.join(tmpdir, "annotated_new_with_summary.pdf")
    append_pdf(annotated_temp, summary_pdf, final_annotated)

    side_by_side = os.path.join(tmpdir, "side_by_side.pdf")
    generate_side_by_side_pdf(old_path, new_path, side_by_side)

    return final_annotated, side_by_side, summary_rows








