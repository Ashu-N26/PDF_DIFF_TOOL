# pdf_diff.py
"""
Advanced PDF diff with character-level highlights and robust page handling.

Functions:
- process_and_generate(old_input, new_input, workdir=None, highlight_opacity=0.5)
  -> returns (annotated_with_summary_path, side_by_side_path, summary_rows)

Key behavior:
- Word-level diff first to find changed/inserted/deleted regions.
- For 'replace' ops, do character-level diff using page.get_text('chars') to create tight boxes
  around only the differing characters (good for numeric changes like 24.76 -> 24.74).
- Insert -> highlight all new chars for the inserted words.
- Delete -> recorded in summary (not highlighted in body).
- Uses OCR (pytesseract) when page has no selectable text.
"""
from typing import List, Dict, Any, Tuple
import fitz
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
import logging

logging.basicConfig(level=logging.INFO)

# Colors
GREEN = (60/255.0, 170/255.0, 60/255.0)
RED = (1.0, 80/255.0, 80/255.0)


def _save_input_to_temp(input_obj: Any, out_path: str) -> str:
    """Write uploaded file-like or accept existing path."""
    if isinstance(input_obj, str):
        if not os.path.exists(input_obj):
            raise FileNotFoundError(input_obj)
        return input_obj
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


# -------------------------
# Extraction helpers
# -------------------------
def extract_page_words(page: fitz.Page) -> List[Dict[str, Any]]:
    """Return ordered list of words with bbox from page.get_text('words')."""
    words = page.get_text("words")  # x0,y0,x1,y1, word, block_no, line_no, word_no
    if not words:
        return []
    # sort by y then x to preserve reading order
    words.sort(key=lambda w: (round(w[1], 1), w[0]))
    out = []
    for w in words:
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], str(w[4])
        t = txt.strip()
        if t == "":
            continue
        out.append({"text": t, "bbox": (x0, y0, x1, y1)})
    return out


def extract_page_chars(page: fitz.Page, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Extract per-character boxes using page.get_text('chars') or OCR fallback.
    Returns list of dicts: {"c": char, "bbox": (x0,y0,x1,y1)}
    """
    chars = page.get_text("chars")  # x0, y0, x1, y1, char, block_no, line_no, char_no
    out = []
    if chars and not use_ocr:
        # sort to strict reading order
        chars.sort(key=lambda c: (round(c[1], 1), c[0], c[6], c[7]))
        for c in chars:
            x0, y0, x1, y1, ch = c[0], c[1], c[2], c[3], str(c[4])
            ch = ch or ""
            out.append({"c": ch, "bbox": (x0, y0, x1, y1)})
        if out:
            return out

    # OCR fallback
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    W, H = img.size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / W, ph / H
    for i, text in enumerate(ocr["text"]):
        txt = str(text).strip()
        if not txt:
            continue
        left = ocr["left"][i]
        top = ocr["top"][i]
        width = ocr["width"][i]
        height = ocr["height"][i]
        # approximate character-level by splitting the word into characters (even spacing)
        if width == 0 or height == 0:
            continue
        char_w = width / max(1, len(txt))
        for k, ch in enumerate(txt):
            x0 = (left + k * char_w) * sx
            y0 = top * sy
            x1 = (left + (k + 1) * char_w) * sx
            y1 = (top + height) * sy
            out.append({"c": ch, "bbox": (x0, y0, x1, y1)})
    return out


def extract_pdf_structure(pdf_path: str) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    For a given pdf path return per-page words list and chars list.
    Returns: words_pages, chars_pages
    """
    doc = fitz.open(pdf_path)
    words_pages = []
    chars_pages = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        words = extract_page_words(page)
        chars = extract_page_chars(page, use_ocr=False)
        if not chars:
            # try OCR explicitly if no chars
            chars = extract_page_chars(page, use_ocr=True)
        words_pages.append(words)
        chars_pages.append(chars)
    doc.close()
    return words_pages, chars_pages


# -------------------------
# Utility: union rects of contiguous chars
# -------------------------
def _union_rects(rects):
    """Given list of rects (x0,y0,x1,y1) return union rect."""
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


# -------------------------
# Map words to char index ranges (on a page)
# -------------------------
def map_words_to_char_ranges(words: List[Dict[str, Any]], chars: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    For a page, given words list and chars list (in reading order), return for each word the (start_idx, end_idx)
    indices into chars such that chars[start:end] correspond to that word.
    Strategy: sequentially scan chars and match words by concatenation with whitespace tolerance.
    """
    res = []
    char_texts = [c["c"] for c in chars]
    char_pos = 0
    n_chars = len(chars)
    for w in words:
        wtxt = w["text"]
        if char_pos >= n_chars:
            res.append((None, None))
            continue
        # Try to find sequence matching wtxt starting from char_pos
        joined = "".join(char_texts[char_pos:])  # rest
        # We need to match wtxt possibly with spaces removed vs OCR; perform greedy search
        # Find earliest index where substring of joined equals wtxt (allow trimming)
        found = -1
        # naive search: slide window up to length of remaining chars
        max_try = min(n_chars - char_pos, len(joined))
        # Use Python find of wtxt in joined (fast)
        idx = joined.find(wtxt)
        if idx >= 0:
            start_idx = char_pos + idx
            end_idx = start_idx + len(wtxt)
            res.append((start_idx, end_idx))
            char_pos = end_idx
        else:
            # fallback: try to match character by character allowing punctuation differences
            # move forward until some characters match
            # as fallback, assign None
            res.append((None, None))
    return res


# -------------------------
# Main diff + annotate
# -------------------------
def token_level_diff_and_annotate(old_pdf_path: str, new_pdf_path: str, out_dir: str, highlight_opacity: float = 0.5):
    """
    Compare old/new pdfs and annotate NEW PDF with green/red highlights.
    Returns: annotated_new_path, summary_rows
    """
    # Extract structure
    old_words_pages, old_chars_pages = extract_pdf_structure(old_pdf_path)
    new_words_pages, new_chars_pages = extract_pdf_structure(new_pdf_path)

    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)

    summary_rows = []

    max_pages = max(len(old_words_pages), len(new_words_pages))
    for p in range(max_pages):
        old_words = old_words_pages[p] if p < len(old_words_pages) else []
        new_words = new_words_pages[p] if p < len(new_words_pages) else []
        old_chars = old_chars_pages[p] if p < len(old_chars_pages) else []
        new_chars = new_chars_pages[p] if p < len(new_chars_pages) else []

        # Build word text lists
        old_word_texts = [w["text"] for w in old_words]
        new_word_texts = [w["text"] for w in new_words]

        sm = SequenceMatcher(None, old_word_texts, new_word_texts, autojunk=False)
        opcodes = sm.get_opcodes()

        # Map words -> char ranges for page-level char alignment
        new_word_to_char = map_words_to_char_ranges(new_words, new_chars)
        old_word_to_char = map_words_to_char_ranges(old_words, old_chars)

        # For each operation, act accordingly
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            if tag == "insert":
                # Highlight inserted words on NEW page (use char ranges when available)
                if j1 is None:
                    continue
                if p < new_doc.page_count:
                    page = new_doc.load_page(p)
                    for widx in range(j1, j2):
                        start_end = new_word_to_char[widx] if widx < len(new_word_to_char) else (None, None)
                        s, e = start_end
                        rects = []
                        if s is not None and e is not None:
                            for ci in range(s, e):
                                if ci < len(new_chars):
                                    rects.append(new_chars[ci]["bbox"])
                        else:
                            # fallback: use word bbox
                            if widx < len(new_words):
                                rects.append(new_words[widx]["bbox"])
                        union = _union_rects(rects)
                        if union:
                            r = fitz.Rect(union)
                            annot = page.add_rect_annot(r)
                            annot.set_colors(stroke=GREEN, fill=GREEN)
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
                summary_rows.append({
                    "page": p+1,
                    "change_type": "insert",
                    "old_snippet": "",
                    "new_snippet": " ".join(new_word_texts[j1:j2])
                })
            elif tag == "delete":
                # Record deletion in summary (no highlight)
                summary_rows.append({
                    "page": p+1,
                    "change_type": "delete",
                    "old_snippet": " ".join(old_word_texts[i1:i2]),
                    "new_snippet": ""
                })
            elif tag == "replace":
                # For replace, do character-level diff inside the affected region to tighten highlights
                old_segment_words = old_word_texts[i1:i2]
                new_segment_words = new_word_texts[j1:j2]
                old_segment_text = " ".join(old_segment_words)
                new_segment_text = " ".join(new_segment_words)

                # If there are char lists available, build character strings with indices
                old_char_str = "".join([c["c"] for c in old_chars]) if old_chars else ""
                new_char_str = "".join([c["c"] for c in new_chars]) if new_chars else ""

                # Determine rough char index bounds for the word ranges using mapping
                # For new side:
                new_char_s = None
                new_char_e = None
                for widx in range(j1, j2):
                    if widx < len(new_word_to_char):
                        s, e = new_word_to_char[widx]
                        if s is not None:
                            new_char_s = s if new_char_s is None else min(new_char_s, s)
                            new_char_e = e if new_char_e is None else max(new_char_e, e)
                # For old side:
                old_char_s = None
                old_char_e = None
                for widx in range(i1, i2):
                    if widx < len(old_word_to_char):
                        s, e = old_word_to_char[widx]
                        if s is not None:
                            old_char_s = s if old_char_s is None else min(old_char_s, s)
                            old_char_e = e if old_char_e is None else max(old_char_e, e)

                # If character indices available, run character-level SequenceMatcher inside those substrings
                if new_char_s is not None and old_char_s is not None:
                    old_sub = "".join([c["c"] for c in old_chars[old_char_s:old_char_e]])
                    new_sub = "".join([c["c"] for c in new_chars[new_char_s:new_char_e]])
                    char_sm = SequenceMatcher(None, old_sub, new_sub, autojunk=False)
                    # For new_sub opcodes, highlight inserted/replace chars precisely
                    if p < new_doc.page_count:
                        page = new_doc.load_page(p)
                        # base index offset in new_chars
                        base_new = new_char_s
                        for ctag, ci1, ci2, cj1, cj2 in char_sm.get_opcodes():
                            # ctag relates old_sub vs new_sub, mapping new indices cj1:cj2
                            if ctag in ("replace", "insert"):
                                # gather boxes for new char indices base_new + cj1 .. base_new + cj2-1
                                rects = []
                                for chi in range(base_new + cj1, base_new + cj2):
                                    if 0 <= chi < len(new_chars):
                                        rects.append(new_chars[chi]["bbox"])
                                union = _union_rects(rects)
                                if union:
                                    r = fitz.Rect(union)
                                    annot = page.add_rect_annot(r)
                                    annot.set_colors(stroke=RED, fill=RED)
                                    annot.set_opacity(float(highlight_opacity))
                                    annot.update()
                    # record summary row
                    summary_rows.append({
                        "page": p+1,
                        "change_type": "replace",
                        "old_snippet": old_segment_text,
                        "new_snippet": new_segment_text
                    })
                else:
                    # Fallback: highlight whole new words area (coarse)
                    if p < new_doc.page_count:
                        page = new_doc.load_page(p)
                        rects = []
                        for widx in range(j1, j2):
                            if widx < len(new_words):
                                rects.append(new_words[widx]["bbox"])
                        union = _union_rects(rects)
                        if union:
                            r = fitz.Rect(union)
                            annot = page.add_rect_annot(r)
                            annot.set_colors(stroke=RED, fill=RED)
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
                    summary_rows.append({
                        "page": p+1,
                        "change_type": "replace",
                        "old_snippet": old_segment_text,
                        "new_snippet": new_segment_text
                    })
            else:
                # unknown tag, record
                summary_rows.append({
                    "page": p+1,
                    "change_type": tag,
                    "old_snippet": " ".join(old_word_texts[i1:i2]) if i1 is not None else "",
                    "new_snippet": " ".join(new_word_texts[j1:j2]) if j1 is not None else ""
                })

    # Save annotated NEW PDF
    annotated_path = os.path.join(out_dir, "annotated_new_highlights.pdf")
    new_doc.save(annotated_path, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()
    return annotated_path, summary_rows


# -------------------------
# Summary PDF builder & append utilities (unchanged)
# -------------------------
def build_summary_pdf(summary_rows: List[Dict[str, Any]], out_path: str, created_by: str = "Ashutosh Nanaware"):
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
    base = fitz.open(base_pdf)
    app = fitz.open(to_append_pdf)
    base.insert_pdf(app)
    base.save(out_path, garbage=4, deflate=True)
    base.close()
    app.close()


# -------------------------
# Side-by-side image-based function (keeps layout exactly)
# -------------------------
from PIL import Image as PILImage


def generate_side_by_side_pdf(old_pdf_path: str, new_pdf_path: str, output_path: str, zoom: float = 1.5, gap: int = 12):
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)
    out_doc = fitz.open()
    try:
        max_pages = max(old_doc.page_count, new_doc.page_count)
        for i in range(max_pages):
            if i < old_doc.page_count:
                op = old_doc.load_page(i)
                op_pix = op.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                o_img = PILImage.open(io.BytesIO(op_pix.tobytes("png")))
            else:
                o_img = PILImage.new("RGB", (int(595 * zoom), int(842 * zoom)), (255,255,255))

            if i < new_doc.page_count:
                npg = new_doc.load_page(i)
                np_pix = npg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                n_img = PILImage.open(io.BytesIO(np_pix.tobytes("png")))
            else:
                n_img = PILImage.new("RGB", (int(595 * zoom), int(842 * zoom)), (255,255,255))

            target_h = max(o_img.height, n_img.height)
            def scale_to_height(img, h):
                w, hh = img.size
                new_w = int(w * (h / hh))
                return img.resize((new_w, h), PILImage.LANCZOS)
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


# -------------------------
# Full pipeline wrapper
# -------------------------
def process_and_generate(old_input: Any, new_input: Any, workdir: str = None, highlight_opacity: float = 0.5):
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









