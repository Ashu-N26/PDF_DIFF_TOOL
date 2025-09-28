# pdf_diff.py
"""
Advanced PDF diff utility (full-featured, character-level highlights, OCR fallback)

Exports:
 - generate_annotated_pdf(old_input, new_input, output_path, highlight_opacity=0.5, created_by="...")
 - generate_side_by_side_pdf(old_input, new_input, output_path, zoom=1.5, gap=12)
 - process_and_generate(old_input, new_input, workdir=None, highlight_opacity=0.5, created_by="...")

Inputs may be:
 - path strings ("/path/to/file.pdf")
 - file-like objects (Streamlit UploadedFile)
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import io
import tempfile
import logging
from difflib import SequenceMatcher
import math

# Lazy imports handled in _ensure_deps
_logger = logging.getLogger("pdf_diff")
logging.basicConfig(level=logging.INFO)


def _ensure_deps():
    """
    Import heavy dependencies on demand and provide them in a dict.
    Raises RuntimeError with helpful message if imports missing.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for pdf_diff: ensure PyMuPDF (fitz), Pillow, pytesseract, and reportlab are installed. "
            "If running in Docker, install system tesseract-ocr. Original error: " + repr(e)
        ) from e

    return {
        "fitz": fitz,
        "Image": Image,
        "pytesseract": pytesseract,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Table": Table,
        "TableStyle": TableStyle,
        "Paragraph": Paragraph,
        "Spacer": Spacer,
        "A4": A4,
        "colors": colors,
        "getSampleStyleSheet": getSampleStyleSheet,
    }


# -------------------------
# IO & helpers
# -------------------------
def _save_input_to_path(input_obj: Any, target_path: str) -> str:
    """
    If input_obj is a string path -> return it.
    If it's a file-like object -> write to target_path and return that path.
    """
    if isinstance(input_obj, str):
        if not os.path.exists(input_obj):
            raise FileNotFoundError(f"File not found: {input_obj}")
        return input_obj
    if hasattr(input_obj, "read"):
        data = input_obj.read()
        # Some file objects return str; convert to bytes
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(target_path, "wb") as f:
            f.write(data)
        try:
            input_obj.seek(0)
        except Exception:
            pass
        return target_path
    raise ValueError("Input must be path string or file-like object")


# -------------------------
# Text extraction with bbox (words & chars) + OCR fallback
# -------------------------
def _extract_words_from_page(page, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Extract words from a PyMuPDF page with bounding boxes.
    Returns [{'text': str, 'bbox': (x0,y0,x1,y1)} ...]
    If no selectable words or use_ocr True -> perform pytesseract OCR and map coords.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]
    pytesseract = deps["pytesseract"]

    words = page.get_text("words")  # list of tuples
    result: List[Dict[str, Any]] = []

    if words and not use_ocr:
        # words is a list; sort by y then x to get reading order
        # ensure we call sort on a list (not a string)
        try:
            words_sorted = list(words)
            words_sorted.sort(key=lambda w: (round(w[1], 1), w[0]))
        except Exception:
            words_sorted = words
        for w in words_sorted:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], str(w[4])
            t = text.strip()
            if t:
                result.append({"text": t, "bbox": (float(x0), float(y0), float(x1), float(y1))})
        if result:
            return result

    # OCR fallback:
    mat = deps["fitz"].Matrix(2, 2)  # higher resolution for better OCR
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    odata = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(odata["text"])
    W_img, H_img = img.size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / W_img, ph / H_img
    for i in range(n):
        txt = str(odata["text"][i]).strip()
        if not txt:
            continue
        left = odata["left"][i]
        top = odata["top"][i]
        width = odata["width"][i]
        height = odata["height"][i]
        x0 = left * sx
        y0 = top * sy
        x1 = (left + width) * sx
        y1 = (top + height) * sy
        # clamp
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(pw, x1), min(ph, y1)
        result.append({"text": txt, "bbox": (x0, y0, x1, y1)})
    return result


def _extract_chars_from_page(page, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Extract per-character bounding boxes. Uses page.get_text('chars') or OCR-based splitting if needed.
    Returns [{'c': str, 'bbox': (x0,y0,x1,y1)} ...]
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]
    pytesseract = deps["pytesseract"]

    chars = page.get_text("chars")  # list of tuples
    out: List[Dict[str, Any]] = []
    if chars and not use_ocr:
        # sort for reading order
        chars_sorted = list(chars)
        try:
            chars_sorted.sort(key=lambda c: (round(c[1], 1), c[0], c[6], c[7]))
        except Exception:
            pass
        for c in chars_sorted:
            x0, y0, x1, y1, ch = c[0], c[1], c[2], c[3], str(c[4])
            out.append({"c": ch, "bbox": (float(x0), float(y0), float(x1), float(y1))})
        if out:
            return out

    # OCR fallback: get word boxes and split into chars evenly
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    odata = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    W_img, H_img = img.size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / W_img, ph / H_img
    n = len(odata["text"])
    for i in range(n):
        word = str(odata["text"][i]).strip()
        if not word:
            continue
        left = odata["left"][i]
        top = odata["top"][i]
        width = odata["width"][i]
        height = odata["height"][i]
        if width <= 0 or len(word) == 0:
            continue
        char_w = width / len(word)
        for k, ch in enumerate(word):
            x0 = (left + k * char_w) * sx
            y0 = top * sy
            x1 = (left + (k + 1) * char_w) * sx
            y1 = (top + height) * sy
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(pw, x1), min(ph, y1)
            out.append({"c": ch, "bbox": (x0, y0, x1, y1)})
    return out


def extract_pdf_structure(pdf_path: str) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Extract words and chars for each page in the PDF.
    Returns: (words_pages, chars_pages)
      - words_pages[page] = [{'text','bbox'}, ...]
      - chars_pages[page] = [{'c','bbox'}, ...]
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    doc = fitz.open(pdf_path)
    words_pages: List[List[Dict[str, Any]]] = []
    chars_pages: List[List[Dict[str, Any]]] = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        words = _extract_words_from_page(page, use_ocr=False)
        if not words:
            words = _extract_words_from_page(page, use_ocr=True)
        chars = _extract_chars_from_page(page, use_ocr=False)
        if not chars:
            chars = _extract_chars_from_page(page, use_ocr=True)
        words_pages.append(words)
        chars_pages.append(chars)
    doc.close()
    return words_pages, chars_pages


# -------------------------
# Mapping helpers
# -------------------------
def _union_rects(rects: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


def _assign_chars_to_words(chars: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Assign each char to a word by center-in-rect heuristic, then nearest fallback.
    Returns list of lists: indices of chars that belong to each word.
    """
    char_centers = []
    for ci, ch in enumerate(chars):
        x0, y0, x1, y1 = ch["bbox"]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        char_centers.append((cx, cy))

    word_bboxes = []
    for w in words:
        x0, y0, x1, y1 = w["bbox"]
        word_bboxes.append((x0, y0, x1, y1))

    word_to_chars: List[List[int]] = [[] for _ in words]
    assigned = [False] * len(chars)

    # First pass: center-in-bbox
    for ci, (cx, cy) in enumerate(char_centers):
        for wi, (x0, y0, x1, y1) in enumerate(word_bboxes):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                word_to_chars[wi].append(ci)
                assigned[ci] = True
                break

    # Second pass: nearest-word fallback
    for ci, flag in enumerate(assigned):
        if flag:
            continue
        cx, cy = char_centers[ci]
        best_w = None
        best_dist = None
        for wi, (x0, y0, x1, y1) in enumerate(word_bboxes):
            wx = (x0 + x1) / 2.0
            wy = (y0 + y1) / 2.0
            d = (cx - wx) * (cx - wx) + (cy - wy) * (cy - wy)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_w = wi
        if best_w is not None:
            word_to_chars[best_w].append(ci)
            assigned[ci] = True

    # Ensure lists are sorted
    for lst in word_to_chars:
        lst.sort()
    return word_to_chars


# -------------------------
# Core diff + annotate logic
# -------------------------
def generate_annotated_pdf(old_input: Any, new_input: Any, output_path: str,
                           highlight_opacity: float = 0.5, created_by: str = "Ashutosh Nanaware") -> Tuple[str, List[Dict[str, Any]]]:
    """
    Produce annotated NEW PDF with:
      - Green (50% opacity) for inserted tokens
      - Red   (50% opacity) for replaced tokens (character-level tight boxes)
    Removed tokens are recorded in summary_rows (not highlighted).
    Returns annotated_output_path and summary_rows.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    SimpleDocTemplate = deps["SimpleDocTemplate"]
    Table = deps["Table"]
    TableStyle = deps["TableStyle"]
    Paragraph = deps["Paragraph"]
    Spacer = deps["Spacer"]
    A4 = deps["A4"]
    colors = deps["colors"]
    getSampleStyleSheet = deps["getSampleStyleSheet"]

    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new.pdf"))

    _logger.info("Extracting PDF structures (words/chars)...")
    old_words_pages, old_chars_pages = extract_pdf_structure(old_path)
    new_words_pages, new_chars_pages = extract_pdf_structure(new_path)

    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)

    summary_rows: List[Dict[str, Any]] = []

    max_pages = max(len(old_words_pages), len(new_words_pages))

    for p in range(max_pages):
        _logger.debug(f"Processing page {p+1}/{max_pages}")
        old_words = old_words_pages[p] if p < len(old_words_pages) else []
        new_words = new_words_pages[p] if p < len(new_words_pages) else []
        old_chars = old_chars_pages[p] if p < len(old_chars_pages) else []
        new_chars = new_chars_pages[p] if p < len(new_chars_pages) else []

        old_texts = [w["text"] for w in old_words]
        new_texts = [w["text"] for w in new_words]

        sm = SequenceMatcher(None, old_texts, new_texts, autojunk=False)
        opcodes = sm.get_opcodes()

        # Prepare mappings
        new_word_to_chars = _assign_chars_to_words(new_chars, new_words) if new_chars and new_words else [[] for _ in new_words]
        old_word_to_chars = _assign_chars_to_words(old_chars, old_words) if old_chars and old_words else [[] for _ in old_words]

        # Load page in new_doc for annotations if exists
        page = new_doc.load_page(p) if p < new_doc.page_count else None

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue

            if tag == "insert":
                # highlight inserted tokens (green)
                new_snippet = " ".join(new_texts[j1:j2])
                summary_rows.append({"page": p + 1, "change_type": "insert", "old_snippet": "", "new_snippet": new_snippet})
                if page is None:
                    continue
                rects = []
                for widx in range(j1, j2):
                    if widx < len(new_word_to_chars):
                        char_idxs = new_word_to_chars[widx]
                        char_rects = [new_chars[ci]["bbox"] for ci in char_idxs if 0 <= ci < len(new_chars)]
                        if char_rects:
                            r = _union_rects(char_rects)
                            if r:
                                rects.append(r)
                        else:
                            # fallback to word bbox
                            if widx < len(new_words):
                                rects.append(new_words[widx]["bbox"])
                    else:
                        if widx < len(new_words):
                            rects.append(new_words[widx]["bbox"])
                # annotate rects with green
                for r in rects:
                    if not r:
                        continue
                    rr = fitz.Rect(r)
                    annot = page.add_rect_annot(rr)
                    annot.set_colors(stroke=(60/255.0, 170/255.0, 60/255.0), fill=(60/255.0, 170/255.0, 60/255.0))
                    annot.set_opacity(float(highlight_opacity))
                    annot.update()

            elif tag == "delete":
                # record deletion in summary, no highlight
                old_snippet = " ".join(old_texts[i1:i2])
                summary_rows.append({"page": p + 1, "change_type": "delete", "old_snippet": old_snippet, "new_snippet": ""})

            elif tag == "replace":
                old_segment = " ".join(old_texts[i1:i2])
                new_segment = " ".join(new_texts[j1:j2])
                summary_rows.append({"page": p + 1, "change_type": "replace", "old_snippet": old_segment, "new_snippet": new_segment})

                # Attempt character-level diff for tight boxes
                # Determine char index ranges for new and old segments
                new_char_indices = []
                for widx in range(j1, j2):
                    if widx < len(new_word_to_chars):
                        new_char_indices.extend(new_word_to_chars[widx])
                old_char_indices = []
                for widx in range(i1, i2):
                    if widx < len(old_word_to_chars):
                        old_char_indices.extend(old_word_to_chars[widx])

                if new_char_indices and old_char_indices and new_chars and old_chars:
                    min_new, max_new = min(new_char_indices), max(new_char_indices) + 1
                    min_old, max_old = min(old_char_indices), max(old_char_indices) + 1

                    old_sub = "".join([old_chars[k]["c"] for k in range(min_old, max_old)])
                    new_sub = "".join([new_chars[k]["c"] for k in range(min_new, max_new)])

                    csm = SequenceMatcher(None, old_sub, new_sub, autojunk=False)
                    if page is not None:
                        base_new = min_new
                        for ctag, ci1, ci2, cj1, cj2 in csm.get_opcodes():
                            # if characters were inserted/replaced in new_sub -> highlight them (red)
                            if ctag in ("replace", "insert"):
                                a = base_new + cj1
                                b = base_new + cj2  # exclusive
                                rects = []
                                for ci in range(a, b):
                                    if 0 <= ci < len(new_chars):
                                        rects.append(new_chars[ci]["bbox"])
                                union = _union_rects(rects)
                                if union:
                                    rr = fitz.Rect(union)
                                    annot = page.add_rect_annot(rr)
                                    annot.set_colors(stroke=(1.0, 80/255.0, 80/255.0), fill=(1.0, 80/255.0, 80/255.0))
                                    annot.set_opacity(float(highlight_opacity))
                                    annot.update()
                else:
                    # coarse fallback: highlight whole word bboxes on new side
                    if page is not None:
                        rects = []
                        for widx in range(j1, j2):
                            if widx < len(new_words):
                                rects.append(new_words[widx]["bbox"])
                        for r in rects:
                            if not r:
                                continue
                            rr = fitz.Rect(r)
                            annot = page.add_rect_annot(rr)
                            annot.set_colors(stroke=(1.0, 80/255.0, 80/255.0), fill=(1.0, 80/255.0, 80/255.0))
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
            else:
                _logger.debug(f"Unhandled opcode tag: {tag} on page {p+1}")

    # Save annotated NEW doc temporarily
    annotated_tmp = os.path.join(tmpdir, "annotated_new.pdf")
    new_doc.save(annotated_tmp, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()

    # Produce summary panel PDF
    summary_tmp = os.path.join(tmpdir, "summary_panel.pdf")
    _build_summary_pdf(summary_rows, summary_tmp, created_by=created_by)

    # Append summary to annotated_tmp
    annotated_with_summary = os.path.join(tmpdir, "annotated_with_summary.pdf")
    _append_pdf_files(annotated_tmp, summary_tmp, annotated_with_summary)

    # Move annotated_with_summary to desired output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(annotated_with_summary, "rb") as fin, open(output_path, "wb") as fout:
        fout.write(fin.read())

    return output_path, summary_rows


# -------------------------
# Summary PDF builder & append
# -------------------------
def _build_summary_pdf(summary_rows: List[Dict[str, Any]], out_path: str, created_by: str = "Ashutosh Nanaware"):
    deps = _ensure_deps()
    SimpleDocTemplate = deps["SimpleDocTemplate"]
    Table = deps["Table"]
    TableStyle = deps["TableStyle"]
    Paragraph = deps["Paragraph"]
    Spacer = deps["Spacer"]
    A4 = deps["A4"]
    colors = deps["colors"]
    getSampleStyleSheet = deps["getSampleStyleSheet"]

    headers = ["Page", "Type", "Old (snippet)", "New (snippet)"]
    data = [headers]
    for r in summary_rows:
        ctype = r.get("change_type", "")
        if ctype == "insert":
            typ = "Inserted"
        elif ctype == "replace":
            typ = "Changed"
        elif ctype == "delete":
            typ = "Removed"
        else:
            typ = ctype
        data.append([r.get("page", ""), typ, r.get("old_snippet", ""), r.get("new_snippet", "")])

    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=12, rightMargin=12, topMargin=12, bottomMargin=12)
    elems = []
    styles = getSampleStyleSheet()
    elems.append(Paragraph("PDF Comparison Summary", styles["Heading1"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(f"Created by: {created_by}", styles["Normal"]))
    elems.append(Spacer(1, 8))

    table = Table(data, repeatRows=1, colWidths=[40, 80, 220, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elems.append(table)
    doc.build(elems)


def _append_pdf_files(base_pdf_path: str, append_pdf_path: str, out_path: str):
    deps = _ensure_deps()
    fitz = deps["fitz"]
    base = fitz.open(base_pdf_path)
    app = fitz.open(append_pdf_path)
    base.insert_pdf(app)
    base.save(out_path, garbage=4, deflate=True)
    base.close()
    app.close()


# -------------------------
# Side-by-side generation (image-based)
# -------------------------
def generate_side_by_side_pdf(old_input: Any, new_input: Any, output_path: str,
                              zoom: float = 1.5, gap: int = 12) -> str:
    """
    Render each page to images and place old (left) and new (right) images side-by-side.
    Returns output_path.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]

    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_ss_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new.pdf"))

    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)
    out_doc = fitz.open()

    try:
        max_pages = max(old_doc.page_count, new_doc.page_count)
        for i in range(max_pages):
            if i < old_doc.page_count:
                op = old_doc.load_page(i)
                op_pix = op.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                o_img = Image.open(io.BytesIO(op_pix.tobytes("png")))
            else:
                o_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255, 255, 255))

            if i < new_doc.page_count:
                npg = new_doc.load_page(i)
                np_pix = npg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                n_img = Image.open(io.BytesIO(np_pix.tobytes("png")))
            else:
                n_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255, 255, 255))

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

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out_doc.save(output_path, garbage=4, deflate=True)
    finally:
        old_doc.close()
        new_doc.close()
        out_doc.close()

    return output_path


# -------------------------
# Full pipeline wrapper
# -------------------------
def process_and_generate(old_input: Any, new_input: Any, workdir: Optional[str] = None,
                         highlight_opacity: float = 0.5, created_by: str = "Ashutosh Nanaware") -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Full pipeline:
      - writes inputs to tmp
      - annotates NEW pdf with highlights
      - builds and appends summary panel
      - creates side-by-side visual PDF
    Returns (annotated_with_summary_path, side_by_side_path, summary_rows)
    """
    tmpdir = workdir or tempfile.mkdtemp(prefix="pdfdiff_full_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new.pdf"))

    annotated_out = os.path.join(tmpdir, "annotated_with_summary.pdf")
    # annotate and produce summary
    annotated_path, summary_rows = generate_annotated_pdf(old_path, new_path, annotated_out, highlight_opacity=highlight_opacity, created_by=created_by)

    side_by_side_out = os.path.join(tmpdir, "side_by_side.pdf")
    generate_side_by_side_pdf(old_path, new_path, side_by_side_out)

    return annotated_path, side_by_side_out, summary_rows


# -------------------------
# Backwards compatibility wrappers (older app versions may call these)
# -------------------------
def generate_annotated_pdf_wrapper(old_input: Any, new_input: Any, output_path: str):
    """
    Backwards wrapper that returns same signature as older calls (write file).
    """
    annotated, summary = generate_annotated_pdf(old_input, new_input, output_path, highlight_opacity=0.5, created_by="Ashutosh Nanaware")
    return annotated


# For compatibility: export consistent names expected by various app versions
# Primary public API:
__all__ = [
    "generate_annotated_pdf",
    "generate_side_by_side_pdf",
    "process_and_generate",
    "generate_annotated_pdf_wrapper",
]











