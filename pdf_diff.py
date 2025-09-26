# pdf_diff.py
"""
Advanced PDF diff utilities with:
 - character-level highlights (tight boxes for numeric/char changes)
 - OCR fallback using pytesseract when page has no selectable text
 - side-by-side visual output (image-based)
 - annotated NEW PDF with appended summary panel (Removed/Inserted/Changed)
Functions accept either file paths (str) or file-like objects (Streamlit UploadedFile).
"""

from typing import List, Dict, Any, Tuple
import os
import io
import tempfile
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_diff")

# Lazy dependency imports are done in _ensure_deps
def _ensure_deps():
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        hint = (
            "Missing required PDF/image/ocr dependencies. "
            "Ensure your environment has PyMuPDF (fitz), Pillow, pytesseract and reportlab installed. "
            "If deploying with Docker, install system packages (tesseract-ocr, poppler, etc.).\n"
            f"Original import error: {e}"
        )
        raise RuntimeError(hint) from e

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


# ------------------------------
# Helpers: save inputs, temp files
# ------------------------------
def _save_input_to_path(input_obj: Any, target_path: str) -> str:
    """
    Accepts path str or file-like object. Writes to target_path if needed and returns a path.
    """
    if isinstance(input_obj, str):
        if not os.path.exists(input_obj):
            raise FileNotFoundError(f"Path not found: {input_obj}")
        return input_obj
    if hasattr(input_obj, "read"):
        data = input_obj.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(target_path, "wb") as f:
            f.write(data)
        try:
            input_obj.seek(0)
        except Exception:
            pass
        return target_path
    raise ValueError("Unsupported input type: provide a file path or a file-like object with .read()")


# ------------------------------
# Text extraction (words & chars) with OCR fallback
# ------------------------------
def _extract_words_from_page(page, use_ocr=False) -> List[Dict[str, Any]]:
    """
    Returns list of words with bbox: [{"text":str, "bbox":(x0,y0,x1,y1)}]
    Tries page.get_text('words') first unless use_ocr True. On OCR fallback uses pytesseract.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]
    pytesseract = deps["pytesseract"]

    words = page.get_text("words")  # list of tuples
    result = []
    if words and not use_ocr:
        # sort by y then x
        words.sort(key=lambda w: (round(w[1], 1), w[0]))
        for w in words:
            x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], str(w[4])
            t = txt.strip()
            if not t:
                continue
            result.append({"text": t, "bbox": (x0, y0, x1, y1)})
        if result:
            return result

    # OCR fallback
    mat = deps["fitz"].Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(data["text"])
    W_img, H_img = img.size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / W_img, ph / H_img
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        left = data["left"][i]
        top = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        x0 = left * sx
        y0 = top * sy
        x1 = (left + w) * sx
        y1 = (top + h) * sy
        # clamp
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(pw, x1), min(ph, y1)
        result.append({"text": txt, "bbox": (x0, y0, x1, y1)})
    return result


def _extract_chars_from_page(page, use_ocr=False) -> List[Dict[str, Any]]:
    """
    Returns per-character info: [{"c": char, "bbox": (x0,y0,x1,y1)}]
    Uses page.get_text('chars') first, otherwise OCR with character splitting.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]
    pytesseract = deps["pytesseract"]

    chars = page.get_text("chars")
    out = []
    if chars and not use_ocr:
        # sort
        chars.sort(key=lambda c: (round(c[1], 1), c[0], c[6], c[7]))
        for c in chars:
            x0, y0, x1, y1, ch = c[0], c[1], c[2], c[3], str(c[4])
            out.append({"c": ch, "bbox": (x0, y0, x1, y1)})
        if out:
            return out

    # OCR fallback: split words into characters with approximate positions
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    W_img, H_img = img.size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / W_img, ph / H_img
    for i, word in enumerate(data["text"]):
        txt = str(word).strip()
        if not txt:
            continue
        left = data["left"][i]
        top = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        if w <= 0 or len(txt) == 0:
            continue
        char_w = w / len(txt)
        for k, ch in enumerate(txt):
            cx0 = (left + k * char_w) * sx
            cy0 = top * sy
            cx1 = (left + (k + 1) * char_w) * sx
            cy1 = (top + h) * sy
            out.append({"c": ch, "bbox": (cx0, cy0, cx1, cy1)})
    return out


# ------------------------------
# Mapping helper: assign chars to words via bbox centers (robust)
# ------------------------------
def _assign_chars_to_words(chars: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Return for each word index a list of char indices (in chars) that belong to it.
    Strategy:
     - for each char compute center; if center inside a word bbox assign it
     - for unassigned chars assign to nearest word by center distance
    """
    char_centers = []
    for ci, ch in enumerate(chars):
        x0, y0, x1, y1 = ch["bbox"]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        char_centers.append((cx, cy))

    word_bboxes = []
    for wi, w in enumerate(words):
        x0, y0, x1, y1 = w["bbox"]
        word_bboxes.append((x0, y0, x1, y1))

    word_char_indices = [[] for _ in words]
    assigned = [False] * len(chars)

    # First pass: in-bbox assignment
    for ci, (cx, cy) in enumerate(char_centers):
        for wi, (x0, y0, x1, y1) in enumerate(word_bboxes):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                word_char_indices[wi].append(ci)
                assigned[ci] = True
                break

    # Second pass: nearest assignment for unassigned chars
    for ci, flag in enumerate(assigned):
        if flag:
            continue
        cx, cy = char_centers[ci]
        best_wi = None
        best_dist = None
        for wi, (x0, y0, x1, y1) in enumerate(word_bboxes):
            # compute center of word bbox
            wx = (x0 + x1) / 2.0
            wy = (y0 + y1) / 2.0
            d = (cx - wx) ** 2 + (cy - wy) ** 2
            if best_dist is None or d < best_dist:
                best_dist = d
                best_wi = wi
        if best_wi is not None:
            word_char_indices[best_wi].append(ci)
            assigned[ci] = True

    return word_char_indices


def _union_rects(rects: List[Tuple[float,float,float,float]]):
    """Return union bounding rect or None"""
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


# ------------------------------
# Core: generate_annotated_pdf
# ------------------------------
def generate_annotated_pdf(old_input: Any, new_input: Any, output_path: str,
                           highlight_opacity: float = 0.5, created_by: str = "Ashutosh Nanaware") -> str:
    """
    Create annotated NEW PDF highlighting inserts (green) and changes (red).
    Append a summary page at the end and save to output_path.
    Accepts path or file-like objects.
    Returns output_path on success.
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

    # Open docs
    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)

    summary_rows: List[Dict[str, Any]] = []

    # Extract structures for all pages first (avoid reopening inside loops)
    old_words_pages = []
    old_chars_pages = []
    for p in range(old_doc.page_count):
        page = old_doc.load_page(p)
        words = _extract_words_from_page(page, use_ocr=False)
        chars = _extract_chars_from_page(page, use_ocr=False)
        if not chars:
            chars = _extract_chars_from_page(page, use_ocr=True)
        old_words_pages.append(words)
        old_chars_pages.append(chars)

    new_words_pages = []
    new_chars_pages = []
    for p in range(new_doc.page_count):
        page = new_doc.load_page(p)
        words = _extract_words_from_page(page, use_ocr=False)
        chars = _extract_chars_from_page(page, use_ocr=False)
        if not chars:
            chars = _extract_chars_from_page(page, use_ocr=True)
        new_words_pages.append(words)
        new_chars_pages.append(chars)

    max_pages = max(len(old_words_pages), len(new_words_pages))

    # For each page, compute diffs
    for p in range(max_pages):
        logger.debug(f"Processing page {p+1}/{max_pages}")
        old_words = old_words_pages[p] if p < len(old_words_pages) else []
        old_chars = old_chars_pages[p] if p < len(old_chars_pages) else []
        new_words = new_words_pages[p] if p < len(new_words_pages) else []
        new_chars = new_chars_pages[p] if p < len(new_chars_pages) else []

        old_word_texts = [w["text"] for w in old_words]
        new_word_texts = [w["text"] for w in new_words]

        sm = SequenceMatcher(None, old_word_texts, new_word_texts, autojunk=False)
        opcodes = sm.get_opcodes()

        # Build mapping word->chars for new & old pages
        new_word_to_chars = _assign_chars_to_words(new_chars, new_words) if new_chars and new_words else [[] for _ in new_words]
        old_word_to_chars = _assign_chars_to_words(old_chars, old_words) if old_chars and old_words else [[] for _ in old_words]

        # Annotate new_doc directly
        if p < new_doc.page_count:
            page = new_doc.load_page(p)
        else:
            page = None

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            if tag == "insert":
                # highlight inserted words in new
                inserted_text = " ".join(new_word_texts[j1:j2])
                summary_rows.append({"page": p+1, "change_type": "insert", "old_snippet": "", "new_snippet": inserted_text})
                if page is None:
                    continue
                rects = []
                for widx in range(j1, j2):
                    char_idxs = new_word_to_chars[widx] if widx < len(new_word_to_chars) else []
                    if char_idxs:
                        # union rect for char boxes
                        rects_per_word = []
                        for ci in char_idxs:
                            if 0 <= ci < len(new_chars):
                                rects_per_word.append(new_chars[ci]["bbox"])
                        u = _union_rects(rects_per_word)
                        if u:
                            rects.append(u)
                    else:
                        # fallback to word bbox
                        if widx < len(new_words):
                            rects.append(new_words[widx]["bbox"])
                # Annotate union of rects individually
                for r in rects:
                    rr = fitz.Rect(r)
                    annot = page.add_rect_annot(rr)
                    annot.set_colors(stroke=(60/255.0,170/255.0,60/255.0), fill=(60/255.0,170/255.0,60/255.0))
                    annot.set_opacity(float(highlight_opacity))
                    annot.update()
            elif tag == "delete":
                deleted_text = " ".join(old_word_texts[i1:i2])
                summary_rows.append({"page": p+1, "change_type": "delete", "old_snippet": deleted_text, "new_snippet": ""})
                # do not highlight deletes in body
            elif tag == "replace":
                old_segment = " ".join(old_word_texts[i1:i2])
                new_segment = " ".join(new_word_texts[j1:j2])
                summary_rows.append({"page": p+1, "change_type": "replace", "old_snippet": old_segment, "new_snippet": new_segment})
                # Prefer character-level tight highlights inside new_chars
                # Determine union range of char indices for the new word range
                # flatten new indices
                new_char_indices = []
                for widx in range(j1, j2):
                    if widx < len(new_word_to_chars):
                        new_char_indices.extend(new_word_to_chars[widx])
                old_char_indices = []
                for widx in range(i1, i2):
                    if widx < len(old_word_to_chars):
                        old_char_indices.extend(old_word_to_chars[widx])

                # If we have char indexes on both sides, do char-level diff
                if new_char_indices and old_char_indices and new_chars and old_chars:
                    # produce strings
                    min_new, max_new = min(new_char_indices), max(new_char_indices) + 1
                    min_old, max_old = min(old_char_indices), max(old_char_indices) + 1
                    new_sub = "".join([new_chars[k]["c"] for k in range(min_new, max_new)])
                    old_sub = "".join([old_chars[k]["c"] for k in range(min_old, max_old)])
                    csm = SequenceMatcher(None, old_sub, new_sub, autojunk=False)
                    if page is not None:
                        base_new = min_new
                        for ctag, ci1, ci2, cj1, cj2 in csm.get_opcodes():
                            # cj1:cj2 are indices in new_sub (relative)
                            if ctag in ("replace", "insert"):
                                # map to absolute char indices
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
                                    annot.set_colors(stroke=(1.0,80/255.0,80/255.0), fill=(1.0,80/255.0,80/255.0))
                                    annot.set_opacity(float(highlight_opacity))
                                    annot.update()
                else:
                    # fallback: highlight whole new-word bounding boxes
                    if page is not None:
                        rects = []
                        for widx in range(j1, j2):
                            if widx < len(new_words):
                                rects.append(new_words[widx]["bbox"])
                        for r in rects:
                            rr = fitz.Rect(r)
                            annot = page.add_rect_annot(rr)
                            annot.set_colors(stroke=(1.0,80/255.0,80/255.0), fill=(1.0,80/255.0,80/255.0))
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
            else:
                # other tags unhandled but logged
                logger.debug(f"Unhandled opcode tag {tag} on page {p+1}")

    # Save annotated new PDF (temporary)
    annotated_tmp = os.path.join(tmpdir, "annotated_new.pdf")
    new_doc.save(annotated_tmp, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()

    # Build summary PDF
    summary_tmp = os.path.join(tmpdir, "summary_panel.pdf")
    _build_summary_pdf_internal(summary_rows, summary_tmp, created_by=created_by)

    # Append summary to annotated PDF
    annotated_with_summary = os.path.join(tmpdir, "annotated_with_summary.pdf")
    _append_pdf_files(annotated_tmp, summary_tmp, annotated_with_summary)

    # Copy final to requested output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Overwrite if exists
    with open(annotated_with_summary, "rb") as fin, open(output_path, "wb") as fout:
        fout.write(fin.read())

    return output_path


# ------------------------------
# Summary builder & append
# ------------------------------
def _build_summary_pdf_internal(summary_rows: List[Dict[str, Any]], out_path: str, created_by: str = "Ashutosh Nanaware"):
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
    table_data = [headers]
    for r in summary_rows:
        ctype = r.get("change_type", "")
        if ctype == "insert":
            tdisplay = "Inserted"
        elif ctype == "replace":
            tdisplay = "Changed"
        elif ctype == "delete":
            tdisplay = "Removed"
        else:
            tdisplay = ctype
        table_data.append([r.get("page", ""), tdisplay, r.get("old_snippet", ""), r.get("new_snippet", "")])

    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=12, rightMargin=12, topMargin=12, bottomMargin=12)
    elems = []
    styles = getSampleStyleSheet()
    elems.append(Paragraph("PDF Comparison - Summary", styles["Heading1"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(f"Created by: {created_by}", styles["Normal"]))
    elems.append(Spacer(1, 8))

    t = Table(table_data, repeatRows=1, colWidths=[40, 80, 220, 220])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elems.append(t)
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


# ------------------------------
# Side-by-side generator
# ------------------------------
def generate_side_by_side_pdf(old_input: Any, new_input: Any, output_path: str,
                              zoom: float = 1.5, gap: int = 12) -> str:
    """
    Render each page of old/new to images and place side-by-side into a PDF.
    Accepts paths or file-like objects.
    Returns output_path.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]

    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new.pdf"))

    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)
    out_doc = fitz.open()

    try:
        max_pages = max(old_doc.page_count, new_doc.page_count)
        for i in range(max_pages):
            # render old
            if i < old_doc.page_count:
                op = old_doc.load_page(i)
                op_pix = op.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                o_img = Image.open(io.BytesIO(op_pix.tobytes("png")))
            else:
                o_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255, 255, 255))

            # render new
            if i < new_doc.page_count:
                npg = new_doc.load_page(i)
                np_pix = npg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                n_img = Image.open(io.BytesIO(np_pix.tobytes("png")))
            else:
                n_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255, 255, 255))

            # scale to common height
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

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out_doc.save(output_path, garbage=4, deflate=True)
    finally:
        old_doc.close()
        new_doc.close()
        out_doc.close()

    return output_path


# ------------------------------
# Full pipeline wrapper
# ------------------------------
def process_and_generate(old_input: Any, new_input: Any, workdir: str = None,
                         highlight_opacity: float = 0.5, created_by: str = "Ashutosh Nanaware"):
    """
    Full pipeline:
      - writes inputs to temp
      - annotate new pdf (character tight)
      - build and append summary panel
      - generate side-by-side visual pdf
    Returns: (annotated_with_summary_path, side_by_side_path, summary_rows)
    """
    tmpdir = workdir or tempfile.mkdtemp(prefix="pdfdiff_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new.pdf"))

    annotated_out = os.path.join(tmpdir, "annotated_with_summary.pdf")
    # annotate and save to annotated_out
    generate_annotated_pdf(old_path, new_path, annotated_out, highlight_opacity=highlight_opacity, created_by=created_by)

    side_by_side_out = os.path.join(tmpdir, "side_by_side.pdf")
    generate_side_by_side_pdf(old_path, new_path, side_by_side_out)

    # Build summary_rows quick read by opening summary page from annotated_out? We can return a small summary by reconstructing
    # For simplicity, produce summary_rows from the annotated_out's appended summary by re-using parts above. But earlier
    # generate_annotated_pdf produced summary page saved inside file. For now return empty list (UI will read summary page from PDF).
    # If needed, you can modify generate_annotated_pdf to optionally return summary_rows.
    # To keep a clear API, we will re-run token diff cheaply and return summary rows:
    # Re-run internal extraction for summary
    try:
        # Use internal routine to re-generate summary rows cheaply by reading diffs (not saving)
        # Here we reuse the core logic from generate_annotated_pdf but avoid saving again.
        # For robustness, call a small helper that re-computes summary rows.
        summary_rows = _compute_summary_rows_only(old_path, new_path)
    except Exception:
        summary_rows = []

    return annotated_out, side_by_side_out, summary_rows


# ------------------------------
# Small helper to recompute summary rows only (used by process_and_generate)
# ------------------------------
def _compute_summary_rows_only(old_path: str, new_path: str) -> List[Dict[str, Any]]:
    """
    Recompute the summary rows (page, change_type, old_snippet, new_snippet) without writing outputs.
    """
    # This replicates parts of generate_annotated_pdf but only returns summary_rows
    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_summary_")
    # We'll call generate_annotated_pdf but tell it to save into tmp and then read summary page? That is heavy.
    # Simpler: re-run a trimmed version of diff logic to return summary rows:
    deps = _ensure_deps()
    fitz = deps["fitz"]

    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)
    old_words_pages = []
    new_words_pages = []
    for p in range(old_doc.page_count):
        page = old_doc.load_page(p)
        old_words_pages.append(_extract_words_from_page(page, use_ocr=False) or _extract_words_from_page(page, use_ocr=True))
    for p in range(new_doc.page_count):
        page = new_doc.load_page(p)
        new_words_pages.append(_extract_words_from_page(page, use_ocr=False) or _extract_words_from_page(page, use_ocr=True))
    old_doc.close()
    new_doc.close()

    summary_rows = []
    max_pages = max(len(old_words_pages), len(new_words_pages))
    for p in range(max_pages):
        old_words = old_words_pages[p] if p < len(old_words_pages) else []
        new_words = new_words_pages[p] if p < len(new_words_pages) else []
        old_texts = [w["text"] for w in old_words]
        new_texts = [w["text"] for w in new_words]
        sm = SequenceMatcher(None, old_texts, new_texts, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "insert":
                summary_rows.append({"page": p+1, "change_type":"insert", "old_snippet":"", "new_snippet":" ".join(new_texts[j1:j2])})
            elif tag == "delete":
                summary_rows.append({"page": p+1, "change_type":"delete", "old_snippet":" ".join(old_texts[i1:i2]), "new_snippet":""})
            elif tag == "replace":
                summary_rows.append({"page": p+1, "change_type":"replace", "old_snippet":" ".join(old_texts[i1:i2]), "new_snippet":" ".join(new_texts[j1:j2])})
    return summary_rows










