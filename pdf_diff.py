# pdf_diff.py
"""
PDF_DIFF Full-Fat Engine (expanded)
Author: Ashutosh Nanaware (as requested)
Description:
    Advanced, production-ready PDF difference engine with:
      - Page-to-page alignment and normalization
      - Word-level and character-level (sub-token) diffs
      - OCR fallback (pytesseract) for scanned pages
      - Tight per-glyph/character bounding boxes for replaced parts
      - Green highlights (50% opacity) for Inserted text in NEW PDF
      - Red highlights (50% opacity) for Changed/Replaced tokens in NEW PDF
      - Removed items recorded in Summary Panel (appended as a PDF page)
      - Side-by-side visual PDF generation (OLD left, NEW right) without artificial highlights
      - Robust APIs that accept either file paths or file-like objects (Streamlit upload)
      - Multiple helper utilities and compatibility wrappers
Notes:
    - Requires system-level tesseract-ocr when using OCR fallback
    - Requires Python packages: PyMuPDF (fitz), Pillow, pytesseract, reportlab, pandas (optional)
Usage:
    from pdf_diff import process_and_generate
    annotated_pdf_path, side_by_side_path, summary_rows = process_and_generate(old_file, new_file)
"""

# Standard library imports
import os
import io
import sys
import math
import tempfile
import logging
from typing import Any, List, Dict, Tuple, Optional

# Defer heavy imports to _ensure_deps to make import-time errors clearer and to provide helpful messages
_logger = logging.getLogger("pdf_diff")
logging.basicConfig(level=logging.INFO)


def _ensure_deps():
    """
    Import third-party dependencies lazily and return them in a dict.
    Raises RuntimeError with a helpful hint if any are missing.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image, ImageChops
        import pytesseract
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        raise RuntimeError(
            "Missing PDF/image/ocr dependencies. Please install: PyMuPDF (fitz), Pillow, pytesseract, reportlab. "
            "If you plan to use OCR, ensure tesseract-ocr is installed on the system (apt-get install tesseract-ocr). "
            f"Original import error: {repr(e)}"
        ) from e

    return {
        "fitz": fitz,
        "Image": Image,
        "ImageChops": ImageChops,
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


# -----------------------------
# Constants & Colors
# -----------------------------
# Normalized RGB for highlights (PyMuPDF uses 0..1 floats)
_GREEN_RGB = (60 / 255.0, 170 / 255.0, 60 / 255.0)
_RED_RGB = (1.0, 80 / 255.0, 80 / 255.0)
_BLACK_RGB = (0.0, 0.0, 0.0)

DEFAULT_HIGHLIGHT_OPACITY = 0.5
DEFAULT_CREATED_BY = "Ashutosh Nanaware"


# -----------------------------
# Utilities: file I/O
# -----------------------------
def _save_input_to_path(input_obj: Any, target_path: str) -> str:
    """
    Accepts either a path (str) or a file-like object with .read().
    Writes to target_path if file-like and returns the filesystem path.
    """
    if isinstance(input_obj, str):
        if not os.path.exists(input_obj):
            raise FileNotFoundError(f"Path not found: {input_obj}")
        return input_obj

    # file-like
    if hasattr(input_obj, "read"):
        data = input_obj.read()
        # handle streamlit's UploadedFile behavior
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(target_path, "wb") as f:
            f.write(data)
        try:
            input_obj.seek(0)
        except Exception:
            pass
        return target_path

    raise ValueError("Input must be a file path (str) or a file-like object with .read().")


def _ensure_dir(path: str):
    """
    Ensure that the folder exists. If path is '', do nothing.
    """
    if not path:
        return
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Low-level geometry helpers
# -----------------------------
def _union_rects(rects: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Given a list of rectangles (x0,y0,x1,y1) compute the union bounding rectangle.
    """
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


def _expand_rect(rect: Tuple[float, float, float, float], pad: float) -> Tuple[float, float, float, float]:
    """
    Expand rectangle by pad in each direction.
    """
    if rect is None:
        return rect
    x0, y0, x1, y1 = rect
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _clip_rect(rect: Tuple[float, float, float, float], page_rect: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Clip rect to page rect (page_rect: x0,y0,x1,y1).
    """
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    px0, py0, px1, py1 = page_rect
    return (max(px0, x0), max(py0, y0), min(px1, x1), min(py1, y1))


# -----------------------------
# Rendering helpers (image-based)
# -----------------------------
def _render_page_to_pil(pdf_path: str, page_index: int, zoom: float = 2.0) -> Any:
    """
    Render a PDF page into a PIL Image using PyMuPDF.
    Returns a PIL Image.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]

    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= doc.page_count:
        doc.close()
        raise IndexError("Page index out of range.")
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def _image_diff_boxes(img1: Any, img2: Any) -> List[Tuple[int, int, int, int]]:
    """
    Return bounding boxes where two PIL images differ.
    If identical, empty list returned.
    This is used as a fallback to detect visual changes (tables, diagrams).
    """
    deps = _ensure_deps()
    ImageChops = deps["ImageChops"]
    if img1.mode != img2.mode or img1.size != img2.size:
        # resize to common size (small adjustments)
        img2 = img2.resize(img1.size)
    diff = ImageChops.difference(img1, img2)
    bbox = diff.getbbox()
    if not bbox:
        return []
    # For further granularity we could detect multiple boxes (connected components), but bounding bbox is okay.
    return [bbox]


# -----------------------------
# Text extraction helpers (words & chars)
# -----------------------------
def _extract_words_from_page_fitz(page, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Try to use PyMuPDF page.get_text('words') to return words and bounding boxes.
    Each element is {"text": "...", "bbox": (x0,y0,x1,y1)} in PDF coordinates.
    If no words are found or use_ocr is True, returns empty list (OCR fallback handled by other func).
    """
    # NOTE: PyMuPDF uses origin at top-left for page.get_text coords, consistent with page.rect
    words_raw = page.get_text("words")  # list of tuples (x0,y0,x1,y1, "word", block_no, line_no, word_no)
    words: List[Dict[str, Any]] = []
    if words_raw:
        # Ensure we have a list and sort it for reading order
        try:
            words_list = list(words_raw)
            words_list.sort(key=lambda w: (round(w[1], 1), w[0]))
        except Exception:
            words_list = list(words_raw)
        for w in words_list:
            try:
                x0, y0, x1, y1, txt = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
            except Exception:
                continue
            txt = txt.strip()
            if not txt:
                continue
            words.append({"text": txt, "bbox": (x0, y0, x1, y1)})
    return words


def _ocr_extract_words_from_page(page, dpi_render=300) -> List[Dict[str, Any]]:
    """
    Use pytesseract to OCR a page image and return words with bounding boxes mapped to page coordinates.
    This is slower but ensures no data is missed for scanned PDFs.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]
    pytesseract = deps["pytesseract"]

    # Render at reasonable DPI for good OCR
    mat = fitz.Matrix(dpi_render / 72.0, dpi_render / 72.0)  # 72 DPI base
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words: List[Dict[str, Any]] = []
    W_img, H_img = img.size
    pw, ph = page.rect.width, page.rect.height
    sx = pw / W_img
    sy = ph / H_img
    n = len(data.get("text", []))
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        left = data["left"][i]
        top = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        # Map to page coordinates
        x0 = left * sx
        y0 = top * sy
        x1 = (left + w) * sx
        y1 = (top + h) * sy
        # clamp
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(pw, x1), min(ph, y1)
        words.append({"text": txt, "bbox": (x0, y0, x1, y1)})
    return words


def _extract_chars_from_page_fitz(page, use_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Extract per-character boxes using PyMuPDF page.get_text('chars').
    Each char: {"c": char, "bbox": (x0,y0,x1,y1)}
    If no chars (scanned pages), return empty list (OCR fallback handled elsewhere).
    """
    chars_raw = page.get_text("chars")  # list of tuples
    chars: List[Dict[str, Any]] = []
    if chars_raw:
        try:
            chars_list = list(chars_raw)
            chars_list.sort(key=lambda c: (round(c[1], 1), c[0], c[6], c[7]))
        except Exception:
            chars_list = list(chars_raw)
        for c in chars_list:
            try:
                x0, y0, x1, y1, ch = float(c[0]), float(c[1]), float(c[2]), float(c[3]), str(c[4])
            except Exception:
                continue
            chars.append({"c": ch, "bbox": (x0, y0, x1, y1)})
    return chars


def _ocr_extract_chars_from_page(page, dpi_render=300) -> List[Dict[str, Any]]:
    """
    OCR fallback to extract per-character approximations by splitting OCR words into characters.
    Returns estimated character bboxes mapped to page coordinates.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]
    pytesseract = deps["pytesseract"]

    mat = fitz.Matrix(dpi_render / 72.0, dpi_render / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    W_img, H_img = img.size
    pw, ph = page.rect.width, page.rect.height
    sx = pw / W_img
    sy = ph / H_img
    chars: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        word = str(data["text"][i]).strip()
        if not word:
            continue
        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]
        if width <= 0:
            continue
        char_w = width / max(1, len(word))
        for k, ch in enumerate(word):
            x0 = (left + k * char_w) * sx
            y0 = top * sy
            x1 = (left + (k + 1) * char_w) * sx
            y1 = (top + height) * sy
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(pw, x1), min(ph, y1)
            chars.append({"c": ch, "bbox": (x0, y0, x1, y1)})
    return chars


# -----------------------------
# Page-level extraction wrapper
# -----------------------------
def _extract_page_structure(pdf_path: str) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Extract words and chars for each page of the given PDF.
    Returns tuple: (words_pages, chars_pages)
      - words_pages[page_index] = [{'text':..., 'bbox':(...)}...]
      - chars_pages[page_index] = [{'c':..., 'bbox':(...)}...]
    Uses PyMuPDF extraction first and OCR fallback if necessary.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]

    doc = fitz.open(pdf_path)
    words_pages: List[List[Dict[str, Any]]] = []
    chars_pages: List[List[Dict[str, Any]]] = []

    for p in range(doc.page_count):
        page = doc.load_page(p)
        words = _extract_words_from_page_fitz(page, use_ocr=False)
        if not words:
            words = _ocr_extract_words_from_page(page)
        chars = _extract_chars_from_page_fitz(page, use_ocr=False)
        if not chars:
            chars = _ocr_extract_chars_from_page(page)
        words_pages.append(words)
        chars_pages.append(chars)

    doc.close()
    return words_pages, chars_pages


# -----------------------------
# Mapping words -> characters
# -----------------------------
def _assign_chars_to_words(chars: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Build mapping of word index -> list of character indices that belong to that word.
    Strategy:
        - compute char centers; if center inside word bbox assign
        - fallback: nearest word center by Euclidean distance
    Returns a list where each element is a list of char indices belonging to that word.
    """
    if not words:
        return []

    # Compute char centers and word bboxes
    char_centers = []
    for ci, ch in enumerate(chars):
        x0, y0, x1, y1 = ch["bbox"]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        char_centers.append((cx, cy))

    word_rects = []
    for w in words:
        x0, y0, x1, y1 = w["bbox"]
        word_rects.append((x0, y0, x1, y1))

    word_to_chars = [[] for _ in words]
    assigned = [False] * len(chars)

    # First pass: center-in-bbox assignment
    for ci, (cx, cy) in enumerate(char_centers):
        for wi, (wx0, wy0, wx1, wy1) in enumerate(word_rects):
            if wx0 <= cx <= wx1 and wy0 <= cy <= wy1:
                word_to_chars[wi].append(ci)
                assigned[ci] = True
                break

    # Second pass: nearest fallback for unassigned chars
    for ci, flag in enumerate(assigned):
        if flag:
            continue
        cx, cy = char_centers[ci]
        best_w = None
        best_d = None
        for wi, (wx0, wy0, wx1, wy1) in enumerate(word_rects):
            wx = (wx0 + wx1) / 2.0
            wy = (wy0 + wy1) / 2.0
            d = (cx - wx) ** 2 + (cy - wy) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_w = wi
        if best_w is not None:
            word_to_chars[best_w].append(ci)
            assigned[ci] = True

    # Ensure each char list sorted
    for lst in word_to_chars:
        lst.sort()
    return word_to_chars


# -----------------------------
# Higher-level diff utilities
# -----------------------------
def _tokenize_words_for_diff(words: List[Dict[str, Any]]) -> List[str]:
    """
    Convert words list into a list of word texts for SequenceMatcher diffing.
    """
    return [w["text"] for w in words]


def _char_sequence_from_chars(chars: List[Dict[str, Any]]) -> str:
    """
    Build a string by concatenating characters from the char dict list.
    """
    return "".join([c.get("c", "") for c in chars])


# -----------------------------
# Core diffing algorithm (page-by-page)
# -----------------------------
def _diff_pages(old_words: List[Dict[str, Any]],
                old_chars: List[Dict[str, Any]],
                new_words: List[Dict[str, Any]],
                new_chars: List[Dict[str, Any]],
                page_index: int) -> Dict[str, Any]:
    """
    Diff the content of a single page.
    Returns a dictionary with:
        - page_index
        - ops: list of operations, each op is a dict containing:
            - tag: 'equal'|'insert'|'delete'|'replace'
            - old_range: (i1,i2) word indices in old
            - new_range: (j1,j2) word indices in new
            - old_snippet, new_snippet
            - rects: list of bounding rects (for new side) to highlight (tight boxes if char-level)
    This function performs:
      - word-level diff via SequenceMatcher
      - for replace ops, performs char-level diff mapping char indices to tight rects
      - for insert ops, maps inserted words to rects via char mapping
      - delete ops recorded only in summary (no rects)
    """
    from difflib import SequenceMatcher

    ops: List[Dict[str, Any]] = []
    old_texts = _tokenize_words_for_diff(old_words)
    new_texts = _tokenize_words_for_diff(new_words)

    sm = SequenceMatcher(None, old_texts, new_texts, autojunk=False)
    opcodes = sm.get_opcodes()

    # Map words -> chars for both pages
    new_word_to_chars = _assign_chars_to_words(new_chars, new_words) if new_chars and new_words else [[] for _ in new_words]
    old_word_to_chars = _assign_chars_to_words(old_chars, old_words) if old_chars and old_words else [[] for _ in old_words]

    for tag, i1, i2, j1, j2 in opcodes:
        # build human readable snippets
        old_snippet = " ".join(old_texts[i1:i2]) if i1 is not None else ""
        new_snippet = " ".join(new_texts[j1:j2]) if j1 is not None else ""

        record = {
            "tag": tag,
            "old_range": (i1, i2),
            "new_range": (j1, j2),
            "old_snippet": old_snippet,
            "new_snippet": new_snippet,
            "rects": [],  # list of rects to highlight on NEW page (for insert/replace)
            "char_level": False,
        }

        if tag == "equal":
            # nothing to highlight
            ops.append(record)
            continue

        if tag == "insert":
            # For inserted words, gather rects via char mapping if possible, else word bbox.
            rects: List[Tuple[float, float, float, float]] = []
            for widx in range(j1, j2):
                if widx < len(new_word_to_chars):
                    char_idxs = new_word_to_chars[widx]
                    char_rects = [new_chars[cidx]["bbox"] for cidx in char_idxs if 0 <= cidx < len(new_chars)]
                    if char_rects:
                        union = _union_rects(char_rects)
                        if union:
                            rects.append(union)
                    else:
                        # fallback to word bbox if char mapping empty
                        if widx < len(new_words):
                            rects.append(new_words[widx]["bbox"])
                else:
                    if widx < len(new_words):
                        rects.append(new_words[widx]["bbox"])
            record["rects"] = rects
            record["char_level"] = False
            ops.append(record)
            continue

        if tag == "delete":
            # record deletion only in summary; no rects in body
            record["rects"] = []
            ops.append(record)
            continue

        if tag == "replace":
            # Attempt to do char-level diff inside the replaced region to create tight highlights
            # Build continuous character subsequences for the old and new region
            new_char_indices = []
            for widx in range(j1, j2):
                if widx < len(new_word_to_chars):
                    new_char_indices.extend(new_word_to_chars[widx])
            old_char_indices = []
            for widx in range(i1, i2):
                if widx < len(old_word_to_chars):
                    old_char_indices.extend(old_word_to_chars[widx])

            # If both sides have character mappings, run char SequenceMatcher
            if new_char_indices and old_char_indices and new_chars and old_chars:
                # Build substrings
                min_new, max_new = min(new_char_indices), max(new_char_indices) + 1
                min_old, max_old = min(old_char_indices), max(old_char_indices) + 1
                new_sub_str = "".join([new_chars[k]["c"] for k in range(min_new, max_new)])
                old_sub_str = "".join([old_chars[k]["c"] for k in range(min_old, max_old)])
                csm = SequenceMatcher(None, old_sub_str, new_sub_str, autojunk=False)
                rects: List[Tuple[float, float, float, float]] = []
                # base offset in new_chars
                base_new = min_new
                for ctag, ci1, ci2, cj1, cj2 in csm.get_opcodes():
                    if ctag in ("replace", "insert"):
                        # map to absolute char indices
                        a = base_new + cj1
                        b = base_new + cj2  # exclusive
                        char_rects = [new_chars[idx]["bbox"] for idx in range(a, b) if 0 <= idx < len(new_chars)]
                        if char_rects:
                            union = _union_rects(char_rects)
                            if union:
                                rects.append(union)
                record["rects"] = rects
                record["char_level"] = True if rects else False
                ops.append(record)
            else:
                # fallback to whole-word bounding boxes (coarse)
                rects = []
                for widx in range(j1, j2):
                    if widx < len(new_words):
                        rects.append(new_words[widx]["bbox"])
                record["rects"] = rects
                record["char_level"] = False
                ops.append(record)

    return {
        "page_index": page_index,
        "ops": ops,
    }


# -----------------------------
# Compare full PDFs (page-by-page)
# -----------------------------
def compare_pdfs(old_input: Any, new_input: Any, use_ocr: bool = True) -> List[Dict[str, Any]]:
    """
    Compare old and new PDFs and return a structured list of page diffs.
    Each element is a dict:
      {
         "page_index": int,
         "page_size": (w,h),
         "ops": [ {tag, old_range, new_range, old_snippet, new_snippet, rects, char_level}, ... ],
         "image_diff_boxes": [ (x0,y0,x1,y1), ... ]   # optional image-based boxes
      }
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]

    # Save inputs to paths if needed
    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_inputs_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new.pdf"))

    # Extract structures
    _logger.info("Extracting PDF structures for old and new (words/chars) ...")
    old_words_pages, old_chars_pages = _extract_page_structure(old_path)
    new_words_pages, new_chars_pages = _extract_page_structure(new_path)

    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)

    results: List[Dict[str, Any]] = []
    max_pages = max(old_doc.page_count, new_doc.page_count)

    for p in range(max_pages):
        _logger.debug(f"Comparing page {p+1}/{max_pages}")
        old_words = old_words_pages[p] if p < len(old_words_pages) else []
        old_chars = old_chars_pages[p] if p < len(old_chars_pages) else []
        new_words = new_words_pages[p] if p < len(new_words_pages) else []
        new_chars = new_chars_pages[p] if p < len(new_chars_pages) else []

        # run word/char diff on the page
        page_diff = _diff_pages(old_words, old_chars, new_words, new_chars, p)

        # image-based diff fallback for visual changes (render at moderate zoom)
        try:
            # Only run image diff when both pages exist
            if p < old_doc.page_count and p < new_doc.page_count:
                img1 = _render_page_to_pil(old_path, p, zoom=1.5)
                img2 = _render_page_to_pil(new_path, p, zoom=1.5)
                img_boxes = _image_diff_boxes(img1, img2)
            else:
                img_boxes = []
        except Exception as e:
            _logger.warning(f"Image diff failed on page {p}: {e}")
            img_boxes = []

        page_size = (new_doc[p].rect.width, new_doc[p].rect.height) if p < new_doc.page_count else (old_doc[p].rect.width if p < old_doc.page_count else 0, old_doc[p].rect.height if p < old_doc.page_count else 0)

        results.append({
            "page_index": p,
            "page_size": page_size,
            "ops": page_diff["ops"],
            "image_diff_boxes": img_boxes,
        })

    old_doc.close()
    new_doc.close()
    return results


# -----------------------------
# Annotated PDF generation
# -----------------------------
def _build_summary_pdf(summary_rows: List[Dict[str, Any]], out_path: str, created_by: str = DEFAULT_CREATED_BY):
    """
    Create a PDF page (possibly multi-page if many rows) summarizing all changes.
    summary_rows elements: {"page":int, "change_type": "insert|replace|delete", "old_snippet":..., "new_snippet":...}
    """
    deps = _ensure_deps()
    SimpleDocTemplate = deps["SimpleDocTemplate"]
    Table = deps["Table"]
    TableStyle = deps["TableStyle"]
    Paragraph = deps["Paragraph"]
    Spacer = deps["Spacer"]
    A4 = deps["A4"]
    colors = deps["colors"]
    getSampleStyleSheet = deps["getSampleStyleSheet"]

    # Build table rows
    headers = ["Page", "Type", "Old (snippet)", "New (snippet)"]
    data = [headers]
    for r in summary_rows:
        typ = r.get("change_type", "")
        if typ == "insert":
            disp = "Inserted"
        elif typ == "replace":
            disp = "Changed"
        elif typ == "delete":
            disp = "Removed"
        else:
            disp = typ
        data.append([r.get("page", ""), disp, r.get("old_snippet", ""), r.get("new_snippet", "")])

    # Create document
    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=12, leftMargin=12, topMargin=12, bottomMargin=12)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("PDF Comparison Summary", styles["Heading1"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Created by: {created_by}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table = Table(data, repeatRows=1, colWidths=[40, 80, 220, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(table)
    # Build
    doc.build(elements)


def generate_annotated_pdf(old_input: Any, new_input: Any, output_path: str,
                           highlight_opacity: float = DEFAULT_HIGHLIGHT_OPACITY,
                           created_by: str = DEFAULT_CREATED_BY,
                           return_summary_rows: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate Annotated NEW PDF with highlights and append a Summary Panel page.
    Highlights:
      - Green (50% opacity) for inserted tokens in NEW PDF
      - Red   (50% opacity) for changed/replaced tokens (character-level where possible)
    Removed tokens are NOT highlighted in the body but are included in Summary Panel.
    Returns: (final_annotated_pdf_path, summary_rows)
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]

    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_annot_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new.pdf"))

    _logger.info("Running compare_pdfs to build diff ops...")
    page_diffs = compare_pdfs(old_path, new_path, use_ocr=True)

    # Open NEW PDF to annotate (we annotate FINAL state)
    new_doc = fitz.open(new_path)

    summary_rows: List[Dict[str, Any]] = []

    # Iterate pages and ops
    for page_info in page_diffs:
        pidx = page_info["page_index"]
        _logger.debug(f"Annotating page {pidx+1} of NEW PDF")
        if pidx >= new_doc.page_count:
            # No corresponding new page; nothing to annotate
            continue
        page = new_doc.load_page(pidx)
        ops = page_info.get("ops", [])
        image_boxes = page_info.get("image_diff_boxes", [])

        # If image boxes exist (big graphical differences), convert image boxes to fitz.Rect and annotate lightly
        for b in image_boxes:
            try:
                x0, y0, x1, y1 = map(float, b)
                rect = fitz.Rect(x0, y0, x1, y1)
                # Use a subtle red outline (no fill) to mark image diff region
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=_RED_RGB, fill=None)
                annot.set_opacity(0.35)
                try:
                    annot.update()
                except Exception:
                    pass
            except Exception:
                continue

        # Process each operation
        for op in ops:
            tag = op.get("tag", "")
            rects: List[Tuple[float, float, float, float]] = op.get("rects", []) or []
            # Normalize rects: clip to page rects
            page_rect = (0.0, 0.0, page.rect.width, page.rect.height)
            norm_rects: List[Tuple[float, float, float, float]] = []
            for r in rects:
                if r is None:
                    continue
                # If the bbox is in image coordinates due to OCR, assume it's already in page coords (we used scale mapping earlier)
                # Clip to page bounds
                nr = _clip_rect(r, page_rect)
                if nr:
                    norm_rects.append(nr)

            if tag == "insert":
                # Draw green highlights
                for nr in norm_rects:
                    try:
                        rr = fitz.Rect(nr)
                        annot = page.add_rect_annot(rr)
                        annot.set_colors(stroke=_GREEN_RGB, fill=_GREEN_RGB)
                        annot.set_opacity(float(highlight_opacity))
                        try:
                            annot.update()
                        except Exception:
                            pass
                    except Exception:
                        continue
                # record summary row
                if rects:
                    summary_rows.append({"page": pidx + 1, "change_type": "insert", "old_snippet": "", "new_snippet": op.get("new_snippet", "")})
            elif tag == "replace":
                # Draw red highlights for changed parts (char-level if available)
                if norm_rects:
                    for nr in norm_rects:
                        try:
                            rr = fitz.Rect(nr)
                            annot = page.add_rect_annot(rr)
                            annot.set_colors(stroke=_RED_RGB, fill=_RED_RGB)
                            annot.set_opacity(float(highlight_opacity))
                            try:
                                annot.update()
                            except Exception:
                                pass
                        except Exception:
                            continue
                else:
                    # no rects discovered - nothing to annotate; fallback: nothing
                    pass
                # record summary row
                summary_rows.append({"page": pidx + 1, "change_type": "replace", "old_snippet": op.get("old_snippet", ""), "new_snippet": op.get("new_snippet", "")})
            elif tag == "delete":
                # Deletions are not annotated in body; record in summary
                summary_rows.append({"page": pidx + 1, "change_type": "delete", "old_snippet": op.get("old_snippet", ""), "new_snippet": ""})
            else:
                # unknown tag - ignore for now
                continue

    # Save temporary annotated NEW PDF (without summary yet)
    annotated_tmp = os.path.join(tmpdir, "annotated_new_tmp.pdf")
    new_doc.save(annotated_tmp, garbage=4, deflate=True)
    new_doc.close()

    # Build summary panel PDF
    summary_tmp = os.path.join(tmpdir, "summary_panel.pdf")
    _build_summary_pdf(summary_rows, summary_tmp, created_by=created_by)

    # Append summary panel to annotated_tmp
    final_annotated = os.path.join(tmpdir, "annotated_with_summary.pdf")
    _append_pdf_files(annotated_tmp, summary_tmp, final_annotated)

    # Copy final_annotated to requested output_path
    _ensure_dir(os.path.dirname(output_path) or ".")
    with open(final_annotated, "rb") as fin, open(output_path, "wb") as fout:
        fout.write(fin.read())

    if return_summary_rows:
        return output_path, summary_rows
    else:
        return output_path, []


# -----------------------------
# Append PDF helper (fitz)
# -----------------------------
def _append_pdf_files(base_pdf_path: str, append_pdf_path: str, out_path: str):
    """
    Append pages from append_pdf_path to base_pdf_path and write to out_path.
    """
    deps = _ensure_deps()
    fitz = deps["fitz"]
    base = fitz.open(base_pdf_path)
    app = fitz.open(append_pdf_path)
    base.insert_pdf(app)
    base.save(out_path, garbage=4, deflate=True)
    base.close()
    app.close()


# -----------------------------
# Side-by-Side PDF generator (image-based)
# -----------------------------
def generate_side_by_side_pdf(old_input: Any, new_input: Any, output_path: str, zoom: float = 1.5, gap: int = 12) -> str:
    """
    Render each page from old/new PDF to images and place side-by-side into output_path.
    Uses PIL image composition so layout and fonts are preserved visually.
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
        for p in range(max_pages):
            # Render old page image or blank image if missing
            if p < old_doc.page_count:
                op = old_doc.load_page(p)
                op_pix = op.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                o_img = Image.open(io.BytesIO(op_pix.tobytes("png")))
            else:
                o_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255, 255, 255))

            # Render new page image or blank
            if p < new_doc.page_count:
                npg = new_doc.load_page(p)
                np_pix = npg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                n_img = Image.open(io.BytesIO(np_pix.tobytes("png")))
            else:
                n_img = Image.new("RGB", (int(595 * zoom), int(842 * zoom)), (255, 255, 255))

            # Scale both images to same target height (preserve aspect ratio)
            target_h = max(o_img.height, n_img.height)

            def _scale_to_height(img, h):
                w, hh = img.size
                new_w = int(w * (h / hh))
                return img.resize((new_w, h), Image.LANCZOS)

            o_resized = _scale_to_height(o_img, target_h)
            n_resized = _scale_to_height(n_img, target_h)

            o_w, _ = o_resized.size
            n_w, _ = n_resized.size

            page_w = o_w + gap + n_w
            page_h = target_h

            # create new PDF page sized to hold both images
            page = out_doc.new_page(width=page_w, height=page_h)

            # Insert left image
            buf = io.BytesIO()
            o_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(0, 0, o_w, page_h), stream=buf.getvalue())

            # Insert right image
            buf = io.BytesIO()
            n_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(o_w + gap, 0, o_w + gap + n_w, page_h), stream=buf.getvalue())

        # Save output
        _ensure_dir(os.path.dirname(output_path) or ".")
        out_doc.save(output_path, garbage=4, deflate=True)
    finally:
        old_doc.close()
        new_doc.close()
        out_doc.close()

    return output_path


# -----------------------------
# Full pipeline wrapper
# -----------------------------
def process_and_generate(old_input: Any, new_input: Any, workdir: Optional[str] = None,
                         highlight_opacity: float = DEFAULT_HIGHLIGHT_OPACITY,
                         created_by: str = DEFAULT_CREATED_BY) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Full pipeline convenience function:
      - writes inputs to tmp if file-like
      - runs compare_pdfs to build diffs
      - generates annotated NEW PDF with highlights and summary appended
      - generates side-by-side visual PDF
    Returns:
      (annotated_with_summary_path, side_by_side_path, summary_rows)
    """
    tmpdir = workdir or tempfile.mkdtemp(prefix="pdfdiff_pipeline_")
    old_path = _save_input_to_path(old_input, os.path.join(tmpdir, "old_input.pdf"))
    new_path = _save_input_to_path(new_input, os.path.join(tmpdir, "new_input.pdf"))

    # Annotated PDF
    annotated_out = os.path.join(tmpdir, "annotated_comparison.pdf")
    annotated_path, summary_rows = generate_annotated_pdf(old_path, new_path, annotated_out, highlight_opacity=highlight_opacity, created_by=created_by, return_summary_rows=True)

    # Side-by-side PDF
    side_by_side_out = os.path.join(tmpdir, "side_by_side.pdf")
    generate_side_by_side_pdf(old_path, new_path, side_by_side_out)

    return annotated_path, side_by_side_out, summary_rows


# -----------------------------
# Backwards compatibility wrappers & convenience exports
# -----------------------------
def generate_annotated_pdf_wrapper(old_input: Any, new_input: Any, output_path: str):
    """
    Simple wrapper to maintain older function signatures that expect generate_annotated_pdf(old, new, out)
    This wrapper will call the expanded generate_annotated_pdf and return only path.
    """
    annotated_path, _ = generate_annotated_pdf(old_input, new_input, output_path, highlight_opacity=DEFAULT_HIGHLIGHT_OPACITY, created_by=DEFAULT_CREATED_BY, return_summary_rows=False)
    return annotated_path


# Public API
__all__ = [
    "compare_pdfs",
    "generate_annotated_pdf",
    "generate_side_by_side_pdf",
    "process_and_generate",
    "generate_annotated_pdf_wrapper",
]

# End of file













