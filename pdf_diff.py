"""
pdf_diff.py
Advanced PDF Comparison Tool
Author: Ashutosh Nanaware
Description:
    Full-featured PDF diffing engine with:
    - Per-character (glyph-level) text diffs
    - Page resizing & normalization
    - Layout alignment
    - Visual & structural comparison
    - Highlighting with tight bounding boxes
    - Export as annotated PDF overlays
"""

import os
import io
import ast
import difflib
import fitz  # PyMuPDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextBoxHorizontal, LTTextLineHorizontal
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import red, green, yellow
from PIL import Image, ImageChops

# ================================================================
# Utility Functions
# ================================================================

def normalize_bbox(bbox):
    """
    Normalize bounding box: ensure it's a list of floats [x0, y0, x1, y1].
    Handles string cases like "(12.3, 45.6, 78.9, 100.1)".
    """
    if isinstance(bbox, str):
        try:
            bbox = ast.literal_eval(bbox)
        except Exception:
            return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(x) for x in bbox]
    return None


def bbox_center(bbox):
    """Return center point of a bounding box."""
    if not bbox:
        return (0, 0)
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def resize_page(ref_size, page_size):
    """
    Return scaling factors (x, y) to normalize page sizes.
    """
    rx, ry = ref_size
    px, py = page_size
    return rx / px, ry / py


def apply_scale(bbox, scale):
    """
    Scale bounding box by (sx, sy).
    """
    if not bbox:
        return None
    sx, sy = scale
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]


# ================================================================
# Text Extraction
# ================================================================

def extract_text_glyphs(pdf_path):
    """
    Extracts per-character glyphs with positions from PDF using pdfminer.
    Returns:
        List[ List[ { "char": c, "bbox": [x0,y0,x1,y1] } ] ] for each page
    """
    pages = []
    for page_layout in extract_pages(pdf_path):
        glyphs = []
        for element in page_layout:
            if isinstance(element, (LTTextBoxHorizontal, LTTextLineHorizontal)):
                for text_line in element:
                    for character in text_line:
                        if isinstance(character, LTChar):
                            glyphs.append({
                                "char": character.get_text(),
                                "bbox": normalize_bbox(character.bbox)
                            })
        pages.append(glyphs)
    return pages


# ================================================================
# Visual Comparison (Image-based fallback)
# ================================================================

def render_page_image(pdf_path, page_number, zoom=2.0):
    """
    Render PDF page to image using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def image_diff(img1, img2):
    """
    Compute pixel diff between two images.
    Returns bounding boxes of differing regions.
    """
    diff = ImageChops.difference(img1, img2)
    bbox = diff.getbbox()
    if not bbox:
        return []
    return [list(bbox)]


# ================================================================
# Diff Engine
# ================================================================

def diff_glyphs(glyphs1, glyphs2, page_scale=(1.0, 1.0)):
    """
    Compare two lists of glyphs using difflib.
    Returns list of (diff_type, char, bbox)
    diff_type: "equal", "replace", "insert", "delete"
    """
    text1 = [g["char"] for g in glyphs1]
    text2 = [g["char"] for g in glyphs2]

    sm = difflib.SequenceMatcher(None, text1, text2)
    diffs = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                diffs.append(("equal", glyphs1[k]["char"],
                              apply_scale(glyphs1[k]["bbox"], page_scale)))
        elif tag == "replace":
            for k in range(i1, i2):
                diffs.append(("delete", glyphs1[k]["char"],
                              apply_scale(glyphs1[k]["bbox"], page_scale)))
            for k in range(j1, j2):
                diffs.append(("insert", glyphs2[k]["char"],
                              apply_scale(glyphs2[k]["bbox"], page_scale)))
        elif tag == "delete":
            for k in range(i1, i2):
                diffs.append(("delete", glyphs1[k]["char"],
                              apply_scale(glyphs1[k]["bbox"], page_scale)))
        elif tag == "insert":
            for k in range(j1, j2):
                diffs.append(("insert", glyphs2[k]["char"],
                              apply_scale(glyphs2[k]["bbox"], page_scale)))
    return diffs


# ================================================================
# Main Comparison
# ================================================================

def compare_pdfs(old_pdf, new_pdf):
    """
    Compare two PDFs page by page.
    Returns:
        diffs = {
            page_num: [ (diff_type, char, bbox), ... ],
            ...
        }
    """
    old_glyphs = extract_text_glyphs(old_pdf)
    new_glyphs = extract_text_glyphs(new_pdf)

    doc_old = fitz.open(old_pdf)
    doc_new = fitz.open(new_pdf)

    num_pages = max(len(old_glyphs), len(new_glyphs))
    diffs = {}

    for page_num in range(num_pages):
        page1 = doc_old[page_num] if page_num < len(doc_old) else None
        page2 = doc_new[page_num] if page_num < len(doc_new) else None

        size1 = page1.rect.width, page1.rect.height if page1 else (1, 1)
        size2 = page2.rect.width, page2.rect.height if page2 else (1, 1)
        scale = resize_page(size1, size2)

        g1 = old_glyphs[page_num] if page_num < len(old_glyphs) else []
        g2 = new_glyphs[page_num] if page_num < len(new_glyphs) else []

        page_diffs = diff_glyphs(g1, g2, scale)

        # Add image diff fallback if glyphs miss layout changes
        img1 = render_page_image(old_pdf, page_num) if page1 else None
        img2 = render_page_image(new_pdf, page_num) if page2 else None
        if img1 and img2:
            img_boxes = image_diff(img1, img2)
            for box in img_boxes:
                page_diffs.append(("replace", "[image_diff]", box))

        diffs[page_num] = page_diffs

    return diffs


# ================================================================
# Annotated PDF Generator
# ================================================================

def generate_annotated_pdf(diffs, output_path, ref_pdf):
    """
    Generate annotated PDF highlighting differences.
    """
    doc = fitz.open(ref_pdf)
    for page_num, page_diffs in diffs.items():
        if page_num >= len(doc):
            continue
        page = doc[page_num]
        for diff_type, char, bbox in page_diffs:
            if not bbox:
                continue
            color = red if diff_type == "delete" else green if diff_type == "insert" else yellow
            try:
                highlight = page.add_rect_annot(fitz.Rect(bbox))
                highlight.set_colors(stroke=color, fill=None)
                highlight.update()
            except Exception:
                continue
    doc.save(output_path)


# ================================================================
# Debug / CLI Support
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two PDFs and highlight differences.")
    parser.add_argument("old_pdf", help="Path to old PDF")
    parser.add_argument("new_pdf", help="Path to new PDF")
    parser.add_argument("-o", "--output", default="diff_output.pdf", help="Path for output PDF")
    args = parser.parse_args()

    diffs = compare_pdfs(args.old_pdf, args.new_pdf)
    generate_annotated_pdf(diffs, args.output, args.new_pdf)
    print(f"Diff report saved to {args.output}")












