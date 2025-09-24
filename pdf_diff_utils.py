# pdf_diff_utils.py
"""
Utilities to:
- extract text lines + bounding boxes from PDF pages (using PyMuPDF)
- diff lines page-by-page
- annotate PDFs (rect annotations)
- create side-by-side PDF (render pages to images and place left/right)
- build a summary table (PDF) using reportlab
"""

import fitz  # PyMuPDF
import re
from difflib import SequenceMatcher
import math
import io
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import numpy as np
import os

# Helper: convert 0-255 RGB to 0-1 float tuple
def rgb255_to_float(rgb):
    r, g, b = rgb
    return (r / 255.0, g / 255.0, b / 255.0)

COLORS = {
    "red": rgb255_to_float((255, 80, 80)),
    "green": rgb255_to_float((60, 170, 60)),
    "black": rgb255_to_float((0, 0, 0)),
}

number_regex = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def extract_lines_with_bboxes(pdf_path):
    """
    Returns:
      pages_lines: list where each element is a list of dicts:
         {'text':str, 'bbox':(x0,y0,x1,y1)}
    """
    doc = fitz.open(pdf_path)
    pages_lines = []
    for pageno in range(doc.page_count):
        page = doc.load_page(pageno)
        words = page.get_text("words")  # list of tuples (x0,y0,x1,y1, "word", block_no, line_no, word_no)
        # Group words by (block_no, line_no)
        words.sort(key=lambda w: (w[5], w[6], w[0]))
        lines = []
        current_key = None
        current_words = []
        for w in words:
            key = (w[5], w[6])
            if key != current_key:
                # flush
                if current_words:
                    xs = [t[0] for t in current_words]
                    ys = [t[1] for t in current_words]
                    xs2 = [t[2] for t in current_words]
                    ys2 = [t[3] for t in current_words]
                    text = " ".join([t[4] for t in current_words])
                    bbox = (min(xs), min(ys), max(xs2), max(ys2))
                    lines.append({"text": text.strip(), "bbox": bbox})
                current_words = [w]
                current_key = key
            else:
                current_words.append(w)
        # final flush
        if current_words:
            xs = [t[0] for t in current_words]
            ys = [t[1] for t in current_words]
            xs2 = [t[2] for t in current_words]
            ys2 = [t[3] for t in current_words]
            text = " ".join([t[4] for t in current_words])
            bbox = (min(xs), min(ys), max(xs2), max(ys2))
            lines.append({"text": text.strip(), "bbox": bbox})
        pages_lines.append(lines)
    doc.close()
    return pages_lines

def numeric_tokens_from_text(s):
    return number_regex.findall(s)

def try_parse_number(tok):
    try:
        # remove commas
        t = tok.replace(",", "")
        if "." in t:
            return float(t)
        else:
            return int(t)
    except:
        return None

def compare_pages_lines(old_lines, new_lines, numeric_detection=True):
    """
    Aligns lists of lines (strings) using SequenceMatcher and classifies
    'equal', 'insert', 'delete', 'replace'.

    Returns a list of diff entries:
      each entry: {
        'type': 'equal'|'insert'|'delete'|'replace',
        'old_text': str or '',
        'new_text': str or '',
        'old_bbox': bbox or None,
        'new_bbox': bbox or None,
        'numeric_info': { 'old_val':..., 'new_val':..., 'trend': 'increased'|'decreased'|'changed' } or None
      }
    """
    old_texts = [l["text"] for l in old_lines]
    new_texts = [l["text"] for l in new_lines]

    sm = SequenceMatcher(None, old_texts, new_texts, autojunk=False)
    ops = sm.get_opcodes()
    diffs = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for oi, nj in zip(range(i1, i2), range(j1, j2)):
                diffs.append({
                    "type": "equal",
                    "old_text": old_texts[oi],
                    "new_text": new_texts[nj],
                    "old_bbox": old_lines[oi]["bbox"],
                    "new_bbox": new_lines[nj]["bbox"],
                    "numeric_info": None
                })
        elif tag == "replace":
            # Pair as many as possible
            len_old = i2 - i1
            len_new = j2 - j1
            pairs = max(len_old, len_new)
            for k in range(pairs):
                oi = i1 + k if (i1 + k) < i2 else None
                nj = j1 + k if (j1 + k) < j2 else None
                old_text = old_lines[oi]["text"] if oi is not None else ""
                new_text = new_lines[nj]["text"] if nj is not None else ""
                old_bbox = old_lines[oi]["bbox"] if oi is not None else None
                new_bbox = new_lines[nj]["bbox"] if nj is not None else None
                numeric_info = None
                if numeric_detection:
                    olds = numeric_tokens_from_text(old_text)
                    news = numeric_tokens_from_text(new_text)
                    if olds and news:
                        # compare first numeric token heuristically
                        old_val = try_parse_number(olds[0])
                        new_val = try_parse_number(news[0])
                        if (old_val is not None) and (new_val is not None):
                            trend = "increased" if new_val > old_val else ("decreased" if new_val < old_val else "unchanged")
                            numeric_info = {"old_val": old_val, "new_val": new_val, "trend": trend}
                diffs.append({
                    "type": "replace",
                    "old_text": old_text,
                    "new_text": new_text,
                    "old_bbox": old_bbox,
                    "new_bbox": new_bbox,
                    "numeric_info": numeric_info
                })
        elif tag == "delete":
            for oi in range(i1, i2):
                diffs.append({
                    "type": "delete",
                    "old_text": old_lines[oi]["text"],
                    "new_text": "",
                    "old_bbox": old_lines[oi]["bbox"],
                    "new_bbox": None,
                    "numeric_info": None
                })
        elif tag == "insert":
            for nj in range(j1, j2):
                diffs.append({
                    "type": "insert",
                    "old_text": "",
                    "new_text": new_lines[nj]["text"],
                    "old_bbox": None,
                    "new_bbox": new_lines[nj]["bbox"],
                    "numeric_info": None
                })
    return diffs

def compare_pdfs_and_annotate(old_pdf_path, new_pdf_path, out_dir, enable_ocr=False, opacity=0.28, numeric_detection=True):
    """
    Main orchestrator:
      - Extract line bboxes for both PDFs
      - Compare page-by-page
      - Annotate old PDF for deletions/reductions (red)
      - Annotate new PDF for insertions/increases (green)
      - Save annotated PDFs to out_dir and return their paths and a diff summary list
    """
    # extract line-level info
    old_pages = extract_lines_with_bboxes(old_pdf_path)
    new_pages = extract_lines_with_bboxes(new_pdf_path)

    max_pages = max(len(old_pages), len(new_pages))
    diffs_all = []  # summary list to build summary table
    # We'll create editable copies of each doc to annotate
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)

    for p in range(max_pages):
        old_lines = old_pages[p] if p < len(old_pages) else []
        new_lines = new_pages[p] if p < len(new_pages) else []
        page_diffs = compare_pages_lines(old_lines, new_lines, numeric_detection=numeric_detection)
        # process diffs: add annotations
        for d in page_diffs:
            entry = {"page": p + 1, "type": d["type"], "old_text": d["old_text"], "new_text": d["new_text"]}
            # if it's an insertion, highlight new_doc page
            if d["type"] == "insert":
                if d["new_bbox"] and p < new_doc.page_count:
                    page = new_doc.load_page(p)
                    rect = fitz.Rect(d["new_bbox"])
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=COLORS["green"], fill=COLORS["green"])
                    annot.set_opacity(opacity)
                    annot.update()
                entry["color"] = "green"
                diffs_all.append(entry)
            elif d["type"] == "delete":
                if d["old_bbox"] and p < old_doc.page_count:
                    page = old_doc.load_page(p)
                    rect = fitz.Rect(d["old_bbox"])
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=COLORS["red"], fill=COLORS["red"])
                    annot.set_opacity(opacity)
                    annot.update()
                entry["color"] = "red"
                diffs_all.append(entry)
            elif d["type"] == "replace":
                # if numeric_info present, annotate old/new appropriately
                if d["numeric_info"] is not None:
                    trend = d["numeric_info"]["trend"]
                    entry["old_val"] = d["numeric_info"]["old_val"]
                    entry["new_val"] = d["numeric_info"]["new_val"]
                    entry["trend"] = trend
                    if p < old_doc.page_count and d["old_bbox"]:
                        page = old_doc.load_page(p)
                        rect = fitz.Rect(d["old_bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=COLORS["red"], fill=COLORS["red"])
                        annot.set_opacity(opacity)
                        annot.update()
                    if p < new_doc.page_count and d["new_bbox"]:
                        page = new_doc.load_page(p)
                        rect = fitz.Rect(d["new_bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=COLORS["green"], fill=COLORS["green"])
                        annot.set_opacity(opacity)
                        annot.update()
                    entry["color"] = "green" if trend == "increased" else "red" if trend == "decreased" else "black"
                    diffs_all.append(entry)
                else:
                    # generic replace: mark old red and new green
                    if p < old_doc.page_count and d["old_bbox"]:
                        page = old_doc.load_page(p)
                        rect = fitz.Rect(d["old_bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=COLORS["red"], fill=COLORS["red"])
                        annot.set_opacity(opacity)
                        annot.update()
                    if p < new_doc.page_count and d["new_bbox"]:
                        page = new_doc.load_page(p)
                        rect = fitz.Rect(d["new_bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=COLORS["green"], fill=COLORS["green"])
                        annot.set_opacity(opacity)
                        annot.update()
                    entry["color"] = "green"
                    diffs_all.append(entry)
            elif d["type"] == "equal":
                # no highlight required, but could record equal entries if desired
                pass

    # Save annotated PDFs
    annotated_old = os.path.join(out_dir, "annotated_old.pdf")
    annotated_new = os.path.join(out_dir, "annotated_new.pdf")
    old_doc.save(annotated_old, garbage=4, deflate=True)
    new_doc.save(annotated_new, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()

    # Build a condensed diff summary list of dicts (for summary page)
    summary_rows = []
    for d in diffs_all:
        snippet_old = (d.get("old_text") or "")[:120]
        snippet_new = (d.get("new_text") or "")[:120]
        summary_rows.append({
            "page": d.get("page"),
            "change_type": d.get("type"),
            "color": d.get("color", ""),
            "old_snippet": snippet_old,
            "new_snippet": snippet_new,
            "old_val": d.get("old_val", ""),
            "new_val": d.get("new_val", ""),
            "trend": d.get("trend", "")
        })

    return annotated_old, annotated_new, summary_rows

def append_pdf(base_pdf, to_append_pdf, out_path):
    """
    Append PDF `to_append_pdf` to end of `base_pdf` and write to out_path
    """
    base = fitz.open(base_pdf)
    app = fitz.open(to_append_pdf)
    base.insert_pdf(app)
    base.save(out_path, garbage=4, deflate=True)
    base.close()
    app.close()

def merge_summary_into_pdf(diff_summary, summary_pdf_path, created_by=""):
    """
    Build a single-page (or multi-page if needed) PDF containing a summary table of diffs.
    Uses reportlab to create a PDF saved at summary_pdf_path.
    """
    # Create table data
    headers = ["Page", "Change", "Old (snippet)", "New (snippet)", "Old val", "New val"]
    rows = [headers]
    for r in diff_summary:
        rows.append([
            r.get("page"),
            r.get("trend") or r.get("change_type"),
            (r.get("old_snippet") or "")[:100],
            (r.get("new_snippet") or "")[:100],
            str(r.get("old_val") or ""),
            str(r.get("new_val") or "")
        ])

    # Use A4 portrait pages; table may overflow to multiple pages
    doc = SimpleDocTemplate(summary_pdf_path, pagesize=A4, rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18)
    elements = []
    style = getSampleStyleSheet()["BodyText"]
    title_style = getSampleStyleSheet()["Heading1"]
    elements.append(Paragraph("PDF Comparison Summary", title_style))
    elements.append(Spacer(1, 8))
    if created_by:
        elements.append(Paragraph(f"Created by: {created_by}", style))
        elements.append(Spacer(1, 10))

    # Build table with style
    table = Table(rows, repeatRows=1, colWidths=[40, 80, 200, 200, 60, 60])
    tblstyle = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]
    )
    table.setStyle(tblstyle)
    elements.append(table)
    doc.build(elements)

def create_side_by_side_pdf(old_pdf_path, new_pdf_path, out_path):
    """
    Creates a PDF where each page is a side-by-side rendering:
      left: old pdf page image
      right: new pdf page image (already annotated)
    No summary panel included in these pages.
    """
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)
    # output doc
    out_doc = fitz.open()

    max_pages = max(old_doc.page_count, new_doc.page_count)
    for i in range(max_pages):
        # render pages to pixmap
        if i < old_doc.page_count:
            old_page = old_doc.load_page(i)
            old_pix = old_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # increase resolution
            old_img = old_pix.tobytes("png")
            ow, oh = old_pix.width, old_pix.height
        else:
            old_img = None
            ow = oh = 1

        if i < new_doc.page_count:
            new_page = new_doc.load_page(i)
            new_pix = new_page.get_pixmap(matrix=fitz.Matrix(2, 2))
            new_img = new_pix.tobytes("png")
            nw, nh = new_pix.width, new_pix.height
        else:
            new_img = None
            nw = nh = 1

        # Normalize heights by using max height and scale widths accordingly
        target_h = max(oh, nh)
        # scale images to same height and create combined width
        if old_img:
            old_img_pil = Image.open(io.BytesIO(old_img))
            scale_old = target_h / old_img_pil.height
            new_ow = int(old_img_pil.width * scale_old)
        else:
            new_ow = 1
        if new_img:
            new_img_pil = Image.open(io.BytesIO(new_img))
            scale_new = target_h / new_img_pil.height
            new_nw = int(new_img_pil.width * scale_new)
        else:
            new_nw = 1

        page_w = new_ow + new_nw
        page_h = target_h

        # create an empty page with width page_w and height page_h
        page = out_doc.new_page(width=page_w, height=page_h)

        # insert old image at left
        x = 0
        if old_img:
            old_img_pil = old_img_pil.resize((new_ow, page_h), Image.LANCZOS)
            buffer = io.BytesIO()
            old_img_pil.save(buffer, format="PNG")
            page.insert_image(fitz.Rect(x, 0, x + new_ow, page_h), stream=buffer.getvalue())
        x += new_ow
        # insert new image at right
        if new_img:
            new_img_pil = new_img_pil.resize((new_nw, page_h), Image.LANCZOS)
            buffer = io.BytesIO()
            new_img_pil.save(buffer, format="PNG")
            page.insert_image(fitz.Rect(x, 0, x + new_nw, page_h), stream=buffer.getvalue())

    out_doc.save(out_path, garbage=4, deflate=True)
    out_doc.close()
    old_doc.close()
    new_doc.close()
