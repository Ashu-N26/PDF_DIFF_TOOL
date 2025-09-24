# pdf_diff_utils.py
"""
Updated utilities for PDF diffing with the following rules:
- Red highlight: changed data (highlighted on NEW PDF only)
- Green highlight: inserted data (highlighted on NEW PDF)
- Removed data: NOT highlighted; instead listed in summary panel appended to final PDF
- No extra highlighting anywhere
"""

import fitz  # PyMuPDF
import re
from difflib import SequenceMatcher
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
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
        words = page.get_text("words")  # (x0,y0,x1,y1, "word", block_no, line_no, word_no)
        words.sort(key=lambda w: (w[5], w[6], w[0]))
        lines = []
        current_key = None
        current_words = []
        for w in words:
            key = (w[5], w[6])
            if key != current_key:
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
    New behavior to match user rules:
      - Changed data (replace) -> highlight NEW PDF only, with RED
      - Inserted data -> highlight NEW PDF only, with GREEN
      - Deleted data -> do NOT highlight OLD PDF; instead record in summary for later inclusion
      - No highlights on OLD PDF (no extra highlighting)
    Returns (annotated_old_path, annotated_new_path, summary_rows)
    Note: annotated_old_path will be a copy of old_pdf without highlights (kept for convenience),
          annotated_new_path contains the red/green highlights per rule.
    """
    old_pages = extract_lines_with_bboxes(old_pdf_path)
    new_pages = extract_lines_with_bboxes(new_pdf_path)

    max_pages = max(len(old_pages), len(new_pages))
    summary_items = []  # collects entries for summary panel (including removed items)
    # Make copies for saving; OLD will remain un-highlighted per rule (but we save a copy)
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)

    for p in range(max_pages):
        old_lines = old_pages[p] if p < len(old_pages) else []
        new_lines = new_pages[p] if p < len(new_pages) else []
        page_diffs = compare_pages_lines(old_lines, new_lines, numeric_detection=numeric_detection)

        for d in page_diffs:
            # For insert -> highlight NEW (green)
            if d["type"] == "insert":
                if d["new_bbox"] and p < new_doc.page_count:
                    page = new_doc.load_page(p)
                    rect = fitz.Rect(d["new_bbox"])
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=COLORS["green"], fill=COLORS["green"])
                    annot.set_opacity(opacity)
                    annot.update()
                # add to summary as an inserted row (so summary shows inserted items too)
                summary_items.append({
                    "page": p + 1,
                    "change_type": "insert",
                    "old_snippet": "",
                    "new_snippet": d["new_text"][:200],
                    "old_val": "",
                    "new_val": ""
                })

            # For delete -> DO NOT highlight anywhere. Add an entry to summary indicating removal.
            elif d["type"] == "delete":
                summary_items.append({
                    "page": p + 1,
                    "change_type": "delete",
                    "old_snippet": d["old_text"][:200],
                    "new_snippet": "",
                    "old_val": "",
                    "new_val": ""
                })
                # No annotation performed

            # For replace -> treat as CHANGED. Highlight NEW PDF with RED only.
            elif d["type"] == "replace":
                # record numeric info if exists
                if d.get("numeric_info") is not None:
                    ni = d["numeric_info"]
                    # We still highlight NEW text in red (changed)
                    if d["new_bbox"] and p < new_doc.page_count:
                        page = new_doc.load_page(p)
                        rect = fitz.Rect(d["new_bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=COLORS["red"], fill=COLORS["red"])
                        annot.set_opacity(opacity)
                        annot.update()
                    summary_items.append({
                        "page": p + 1,
                        "change_type": "replace",
                        "old_snippet": d["old_text"][:200],
                        "new_snippet": d["new_text"][:200],
                        "old_val": ni.get("old_val"),
                        "new_val": ni.get("new_val"),
                        "trend": ni.get("trend")
                    })
                else:
                    # generic text change: highlight NEW element red only
                    if d["new_bbox"] and p < new_doc.page_count:
                        page = new_doc.load_page(p)
                        rect = fitz.Rect(d["new_bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=COLORS["red"], fill=COLORS["red"])
                        annot.set_opacity(opacity)
                        annot.update()
                    summary_items.append({
                        "page": p + 1,
                        "change_type": "replace",
                        "old_snippet": d["old_text"][:200],
                        "new_snippet": d["new_text"][:200],
                        "old_val": "",
                        "new_val": ""
                    })

            # equal -> do nothing
            else:
                pass

    # Save annotated outputs
    # Per rule: old PDF should have no highlights — we save original as annotated_old (copy)
    annotated_old = os.path.join(out_dir, "annotated_old_no_highlights.pdf")
    annotated_new = os.path.join(out_dir, "annotated_new_highlights.pdf")
    # Save copies
    old_doc.save(annotated_old, garbage=4, deflate=True)
    new_doc.save(annotated_new, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()

    # Build summary rows suitable for the summary panel
    summary_rows = []
    for r in summary_items:
        summary_rows.append({
            "page": r.get("page"),
            "change_type": r.get("change_type"),
            "old_snippet": r.get("old_snippet", ""),
            "new_snippet": r.get("new_snippet", ""),
            "old_val": r.get("old_val", ""),
            "new_val": r.get("new_val", ""),
            "trend": r.get("trend", "")
        })

    return annotated_old, annotated_new, summary_rows

def append_pdf(base_pdf, to_append_pdf, out_path):
    base = fitz.open(base_pdf)
    app = fitz.open(to_append_pdf)
    base.insert_pdf(app)
    base.save(out_path, garbage=4, deflate=True)
    base.close()
    app.close()

def merge_summary_into_pdf(diff_summary, summary_pdf_path, created_by=""):
    """
    Build a PDF summary panel that includes:
     - Removed items (change_type == 'delete') clearly labelled
     - Inserted items and replaced items with snippets and numeric old/new values if present
    """
    headers = ["Page", "Change Type", "Old (snippet)", "New (snippet)", "Old val", "New val", "Trend"]
    rows = [headers]
    for r in diff_summary:
        rows.append([
            r.get("page"),
            r.get("change_type"),
            (r.get("old_snippet") or "")[:120],
            (r.get("new_snippet") or "")[:120],
            str(r.get("old_val") or ""),
            str(r.get("new_val") or ""),
            str(r.get("trend") or "")
        ])

    doc = SimpleDocTemplate(summary_pdf_path, pagesize=A4, rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18)
    elements = []
    style = getSampleStyleSheet()["BodyText"]
    title_style = getSampleStyleSheet()["Heading1"]
    elements.append(Paragraph("PDF Comparison Summary", title_style))
    elements.append(Spacer(1, 6))
    if created_by:
        elements.append(Paragraph(f"Created by: {created_by}", style))
        elements.append(Spacer(1, 8))

    table = Table(rows, repeatRows=1, colWidths=[40, 70, 160, 160, 60, 60, 50])
    tblstyle = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ])
    table.setStyle(tblstyle)
    elements.append(table)
    doc.build(elements)

def create_side_by_side_pdf(old_pdf_path, new_pdf_path, out_path):
    """
    Creates side-by-side pages:
      left: old PDF page image (no highlights)
      right: new PDF page image (with highlights)
    No summary panel on this file per requirement.
    """
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)
    out_doc = fitz.open()

    max_pages = max(old_doc.page_count, new_doc.page_count)
    for i in range(max_pages):
        # render pages
        old_img = None
        new_img = None
        if i < old_doc.page_count:
            old_page = old_doc.load_page(i)
            old_pix = old_page.get_pixmap(matrix=fitz.Matrix(2, 2))
            old_img = old_pix.tobytes("png")
            old_pil = Image.open(io.BytesIO(old_img))
        if i < new_doc.page_count:
            new_page = new_doc.load_page(i)
            new_pix = new_page.get_pixmap(matrix=fitz.Matrix(2, 2))
            new_img = new_pix.tobytes("png")
            new_pil = Image.open(io.BytesIO(new_img))

        # handle missing pages gracefully
        ow, oh = (old_pil.width, old_pil.height) if old_img else (1, 1)
        nw, nh = (new_pil.width, new_pil.height) if new_img else (1, 1)
        target_h = max(oh, nh)
        # scale widths
        if old_img:
            scale_old = target_h / old_pil.height
            new_ow = int(old_pil.width * scale_old)
            old_resized = old_pil.resize((new_ow, target_h), Image.LANCZOS)
        else:
            new_ow = int(0.5 * (target_h))  # placeholder
            old_resized = Image.new("RGB", (new_ow, target_h), (255, 255, 255))
        if new_img:
            scale_new = target_h / new_pil.height
            new_nw = int(new_pil.width * scale_new)
            new_resized = new_pil.resize((new_nw, target_h), Image.LANCZOS)
        else:
            new_nw = int(0.5 * (target_h))
            new_resized = Image.new("RGB", (new_nw, target_h), (255, 255, 255))

        page_w = new_ow + new_nw
        page_h = target_h
        page = out_doc.new_page(width=page_w, height=page_h)

        # left image
        buffer = io.BytesIO()
        old_resized.save(buffer, format="PNG")
        page.insert_image(fitz.Rect(0, 0, new_ow, page_h), stream=buffer.getvalue())
        # right image
        buffer = io.BytesIO()
        new_resized.save(buffer, format="PNG")
        page.insert_image(fitz.Rect(new_ow, 0, new_ow + new_nw, page_h), stream=buffer.getvalue())

    out_doc.save(out_path, garbage=4, deflate=True)
    out_doc.close()
    old_doc.close()
    new_doc.close()

