# pdf_diff_utils.py
"""
Core utilities for token-level PDF diff + annotation.

Approach:
- Use PyMuPDF (fitz) to extract page words with coordinates (get_text('words')).
- For each page, build token sequences for old and new with word-level bboxes.
- Use difflib.SequenceMatcher on token strings to detect equal/insert/delete/replace opcodes.
- For 'insert' tokens -> highlight NEW token in GREEN (opacity param).
- For 'replace' tokens -> highlight NEW token in RED (opacity param). We do NOT highlight old.
- For 'delete' tokens -> record in summary (no highlight anywhere).
- Build a summary list (rows) with page, type, old_snippet, new_snippet, and for numeric token replace old/new values when detectable.
- Create summary PDF with reportlab and append to annotated NEW PDF.

- Also create a side-by-side PDF by rendering pages to images and placing old left, new(right with highlights).
"""

import fitz  # PyMuPDF
from difflib import SequenceMatcher
import re
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import math

number_regex = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def rgb_float(r, g, b):
    return (r/255.0, g/255.0, b/255.0)

GREEN = rgb_float(60, 170, 60)
RED = rgb_float(255, 80, 80)

def extract_page_tokens(pdf_path):
    """
    For each page return a list of tokens:
      pages = [
        [ {'text': token_str, 'bbox': (x0,y0,x1,y1), 'word_index': idx}, ... ],
        ...
      ]
    Uses page.get_text("words") which returns tuples (x0,y0,x1,y1, "word", block_no, line_no, word_no)
    We will keep the bbox of each 'word' token.
    """
    doc = fitz.open(pdf_path)
    pages_tokens = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        words = page.get_text("words")  # list of tuples
        # sort by y (top), then x (left)
        words.sort(key=lambda w: (round(w[1], 1), w[0]))
        tokens = []
        for idx, w in enumerate(words):
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            # Clean token
            token_text = str(word).strip()
            if token_text == "":
                continue
            tokens.append({"text": token_text, "bbox": (x0, y0, x1, y1), "idx": idx})
        pages_tokens.append(tokens)
    doc.close()
    return pages_tokens

def try_parse_first_number(s):
    m = number_regex.search(s)
    if not m:
        return None
    t = m.group(0)
    try:
        if "." in t:
            return float(t.replace(",", ""))
        else:
            return int(t.replace(",", ""))
    except:
        return None

def token_level_diff_and_annotate(old_pdf, new_pdf, out_dir, highlight_opacity=0.5):
    """
    Main worker:
    - read token lists
    - for each page run SequenceMatcher on token texts
    - apply highlights to NEW doc where required
    - build summary rows for insert/delete/replace
    Returns:
      annotated_new_path, summary_rows
    """
    old_tokens_pages = extract_page_tokens(old_pdf)
    new_tokens_pages = extract_page_tokens(new_pdf)
    max_pages = max(len(old_tokens_pages), len(new_tokens_pages))

    old_doc = fitz.open(old_pdf)
    new_doc = fitz.open(new_pdf)

    summary = []  # list of dicts: page, type, old_snippet, new_snippet, old_val, new_val

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
            elif tag == "insert":
                # j1..j2-1 are inserted tokens in new
                for nj in range(j1, j2):
                    tok = new_tokens[nj]
                    # annotate on new_doc page p at tok['bbox']
                    if p < new_doc.page_count:
                        page = new_doc.load_page(p)
                        rect = fitz.Rect(tok["bbox"])
                        # Use rectangle annotation with fill color for visibility
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=GREEN, fill=GREEN)
                        annot.set_opacity(highlight_opacity)
                        annot.update()
                    summary.append({
                        "page": p + 1,
                        "change_type": "insert",
                        "old_snippet": "",
                        "new_snippet": tok["text"],
                        "old_val": "",
                        "new_val": ""
                    })
            elif tag == "delete":
                # i1..i2-1 are deleted tokens from old -> record in summary (do not annotate)
                removed_snippet = " ".join(old_texts[i1:i2])
                summary.append({
                    "page": p + 1,
                    "change_type": "delete",
                    "old_snippet": removed_snippet,
                    "new_snippet": "",
                    "old_val": "",
                    "new_val": ""
                })
            elif tag == "replace":
                # tokens in old[i1:i2] replaced by new[j1:j2]
                # We'll try pairwise: highlight new tokens RED and record old->new in summary.
                len_old = i2 - i1
                len_new = j2 - j1
                pairs = max(len_old, len_new)
                for k in range(pairs):
                    oi = i1 + k if (i1 + k) < i2 else None
                    nj = j1 + k if (j1 + k) < j2 else None
                    old_text = old_tokens[oi]["text"] if oi is not None else ""
                    new_text = new_tokens[nj]["text"] if nj is not None else ""
                    # annotate new token (if exists) in RED
                    if (nj is not None) and (p < new_doc.page_count):
                        page = new_doc.load_page(p)
                        rect = fitz.Rect(new_tokens[nj]["bbox"])
                        annot = page.add_rect_annot(rect)
                        annot.set_colors(stroke=RED, fill=RED)
                        annot.set_opacity(highlight_opacity)
                        annot.update()
                    # attempt numeric detection
                    old_num = try_parse_first_number(old_text)
                    new_num = try_parse_first_number(new_text)
                    row = {
                        "page": p + 1,
                        "change_type": "replace",
                        "old_snippet": old_text,
                        "new_snippet": new_text,
                        "old_val": old_num if old_num is not None else "",
                        "new_val": new_num if new_num is not None else ""
                    }
                    summary.append(row)

    # Save annotated new PDF
    annotated_new_path = os.path.join(out_dir, "annotated_new_highlights.pdf")
    new_doc.save(annotated_new_path, garbage=4, deflate=True)
    old_doc.close()
    new_doc.close()

    return annotated_new_path, summary

def build_summary_pdf(summary_rows, out_path, created_by="Ashutosh Nanaware"):
    """
    Create a readable summary panel PDF listing Inserted / Changed / Removed items.
    """
    # Group rows by change_type for a clear presentation (Inserted, Changed, Removed)
    inserted = [r for r in summary_rows if r["change_type"] == "insert"]
    replaced = [r for r in summary_rows if r["change_type"] == "replace"]
    removed = [r for r in summary_rows if r["change_type"] == "delete"]

    rows = []
    headers = ["Page", "Type", "Old (snippet)", "New (snippet)", "Old val", "New val"]
    rows.append(headers)

    # Inserted
    for r in inserted:
        rows.append([r["page"], "Inserted", "", r["new_snippet"], "", ""])
    # Replaced
    for r in replaced:
        rows.append([r["page"], "Changed", r.get("old_snippet", ""), r.get("new_snippet",""), str(r.get("old_val","")), str(r.get("new_val",""))])
    # Removed
    for r in removed:
        rows.append([r["page"], "Removed", r.get("old_snippet",""), "", "", ""])

    # Build PDF with reportlab
    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=12, leftMargin=12, topMargin=12, bottomMargin=12)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("PDF Comparison Summary", styles["Heading1"]))
    elements.append(Paragraph(f"Created by: {created_by}", styles["Normal"]))
    elements.append(Spacer(1, 6))

    # If huge, table will span pages since repeatRows=1
    table = Table(rows, repeatRows=1, colWidths=[40, 70, 180, 180, 60, 60])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    doc.build(elements)

def append_pdf(base_pdf, to_append_pdf, out_path):
    """
    Append to_append_pdf pages to base_pdf and write to out_path (PyMuPDF).
    """
    base = fitz.open(base_pdf)
    app = fitz.open(to_append_pdf)
    base.insert_pdf(app)
    base.save(out_path, garbage=4, deflate=True)
    base.close()
    app.close()

def create_side_by_side_pdf(old_pdf_path, new_pdf_path, out_path):
    """
    Create a side-by-side PDF (old left, new right). Renders pages to PNG and places them.
    """
    old = fitz.open(old_pdf_path)
    new = fitz.open(new_pdf_path)
    out_doc = fitz.open()

    max_pages = max(old.page_count, new.page_count)
    for i in range(max_pages):
        # get pixmaps (render) at 2x for quality
        if i < old.page_count:
            op = old.load_page(i)
            o_pix = op.get_pixmap(matrix=fitz.Matrix(2,2))
            o_img = Image.open(io.BytesIO(o_pix.tobytes("png")))
        else:
            o_img = Image.new("RGB", (800, 1000), (255,255,255))
        if i < new.page_count:
            npg = new.load_page(i)
            n_pix = npg.get_pixmap(matrix=fitz.Matrix(2,2))
            n_img = Image.open(io.BytesIO(n_pix.tobytes("png")))
        else:
            n_img = Image.new("RGB", (800, 1000), (255,255,255))

        # scale both to same height
        target_h = max(o_img.height, n_img.height)
        o_w = int(o_img.width * (target_h / o_img.height))
        n_w = int(n_img.width * (target_h / n_img.height))
        o_resized = o_img.resize((o_w, target_h), Image.LANCZOS)
        n_resized = n_img.resize((n_w, target_h), Image.LANCZOS)

        page_w = o_w + n_w
        page_h = target_h
        page = out_doc.new_page(width=page_w, height=page_h)

        # insert left
        buf = io.BytesIO()
        o_resized.save(buf, format="PNG")
        page.insert_image(fitz.Rect(0, 0, o_w, page_h), stream=buf.getvalue())
        # insert right
        buf = io.BytesIO()
        n_resized.save(buf, format="PNG")
        page.insert_image(fitz.Rect(o_w, 0, o_w + n_w, page_h), stream=buf.getvalue())

    out_doc.save(out_path, garbage=4, deflate=True)
    out_doc.close()
    old.close()
    new.close()

def process_and_annotate_pdfs(old_pdf_path, new_pdf_path, out_dir, highlight_opacity=0.5):
    """
    Full pipeline:
    - token diff & annotate new PDF
    - build summary PDF
    Returns: annotated_new_path, summary_pdf_path, summary_rows
    """
    annotated_new_path, summary_rows = token_level_diff_and_annotate(old_pdf_path, new_pdf_path, out_dir, highlight_opacity=highlight_opacity)

    summary_pdf_path = os.path.join(out_dir, "summary_panel.pdf")
    build_summary_pdf(summary_rows, summary_pdf_path, created_by="Ashutosh Nanaware")

    return annotated_new_path, summary_pdf_path, summary_rows


