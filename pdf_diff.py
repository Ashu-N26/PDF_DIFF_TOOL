# pdf_diff.py
"""
pdf_diff.py
Robust PDF diff utilities for:
  - generate_annotated_pdf(old_pdf_path, new_pdf_path, output_path, ...)
  - generate_side_by_side_pdf(old_pdf_path, new_pdf_path, output_path)

Behavior:
  - Inserted tokens -> GREEN highlight on NEW PDF (opacity default 0.5)
  - Changed tokens  -> RED highlight on NEW PDF (opacity default 0.5)
  - Removed tokens  -> NOT highlighted; listed in the summary panel appended to final PDF
  - If a page has no selectable text, record that fact in the summary (ask user to OCR)
"""

from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
from difflib import SequenceMatcher
import re
import os
import io
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image

# Colors (float 0..1 tuples)
GREEN = (60/255.0, 170/255.0, 60/255.0)
RED = (255/255.0, 80/255.0, 80/255.0)

# number regex for numeric detection in summary
number_regex = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def try_parse_first_number(s: str):
    """Return first numeric token as int/float if present, else None."""
    if not s:
        return None
    m = number_regex.search(s)
    if not m:
        return None
    tok = m.group(0).replace(",", "")
    try:
        if "." in tok:
            return float(tok)
        else:
            return int(tok)
    except Exception:
        return None

def extract_page_tokens(pdf_path: str) -> List[List[Dict[str, Any]]]:
    """
    Extract per-page words (tokens) and their bounding boxes.

    Returns:
      pages_tokens: list indexed by page, each is a list of dicts:
         { "text": str, "bbox": (x0, y0, x1, y1) }
    """
    doc = fitz.open(pdf_path)
    pages_tokens = []
    try:
        for p in range(doc.page_count):
            page = doc.load_page(p)
            words = page.get_text("words")  # tuples: x0,y0,x1,y1,word,block,line,wordno
            # If no words found, append empty list
            if not words:
                pages_tokens.append([])
                continue
            # sort by (block, line, x0) to preserve reading order
            words.sort(key=lambda w: (w[5], w[6], w[0], w[1]))
            tokens = []
            for w in words:
                x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                txt = str(word).strip()
                if txt == "":
                    continue
                tokens.append({"text": txt, "bbox": (x0, y0, x1, y1)})
            pages_tokens.append(tokens)
    finally:
        doc.close()
    return pages_tokens

def _expand_bbox(bbox: Tuple[float, float, float, float], expand: float = 1.5):
    """Expand bbox by `expand` points in all directions (keeps floats)."""
    x0, y0, x1, y1 = bbox
    return (x0 - expand, y0 - expand, x1 + expand, y1 + expand)

def _group_opcode_summary(tag: str, old_texts: List[str], new_texts: List[str], i1: int, i2: int, j1: int, j2: int) -> Dict[str, str]:
    """
    Build a concise summary dict for the given opcode ranges.
    Fields: change_type, old_snippet, new_snippet, old_val, new_val
    """
    old_snip = " ".join(old_texts[i1:i2]) if i1 is not None and i2 is not None else ""
    new_snip = " ".join(new_texts[j1:j2]) if j1 is not None and j2 is not None else ""
    old_val = try_parse_first_number(old_snip) if old_snip else ""
    new_val = try_parse_first_number(new_snip) if new_snip else ""
    change_type = ""
    if tag == "insert":
        change_type = "insert"
    elif tag == "delete":
        change_type = "delete"
    elif tag == "replace":
        change_type = "replace"
    else:
        change_type = "unknown"
    return {
        "change_type": change_type,
        "old_snippet": old_snip,
        "new_snippet": new_snip,
        "old_val": old_val if old_val is not None else "",
        "new_val": new_val if new_val is not None else ""
    }

def token_level_diff_and_annotate(old_pdf_path: str, new_pdf_path: str, out_dir: str, highlight_opacity: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Token-level diff and annotate NEW PDF.
    - Highlights NEW tokens for 'insert' (green) and 'replace' (red).
    - Records 'delete' items (removed) in the summary list.
    Returns:
      annotated_new_path, summary_rows
    """
    # Prepare token lists
    old_pages_tokens = extract_page_tokens(old_pdf_path)
    new_pages_tokens = extract_page_tokens(new_pdf_path)
    max_pages = max(len(old_pages_tokens), len(new_pages_tokens))

    # Open docs
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)

    summary_rows: List[Dict[str, Any]] = []

    try:
        for p in range(max_pages):
            old_tokens = old_pages_tokens[p] if p < len(old_pages_tokens) else []
            new_tokens = new_pages_tokens[p] if p < len(new_pages_tokens) else []

            # If both have no tokens: note in summary and skip page diff
            if not old_tokens and not new_tokens:
                summary_rows.append({
                    "page": p + 1,
                    "change_type": "no_text",
                    "old_snippet": "",
                    "new_snippet": "",
                    "old_val": "",
                    "new_val": "",
                    "note": "No selectable text on page — consider OCR if this page contains text"
                })
                continue

            old_texts = [t["text"] for t in old_tokens]
            new_texts = [t["text"] for t in new_tokens]

            sm = SequenceMatcher(None, old_texts, new_texts, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue

                if tag == "insert":
                    # Annotate new tokens j1..j2-1 (green)
                    if new_doc.page_count > p:
                        page = new_doc.load_page(p)
                        for nj in range(j1, j2):
                            tok = new_tokens[nj]
                            bbox = _expand_bbox(tok["bbox"], expand=1.2)
                            rect = fitz.Rect(bbox)
                            annot = page.add_rect_annot(rect)
                            annot.set_colors(stroke=GREEN, fill=GREEN)
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
                    # Summary: group snippet
                    summary_rows.append({
                        "page": p + 1,
                        **_group_opcode_summary("insert", old_texts, new_texts, i1=None, i2=None, j1=j1, j2=j2)
                    })

                elif tag == "delete":
                    # Do NOT annotate old; record in summary
                    summary_rows.append({
                        "page": p + 1,
                        **_group_opcode_summary("delete", old_texts, new_texts, i1=i1, i2=i2, j1=None, j2=None)
                    })

                elif tag == "replace":
                    # Annotate all new tokens j1..j2-1 in RED
                    if new_doc.page_count > p:
                        page = new_doc.load_page(p)
                        for nj in range(j1, j2):
                            tok = new_tokens[nj]
                            bbox = _expand_bbox(tok["bbox"], expand=1.2)
                            rect = fitz.Rect(bbox)
                            annot = page.add_rect_annot(rect)
                            annot.set_colors(stroke=RED, fill=RED)
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
                    # Summary: show grouped old->new snippet and numbers if detected
                    summary_rows.append({
                        "page": p + 1,
                        **_group_opcode_summary("replace", old_texts, new_texts, i1=i1, i2=i2, j1=j1, j2=j2)
                    })
                else:
                    # safety
                    summary_rows.append({
                        "page": p + 1,
                        "change_type": tag,
                        "old_snippet": " ".join(old_texts[i1:i2]) if i1 is not None else "",
                        "new_snippet": " ".join(new_texts[j1:j2]) if j1 is not None else "",
                        "old_val": "",
                        "new_val": ""
                    })

        # Save annotated NEW PDF to a temp file
        annotated_new_path = os.path.join(out_dir, "annotated_new_highlights.pdf")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(annotated_new_path), exist_ok=True)
        new_doc.save(annotated_new_path, garbage=4, deflate=True)
    finally:
        old_doc.close()
        new_doc.close()

    return annotated_new_path, summary_rows

def build_summary_pdf(summary_rows: List[Dict[str, Any]], out_path: str, created_by: str = "Ashutosh Nanaware"):
    """
    Build a summary panel PDF from summary_rows and save to out_path.
    summary_rows expected fields:
      page, change_type, old_snippet, new_snippet, old_val, new_val, note (optional)
    """
    # Prepare rows grouped: Inserted, Changed, Removed, No-text
    headers = ["Page", "Type", "Old (snippet)", "New (snippet)", "Old val", "New val", "Note"]
    table_rows = [headers]

    for r in summary_rows:
        ctype = r.get("change_type", "")
        page = r.get("page", "")
        old_snip = (r.get("old_snippet") or "")[:240]
        new_snip = (r.get("new_snippet") or "")[:240]
        old_val = str(r.get("old_val", "") or "")
        new_val = str(r.get("new_val", "") or "")
        note = r.get("note", "")

        # Display friendly type names
        t_display = ""
        if ctype == "insert":
            t_display = "Inserted"
        elif ctype == "replace":
            t_display = "Changed"
        elif ctype == "delete":
            t_display = "Removed"
        elif ctype == "no_text":
            t_display = "No selectable text"
        else:
            t_display = ctype or "Other"

        table_rows.append([page, t_display, old_snip, new_snip, old_val, new_val, note])

    # Render PDF
    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=12, leftMargin=12, topMargin=12, bottomMargin=12)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("PDF Comparison Summary", styles["Heading1"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Created by: {created_by}", styles["Normal"]))
    elements.append(Spacer(1, 8))

    # Table
    col_widths = [40, 70, 180, 180, 60, 60, 80]
    t = Table(table_rows, repeatRows=1, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DDDDDD")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (0,-1), "CENTER")
    ]))
    elements.append(t)
    doc.build(elements)

def append_pdf(base_pdf_path: str, to_append_pdf_path: str, out_path: str):
    """
    Append to_append_pdf pages to base_pdf and write as out_path (using PyMuPDF).
    """
    base = fitz.open(base_pdf_path)
    app = fitz.open(to_append_pdf_path)
    try:
        base.insert_pdf(app)
        base.save(out_path, garbage=4, deflate=True)
    finally:
        base.close()
        app.close()

def generate_annotated_pdf(old_pdf_path: str, new_pdf_path: str, output_path: str, created_by: str = "Ashutosh Nanaware", highlight_opacity: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full pipeline to create annotated NEW PDF (with highlights) and append a summary panel.
    Writes final file to `output_path`.

    Returns (output_path, summary_rows).
    """
    # Create temp directory for intermediate files
    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
    try:
        # 1) token diff & annotate new PDF
        annotated_new_temp, summary_rows = token_level_diff_and_annotate(old_pdf_path, new_pdf_path, tmpdir, highlight_opacity=highlight_opacity)

        # 2) build summary PDF
        summary_temp = os.path.join(tmpdir, "summary_panel.pdf")
        build_summary_pdf(summary_rows, summary_temp, created_by=created_by)

        # 3) append summary to annotated new PDF -> final output
        append_pdf(annotated_new_temp, summary_temp, output_path)

    except Exception as e:
        # Clean up temp files on error and re-raise with context
        raise RuntimeError(f"Failed to generate annotated PDF: {e}") from e
    finally:
        # try to remove tempdir but ignore errors
        try:
            # don't aggressively delete to allow debugging on remote; remove files individually if exists
            pass
        except Exception:
            pass

    return output_path, summary_rows

def generate_side_by_side_pdf(old_pdf_path: str, new_pdf_path: str, output_path: str, zoom: float = 1.8):
    """
    Create a side-by-side PDF (old on left, new on right).
    Renders each page to image and places them on a wide page.

    Parameters:
      zoom: render zoom factor (1.0 = 72dpi; >1 improves quality)
    """
    old_doc = fitz.open(old_pdf_path)
    new_doc = fitz.open(new_pdf_path)
    out_doc = fitz.open()

    try:
        max_pages = max(old_doc.page_count, new_doc.page_count)
        for i in range(max_pages):
            # Render old page
            if i < old_doc.page_count:
                opage = old_doc.load_page(i)
                mp = fitz.Matrix(zoom, zoom)
                o_pix = opage.get_pixmap(matrix=mp, alpha=False)
                o_img = Image.open(io.BytesIO(o_pix.tobytes("png")))
            else:
                # blank placeholder
                o_img = Image.new("RGB", (800, 1000), (255, 255, 255))

            # Render new page
            if i < new_doc.page_count:
                npage = new_doc.load_page(i)
                mp = fitz.Matrix(zoom, zoom)
                n_pix = npage.get_pixmap(matrix=mp, alpha=False)
                n_img = Image.open(io.BytesIO(n_pix.tobytes("png")))
            else:
                n_img = Image.new("RGB", (800, 1000), (255, 255, 255))

            # Normalize heights
            target_h = max(o_img.height, n_img.height)
            o_w = int(o_img.width * (target_h / o_img.height))
            n_w = int(n_img.width * (target_h / n_img.height))
            o_resized = o_img.resize((o_w, target_h), Image.LANCZOS)
            n_resized = n_img.resize((n_w, target_h), Image.LANCZOS)

            page_w = o_w + n_w
            page_h = target_h
            page = out_doc.new_page(width=page_w, height=page_h)

            # insert left image
            buf = io.BytesIO()
            o_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(0, 0, o_w, page_h), stream=buf.getvalue())

            # insert right image
            buf = io.BytesIO()
            n_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(o_w, 0, o_w + n_w, page_h), stream=buf.getvalue())

        out_doc.save(output_path, garbage=4, deflate=True)
    finally:
        old_doc.close()
        new_doc.close()
        out_doc.close()

    return output_path

# Optional helper for quick local testing:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python pdf_diff.py old.pdf new.pdf output_annotated.pdf")
        sys.exit(1)
    oldp, newp, outp = sys.argv[1:4]
    print("Generating annotated PDF...")
    generate_annotated_pdf(oldp, newp, outp)
    print("Done. Output:", outp)



