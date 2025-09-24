# pdf_diff.py
"""
Robust PDF diff utilities with lazy imports and clear error messages.

Provides:
  - generate_annotated_pdf(old_input, new_input, output_path, created_by="...", highlight_opacity=0.5)
  - generate_side_by_side_pdf(old_input, new_input, output_path, zoom=1.8)

old_input/new_input may be:
  - a filesystem path (str) OR
  - a file-like object with .read() (Streamlit's UploadedFile)

This module avoids importing PyMuPDF (fitz) at module import time to prevent Streamlit start-up crashes
when PyMuPDF isn't installed or incompatible with the Python runtime. Instead it imports inside functions
and raises a clear RuntimeError if the dependency is missing.
"""
from typing import Any, Dict, List, Tuple
import tempfile
import os
import io
import re
from difflib import SequenceMatcher

# light-weight standard helpers only (no heavy third-party imports here)
number_regex = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def _save_input_to_temp(input_obj: Any, target_dir: str, name: str) -> str:
    """
    Accept either a path string or a file-like object. Return a file path.
    """
    if isinstance(input_obj, str):
        # already a path on disk
        if not os.path.exists(input_obj):
            raise FileNotFoundError(f"File path not found: {input_obj}")
        return input_obj
    # expect file-like with .read()
    if hasattr(input_obj, "read"):
        out_path = os.path.join(target_dir, name)
        with open(out_path, "wb") as f:
            # if stream provides bytes directly
            data = input_obj.read()
            # some UploadedFile return bytes; if it's a buffer object, keep as bytes
            if isinstance(data, str):
                data = data.encode("utf-8")
            f.write(data)
        # reset stream pointer if it supports seek
        try:
            input_obj.seek(0)
        except Exception:
            pass
        return out_path
    raise ValueError("Unsupported input type for PDF: provide file path or file-like object")

def _ensure_pdf_deps():
    """
    Lazy import of heavy libs. If import fails, raise a helpful RuntimeError with installation guidance.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        hint = (
            "Missing or incompatible dependency for PDF rendering/annotation (PyMuPDF/Pillow/reportlab).\n"
            "On Streamlit Cloud, make sure your requirements.txt includes a supported PyMuPDF wheel.\n"
            "Recommended: add 'PyMuPDF==1.23.6' (or a stable 1.23.x/1.24.x release) and compatible Pillow/reportlab versions.\n"
            "If your environment uses Python 3.13, PyMuPDF may not yet have wheels for that Python version — try Python 3.11 or 3.10.\n"
            "After updating requirements.txt, commit & push to GitHub to trigger Streamlit rebuild."
        )
        raise RuntimeError(f"{hint}\nOriginal error: {e}") from e

    # return modules (so callers can use them)
    return {
        "fitz": fitz,
        "Image": Image,
        "A4": A4,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Table": Table,
        "TableStyle": TableStyle,
        "Paragraph": Paragraph,
        "Spacer": Spacer,
        "colors": colors,
        "getSampleStyleSheet": getSampleStyleSheet,
    }

def try_parse_first_number(s: str):
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

def generate_annotated_pdf(old_input: Any, new_input: Any, output_path: str,
                           created_by: str = "Ashutosh Nanaware",
                           highlight_opacity: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full pipeline:
      - Accept old_input/new_input (path or file-like)
      - Run token-level diff using PyMuPDF word coordinates
      - Annotate NEW PDF: inserted tokens (green), replaced tokens (red); removed tokens recorded in summary only
      - Append summary page to annotated new PDF and write to output_path
    Returns (output_path, summary_rows)
    """
    # lazy import and helpful error if missing
    deps = _ensure_pdf_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]
    SimpleDocTemplate = deps["SimpleDocTemplate"]
    Table = deps["Table"]
    TableStyle = deps["TableStyle"]
    Paragraph = deps["Paragraph"]
    Spacer = deps["Spacer"]
    colors = deps["colors"]
    getSampleStyleSheet = deps["getSampleStyleSheet"]

    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
    old_path = _save_input_to_temp(old_input, tmpdir, "old.pdf")
    new_path = _save_input_to_temp(new_input, tmpdir, "new.pdf")

    # token extraction using PyMuPDF (per-page 'words')
    def extract_page_tokens(pdf_path: str):
        doc = fitz.open(pdf_path)
        pages = []
        try:
            for p in range(doc.page_count):
                page = doc.load_page(p)
                words = page.get_text("words")  # list of tuples
                if not words:
                    pages.append([])
                    continue
                words.sort(key=lambda w: (w[5], w[6], w[1], w[0]))
                toks = [{"text": str(w[4]).strip(), "bbox": (w[0], w[1], w[2], w[3])} for w in words if str(w[4]).strip() != ""]
                pages.append(toks)
        finally:
            doc.close()
        return pages

    old_tokens_pages = extract_page_tokens(old_path)
    new_tokens_pages = extract_page_tokens(new_path)
    max_pages = max(len(old_tokens_pages), len(new_tokens_pages))

    # open docs for annotation
    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)

    summary_rows: List[Dict[str, Any]] = []

    try:
        for p in range(max_pages):
            old_tokens = old_tokens_pages[p] if p < len(old_tokens_pages) else []
            new_tokens = new_tokens_pages[p] if p < len(new_tokens_pages) else []

            if not old_tokens and not new_tokens:
                # no selectable text on this page
                summary_rows.append({
                    "page": p+1,
                    "change_type": "no_text",
                    "old_snippet": "",
                    "new_snippet": "",
                    "old_val": "",
                    "new_val": "",
                    "note": "No selectable text on this page — consider OCR"
                })
                continue

            old_texts = [t["text"] for t in old_tokens]
            new_texts = [t["text"] for t in new_tokens]

            sm = SequenceMatcher(None, old_texts, new_texts, autojunk=False)

            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                if tag == "insert":
                    # annotate new tokens as GREEN
                    if p < new_doc.page_count:
                        page = new_doc.load_page(p)
                        for nj in range(j1, j2):
                            tok = new_tokens[nj]
                            x0, y0, x1, y1 = tok["bbox"]
                            rect = fitz.Rect(x0-1.2, y0-1.2, x1+1.2, y1+1.2)
                            annot = page.add_rect_annot(rect)
                            annot.set_colors(stroke=GREEN, fill=GREEN)
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
                    summary_rows.append({
                        "page": p+1,
                        "change_type": "insert",
                        "old_snippet": "",
                        "new_snippet": " ".join(new_texts[j1:j2]),
                        "old_val": "",
                        "new_val": ""
                    })
                elif tag == "delete":
                    # record deletions in summary only
                    summary_rows.append({
                        "page": p+1,
                        "change_type": "delete",
                        "old_snippet": " ".join(old_texts[i1:i2]),
                        "new_snippet": "",
                        "old_val": "",
                        "new_val": ""
                    })
                elif tag == "replace":
                    # annotate new tokens as RED
                    if p < new_doc.page_count:
                        page = new_doc.load_page(p)
                        for nj in range(j1, j2):
                            tok = new_tokens[nj]
                            x0, y0, x1, y1 = tok["bbox"]
                            rect = fitz.Rect(x0-1.2, y0-1.2, x1+1.2, y1+1.2)
                            annot = page.add_rect_annot(rect)
                            annot.set_colors(stroke=RED, fill=RED)
                            annot.set_opacity(float(highlight_opacity))
                            annot.update()
                    old_snip = " ".join(old_texts[i1:i2])
                    new_snip = " ".join(new_texts[j1:j2])
                    summary_rows.append({
                        "page": p+1,
                        "change_type": "replace",
                        "old_snippet": old_snip,
                        "new_snippet": new_snip,
                        "old_val": try_parse_first_number(old_snip) or "",
                        "new_val": try_parse_first_number(new_snip) or ""
                    })
                else:
                    # fallback
                    summary_rows.append({
                        "page": p+1,
                        "change_type": tag,
                        "old_snippet": " ".join(old_texts[i1:i2]) if i1 is not None else "",
                        "new_snippet": " ".join(new_texts[j1:j2]) if j1 is not None else "",
                        "old_val": "",
                        "new_val": ""
                    })

        # save annotated new PDF (temporary, then we will append summary)
        annotated_new_tmp = os.path.join(tmpdir, "annotated_new_tmp.pdf")
        new_doc.save(annotated_new_tmp, garbage=4, deflate=True)

        # build summary PDF using reportlab
        sample_styles = getSampleStyleSheet()
        summary_tmp = os.path.join(tmpdir, "summary_panel.pdf")
        # build rows grouped by change_type
        headers = ["Page", "Type", "Old (snippet)", "New (snippet)", "Old val", "New val", "Note"]
        table_rows = [headers]
        for r in summary_rows:
            page = r.get("page")
            ctype = r.get("change_type", "")
            note = r.get("note", "")
            tdisp = "Other"
            if ctype == "insert":
                tdisp = "Inserted"
            elif ctype == "replace":
                tdisp = "Changed"
            elif ctype == "delete":
                tdisp = "Removed"
            elif ctype == "no_text":
                tdisp = "No selectable text"
            table_rows.append([
                page,
                tdisp,
                (r.get("old_snippet") or "")[:240],
                (r.get("new_snippet") or "")[:240],
                str(r.get("old_val") or ""),
                str(r.get("new_val") or ""),
                note
            ])
        # create summary PDF
        doc = SimpleDocTemplate(summary_tmp, pagesize=A4, rightMargin=12, leftMargin=12, topMargin=12, bottomMargin=12)
        elems = []
        elems.append(Paragraph("PDF Comparison Summary", sample_styles["Heading1"]))
        elems.append(Spacer(1, 6))
        elems.append(Paragraph(f"Created by: {created_by}", sample_styles["Normal"]))
        elems.append(Spacer(1, 8))
        from reportlab.platypus import Table as RLTable  # local import to be explicit
        t = RLTable(table_rows, repeatRows=1, colWidths=[40, 70, 180, 180, 60, 60, 80])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DDDDDD")),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        elems.append(t)
        doc.build(elems)

        # append the summary PDF pages to annotated_new_tmp
        base = fitz.open(annotated_new_tmp)
        summary_doc = fitz.open(summary_tmp)
        base.insert_pdf(summary_doc)
        # save final output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        base.save(output_path, garbage=4, deflate=True)
        base.close()
        summary_doc.close()

    finally:
        old_doc.close()
        new_doc.close()

    return output_path, summary_rows

def generate_side_by_side_pdf(old_input: Any, new_input: Any, output_path: str, zoom: float = 1.8) -> str:
    """
    Create side-by-side PDF pages rendering old and new pages as images and placing them left/right.
    Accepts file path or file-like for inputs. Writes to output_path.
    """
    deps = _ensure_pdf_deps()
    fitz = deps["fitz"]
    Image = deps["Image"]

    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
    old_path = _save_input_to_temp(old_input, tmpdir, "old.pdf")
    new_path = _save_input_to_temp(new_input, tmpdir, "new.pdf")

    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)
    out_doc = fitz.open()

    try:
        max_pages = max(old_doc.page_count, new_doc.page_count)
        for i in range(max_pages):
            # render old page
            if i < old_doc.page_count:
                op = old_doc.load_page(i)
                o_pix = op.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                o_img = Image.open(io.BytesIO(o_pix.tobytes("png")))
            else:
                o_img = Image.new("RGB", (800, 1000), (255, 255, 255))

            # render new page
            if i < new_doc.page_count:
                npg = new_doc.load_page(i)
                n_pix = npg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                n_img = Image.open(io.BytesIO(n_pix.tobytes("png")))
            else:
                n_img = Image.new("RGB", (800, 1000), (255, 255, 255))

            target_h = max(o_img.height, n_img.height)
            o_w = int(o_img.width * (target_h / o_img.height))
            n_w = int(n_img.width * (target_h / n_img.height))
            o_resized = o_img.resize((o_w, target_h), Image.LANCZOS)
            n_resized = n_img.resize((n_w, target_h), Image.LANCZOS)

            page_w = o_w + n_w
            page_h = target_h
            page = out_doc.new_page(width=page_w, height=page_h)

            buf = io.BytesIO()
            o_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(0, 0, o_w, page_h), stream=buf.getvalue())

            buf = io.BytesIO()
            n_resized.save(buf, format="PNG")
            page.insert_image(fitz.Rect(o_w, 0, o_w + n_w, page_h), stream=buf.getvalue())

        out_doc.save(output_path, garbage=4, deflate=True)
    finally:
        old_doc.close()
        new_doc.close()
        out_doc.close()

    return output_path




