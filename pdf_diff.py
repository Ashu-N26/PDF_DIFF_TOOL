# pdf_diff.py
import os
import re
import fitz            # PyMuPDF
import difflib
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def extract_words_and_text(pdf_path):
    """
    Return:
      pages_words: list per page -> list of dicts {'text': str, 'bbox': (x0,y0,x1,y1)}
      pages_text: list per page -> full page text
    """
    doc = fitz.open(pdf_path)
    pages_words = []
    pages_text = []
    for p in doc:
        words = p.get_text("words")  # list of tuples (x0,y0,x1,y1, "word", block, line, wordno)
        # sort words top->bottom then left->right
        words.sort(key=lambda w: (round(w[1], 2), round(w[0], 2)))
        page_words = []
        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            page_words.append({"text": str(text), "bbox": (x0, y0, x1, y1)})
        pages_words.append(page_words)
        pages_text.append(p.get_text("text"))
    doc.close()
    return pages_words, pages_text

def page_level_diffs(old_words, new_words):
    """
    For one page produce lists of bboxes to highlight in old & new.
    """
    old_seq = [w["text"] for w in old_words]
    new_seq = [w["text"] for w in new_words]
    sm = difflib.SequenceMatcher(None, old_seq, new_seq)
    old_boxes = []
    new_boxes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                if i < len(old_words):
                    old_boxes.append(old_words[i]["bbox"])
        if tag in ("replace", "insert"):
            for j in range(j1, j2):
                if j < len(new_words):
                    new_boxes.append(new_words[j]["bbox"])
    return old_boxes, new_boxes

def merge_overlapping_boxes(boxes, eps=1.0):
    """Union nearby/overlapping boxes to reduce clutter."""
    if not boxes:
        return []
    rects = [fitz.Rect(b) for b in boxes]
    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged = []
    cur = rects[0]
    for r in rects[1:]:
        if cur.intersects(r) or cur.tl.distance_to(r.tl) <= eps:
            cur = cur | r  # union
        else:
            merged.append(cur)
            cur = r
    merged.append(cur)
    return [(r.x0, r.y0, r.x1, r.y1) for r in merged]

def annotate_pdf(original_pdf, highlights_per_page, color_rgb, opacity, out_path, cover_full_page_for_missing=False):
    """
    Annotate by adding filled rect annotations (transparent).
    color_rgb: (r,g,b) each 0..1
    highlights_per_page: list of lists of (x0,y0,x1,y1)
    """
    doc = fitz.open(original_pdf)
    n_pages = doc.page_count
    for pnum in range(len(highlights_per_page)):
        if pnum >= n_pages:
            break
        page = doc.load_page(pnum)
        rects = highlights_per_page[pnum]
        # if asked to cover whole page (for added/removed pages)
        if cover_full_page_for_missing and len(rects) == 0:
            rects = [(page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1)]
        for r in rects:
            rect = fitz.Rect(r)
            try:
                annot = page.add_rect_annot(rect)
                annot.set_colors({"fill": color_rgb})
                annot.set_opacity(opacity)
                annot.update()
            except Exception:
                # fallback: draw a semi-transparent rectangle directly
                shape = page.new_shape()
                shape.draw_rect(rect)
                shape.finish(fill=color_rgb, color=color_rgb, width=0)
                shape.commit()
    doc.save(out_path, garbage=4, deflate=True)
    doc.close()

def build_summary_pdf(old_texts, new_texts, out_path):
    """
    Heuristic minima extraction and simple change log page.
    We'll search for keywords lines and produce a small table.
    """
    KEYWORDS = ["MDA", "DA", "VIS", "RVR", "MINIMA", "CATEGORY", "CAT"]
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 60, "Summary of extracted minima / changed lines")
    c.setFont("Helvetica", 10)
    y = h - 90
    found = False
    # Collect keyword lines (old and new)
    for pidx, (ot, nt) in enumerate(zip(old_texts, new_texts)):
        lines_old = ot.splitlines()
        lines_new = nt.splitlines()
        for L in lines_old + lines_new:
            for kw in KEYWORDS:
                if kw in L.upper():
                    c.drawString(50, y, f"Page {pidx+1}: {L.strip()[:150]}")
                    y -= 14
                    found = True
                    if y < 80:
                        c.showPage()
                        y = h - 60
    if not found:
        c.drawString(50, y, "No obvious minima-like keywords found (heuristic). Review manually.")
    c.showPage()
    c.save()

def render_side_by_side(old_pdf, new_pdf, out_path, image_scale=2):
    """
    Render pages to images and stitch them side-by-side into a single multipage PDF.
    """
    old_doc = fitz.open(old_pdf)
    new_doc = fitz.open(new_pdf)
    max_pages = max(old_doc.page_count, new_doc.page_count)
    images = []
    for i in range(max_pages):
        # render old page (or blank) to PIL image
        if i < old_doc.page_count:
            old_page = old_doc.load_page(i)
            pix_old = old_page.get_pixmap(matrix=fitz.Matrix(image_scale, image_scale), alpha=False)
            img_old = Image.frombytes("RGB", [pix_old.width, pix_old.height], pix_old.samples)
        else:
            img_old = Image.new("RGB", (800, 1000), (255, 255, 255))

        if i < new_doc.page_count:
            new_page = new_doc.load_page(i)
            pix_new = new_page.get_pixmap(matrix=fitz.Matrix(image_scale, image_scale), alpha=False)
            img_new = Image.frombytes("RGB", [pix_new.width, pix_new.height], pix_new.samples)
        else:
            img_new = Image.new("RGB", (800, 1000), (255, 255, 255))

        # scale heights equal
        h = max(img_old.height, img_new.height)
        w_old = int(img_old.width * (h / img_old.height))
        w_new = int(img_new.width * (h / img_new.height))
        img_old = img_old.resize((w_old, h), Image.LANCZOS)
        img_new = img_new.resize((w_new, h), Image.LANCZOS)

        combined = Image.new("RGB", (w_old + w_new, h), (255, 255, 255))
        combined.paste(img_old, (0, 0))
        combined.paste(img_new, (w_old, 0))
        images.append(combined)

    # Save combined images to multi-page PDF
    if images:
        images[0].save(out_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
    old_doc.close()
    new_doc.close()

def compare_pdfs(old_pdf, new_pdf, out_dir, prefix=""):
    """
    High-level function called by app:
      - Extract words and texts
      - Compute per-page word diffs
      - Annotate old & new PDFs (red for old removals/changes, green for new insertions/changes)
      - Generate summary page
      - Create merged output (summary + annotated new)
      - Create side-by-side PDF
    Returns file names (relative to out_dir)
    """
    os.makedirs(out_dir, exist_ok=True)
    pages_old_words, pages_old_text = extract_words_and_text(old_pdf)
    pages_new_words, pages_new_text = extract_words_and_text(new_pdf)

    max_pages = max(len(pages_old_words), len(pages_new_words))
    old_highlights = [[] for _ in range(max_pages)]
    new_highlights = [[] for _ in range(max_pages)]

    for i in range(max_pages):
        old_w = pages_old_words[i] if i < len(pages_old_words) else []
        new_w = pages_new_words[i] if i < len(pages_new_words) else []
        if old_w and new_w:
            o_boxes, n_boxes = page_level_diffs(old_w, new_w)
            old_highlights[i] = merge_overlapping_boxes(o_boxes)
            new_highlights[i] = merge_overlapping_boxes(n_boxes)
        elif old_w and not new_w:
            # page removed -> mark full old page
            old_highlights[i] = []  # will be treated as full-page
        elif new_w and not old_w:
            new_highlights[i] = []  # full-page added

    # Annotate old & new PDFs
    annotated_old = f"{prefix}annotated_old.pdf"
    annotated_new = f"{prefix}annotated_new.pdf"
    annotated_old_path = os.path.join(out_dir, annotated_old)
    annotated_new_path = os.path.join(out_dir, annotated_new)

    # red on old (removed/changed)
    annotate_pdf(old_pdf, old_highlights, color_rgb=(1, 0.2, 0.2), opacity=0.35,
                 out_path=annotated_old_path, cover_full_page_for_missing=True)
    # green on new (inserted/changed)
    annotate_pdf(new_pdf, new_highlights, color_rgb=(0.2, 0.8, 0.3), opacity=0.35,
                 out_path=annotated_new_path, cover_full_page_for_missing=True)

    # Summary page
    summary_pdf = f"{prefix}summary.pdf"
    summary_path = os.path.join(out_dir, summary_pdf)
    build_summary_pdf(pages_old_text, pages_new_text, summary_path)

    # Merged output = summary + annotated_new
    merged_pdf = f"{prefix}merged_report.pdf"
    merged_path = os.path.join(out_dir, merged_pdf)
    merged_doc = fitz.open()
    # insert summary then annotated new
    merged_doc.insert_pdf(fitz.open(summary_path))
    merged_doc.insert_pdf(fitz.open(annotated_new_path))
    merged_doc.save(merged_path, garbage=4, deflate=True)
    merged_doc.close()

    # Side-by-side
    side_by_side_pdf = f"{prefix}side_by_side.pdf"
    side_by_side_path = os.path.join(out_dir, side_by_side_pdf)
    render_side_by_side(annotated_old_path, annotated_new_path, side_by_side_path)

    return annotated_new, side_by_side_pdf, summary_pdf, merged_pdf

