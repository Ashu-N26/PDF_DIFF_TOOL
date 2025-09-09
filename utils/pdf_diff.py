import os
import io
import re
import fitz  # PyMuPDF
import difflib
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# --- helpers
def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_sample_pdfs(old_path, new_path):
    """
    Create very small sample PDFs (reportlab) if not present.
    """
    if not os.path.exists(old_path):
        c = canvas.Canvas(old_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(50, 700, "OLD CHARTS - SAMPLE")
        c.drawString(50, 680, "MDA  300  VIS  800m   RVR  1200")
        c.drawString(50, 640, "Waypoint ABC  Alt 5000")
        c.save()
    if not os.path.exists(new_path):
        c = canvas.Canvas(new_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(50, 700, "NEW CHARTS - SAMPLE")
        c.drawString(50, 680, "MDA  280  VIS  800m   RVR  1200")  # changed value
        c.drawString(50, 640, "Waypoint ABC  Alt 5200")  # changed value
        c.save()

# color helpers (RGBA)
RED_FILL = (220, 40, 40, 128)
GREEN_FILL = (20, 160, 60, 128)

def _page_to_image(page, scale=2.0):
    """Render page to PIL image. Returns PIL.Image (RGB) and pixel scale (px per point)."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    px_per_pt = pix.width / page.rect.width
    return img, px_per_pt

def _words_by_coords(page):
    """
    Returns list of dicts: [{'text': w, 'rect': (x0,y0,x1,y1)}, ...]
    coordinates in PDF points (origin top-left).
    """
    words = page.get_text("words")  # list of tuples
    res = []
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        res.append({"text": text, "rect": (x0, y0, x1, y1)})
    return res

def _collect_rects_for_chunk(page, chunk_text, used_indices, px_per_pt, scale=1.0):
    """
    Try to find bounding rects on `page` for chunk_text.
    1) Try page.search_for (phrase search)
    2) Fallback to scanning words and matching tokens
    Returns list of rectangles in pixel coords [(x0,y0,x1,y1), ...]
    """
    rects = []
    if not chunk_text or not page:
        return rects

    phrase = chunk_text.strip()
    if len(phrase) > 1:
        try:
            hits = page.search_for(phrase, hit_max=32)
            for r in hits:
                rects.append((r.x0*px_per_pt, r.y0*px_per_pt, r.x1*px_per_pt, r.y1*px_per_pt))
            if rects:
                return rects
        except Exception:
            pass

    # fallback: word scanning
    words = page.get_text("words")  # (x0,y0,x1,y1, word, blockno, line_no, wordno)
    tokens = re.findall(r"\w+[\w\-/\.]*", phrase)
    if not tokens:
        return rects

    # map token -> word entries indices
    for t in tokens:
        t_low = t.lower()
        for i,w in enumerate(words):
            if i in used_indices: 
                continue
            word_txt = str(w[4])
            if word_txt.lower() == t_low:
                used_indices.add(i)
                rects.append((w[0]*px_per_pt, w[1]*px_per_pt, w[2]*px_per_pt, w[3]*px_per_pt))
                break
    return rects

def _draw_overlay(img, rects, color):
    """Draw semi-transparent rectangles onto a copy of img."""
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    for (x0,y0,x1,y1) in rects:
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=None)
    composed = Image.alpha_composite(img.convert("RGBA"), overlay)
    return composed.convert("RGB")

def _save_images_as_pdf(images, out_path, dpi=150):
    if not images:
        return None
    # PIL can save multi-page PDF
    first, rest = images[0], images[1:]
    first.save(out_path, "PDF", resolution=dpi, save_all=True, append_images=rest)

# --- main compare routine
def compare_pdfs(old_path, new_path, out_dir, prefix=""):
    """
    Compare two PDFs and produce:
      - annotated_old.pdf
      - annotated_new.pdf
      - merged_report.pdf  (summary + annotated_new)
      - side_by_side.pdf
    Returns paths dict.
    """
    ensure_dirs([out_dir])
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    prefix = prefix or f"{ts}_"

    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)
    page_count = max(len(old_doc), len(new_doc))

    annotated_old_images = []
    annotated_new_images = []

    summary_entries = []  # per page summary
    global_changes = {"added":0, "removed":0, "changed":0}

    # We'll track used indices for robust mapping
    for p in range(page_count):
        old_page = old_doc.load_page(p) if p < len(old_doc) else None
        new_page = new_doc.load_page(p) if p < len(new_doc) else None

        old_text = old_page.get_text("text") if old_page else ""
        new_text = new_page.get_text("text") if new_page else ""

        old_lines = [ln.strip() for ln in old_text.splitlines() if ln.strip()]
        new_lines = [ln.strip() for ln in new_text.splitlines() if ln.strip()]

        # compute line-level opcodes
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        opcodes = matcher.get_opcodes()

        # prepare used index sets and rect lists
        old_used = set()
        new_used = set()
        old_rects = []
        new_rects = []
        page_summary = {"page": p+1, "added":0, "removed":0, "changed":0}

        # For each opcode, collect mapped rects for changed/added/removed chunks
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            old_chunk = " ".join(old_lines[i1:i2]) if i1 < i2 else ""
            new_chunk = " ".join(new_lines[j1:j2]) if j1 < j2 else ""
            if tag in ("replace", "delete"):
                # mark old_chunk as removed/changed
                found = []
                if old_page and old_chunk:
                    found = _collect_rects_for_chunk(old_page, old_chunk, old_used, px_per_pt=_page_to_image(old_page)[1])
                if not found and old_page:
                    # fallback: highlight blocks from page.get_text("blocks")
                    blocks = old_page.get_text("blocks")
                    for b in blocks:
                        txt = str(b[4]).strip()
                        if len(txt)>2 and (txt in old_chunk or old_chunk in txt or len(old_chunk)>0 and txt.lower().find(old_chunk[:20].lower())!=-1):
                            r = (b[0]*_page_to_image(old_page)[1], b[1]*_page_to_image(old_page)[1],
                                 b[2]*_page_to_image(old_page)[1], b[3]*_page_to_image(old_page)[1])
                            found.append(r)
                if found:
                    old_rects.extend(found)
                    page_summary["removed"] += 1
                    global_changes["removed"] += 1
                else:
                    # no mapping -> small full-block fallback (do not mark whole page)
                    # we mark nothing but register as changed
                    page_summary["changed"] += 1
                    global_changes["changed"] += 1

            if tag in ("replace", "insert"):
                # mark new_chunk as added/changed
                found = []
                if new_page and new_chunk:
                    found = _collect_rects_for_chunk(new_page, new_chunk, new_used, px_per_pt=_page_to_image(new_page)[1])
                if not found and new_page:
                    blocks = new_page.get_text("blocks")
                    for b in blocks:
                        txt = str(b[4]).strip()
                        if len(txt)>2 and (txt in new_chunk or new_chunk in txt or len(new_chunk)>0 and txt.lower().find(new_chunk[:20].lower())!=-1):
                            r = (b[0]*_page_to_image(new_page)[1], b[1]*_page_to_image(new_page)[1],
                                 b[2]*_page_to_image(new_page)[1], b[3]*_page_to_image(new_page)[1])
                            found.append(r)
                if found:
                    new_rects.extend(found)
                    page_summary["added"] += 1
                    global_changes["added"] += 1
                else:
                    page_summary["changed"] += 1
                    global_changes["changed"] += 1

        # render old and new page images and draw rects
        scale = 2.0
        if old_page:
            img_old, px_per_pt_old = _page_to_image(old_page, scale=scale)
            if old_rects:
                composed = _draw_overlay(img_old, old_rects, RED_FILL)
            else:
                composed = img_old
            annotated_old_images.append(composed)

        if new_page:
            img_new, px_per_pt_new = _page_to_image(new_page, scale=scale)
            if new_rects:
                composed = _draw_overlay(img_new, new_rects, GREEN_FILL)
            else:
                composed = img_new
            annotated_new_images.append(composed)

        summary_entries.append(page_summary)

    # Save annotated PDFs
    annotated_old_path = os.path.join(out_dir, f"{prefix}annotated_old.pdf")
    annotated_new_path = os.path.join(out_dir, f"{prefix}annotated_new.pdf")
    side_by_side_path = os.path.join(out_dir, f"{prefix}side_by_side.pdf")
    merged_path = os.path.join(out_dir, f"{prefix}merged_report.pdf")
    summary_path = os.path.join(out_dir, f"{prefix}summary.txt")

    if annotated_old_images:
        _save_images_as_pdf(annotated_old_images, annotated_old_path)
    else:
        annotated_old_path = None
    if annotated_new_images:
        _save_images_as_pdf(annotated_new_images, annotated_new_path)
    else:
        annotated_new_path = None

    # side-by-side: combine page images horizontally
    side_images = []
    max_pages = max(len(annotated_old_images), len(annotated_new_images))
    for i in range(max_pages):
        left = annotated_old_images[i] if i < len(annotated_old_images) else Image.new("RGB", (800, 1000), (255,255,255))
        right = annotated_new_images[i] if i < len(annotated_new_images) else Image.new("RGB", (800, 1000), (255,255,255))
        # normalize heights
        h = max(left.height, right.height)
        def fit_h(img, h):
            if img.height == h:
                return img
            w = int(img.width * (h / img.height))
            return img.resize((w, h))
        L = fit_h(left, h)
        R = fit_h(right, h)
        combined = Image.new("RGB", (L.width + R.width, h), (255,255,255))
        combined.paste(L, (0,0))
        combined.paste(R, (L.width, 0))
        side_images.append(combined)
    if side_images:
        _save_images_as_pdf(side_images, side_by_side_path)

    # merged_report: first page = summary (reportlab), then append annotated_new pages if exist
    # create single-page summary PDF
    summary_pdf_path = os.path.join(out_dir, f"{prefix}summary.pdf")
    c = canvas.Canvas(summary_pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 750, "PDF DIFF REPORT")
    c.setFont("Helvetica", 11)
    c.drawString(40, 730, f"Generated: {datetime.utcnow().isoformat()} UTC")
    c.drawString(40, 710, f"Old: {os.path.basename(old_path)}")
    c.drawString(40, 695, f"New: {os.path.basename(new_path)}")
    c.drawString(40, 675, f"Totals — Added: {global_changes['added']}  Removed: {global_changes['removed']}  Changed: {global_changes['changed']}")
    y = 640
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Page-wise summary:")
    y -= 18
    c.setFont("Helvetica", 10)
    for ent in summary_entries:
        c.drawString(45, y, f"Page {ent['page']}: +{ent['added']}  -{ent['removed']}  *{ent['changed']}")
        y -= 14
        if y < 80:
            c.showPage()
            y = 740
    c.save()

    # merge summary + annotated_new into merged_report using PyMuPDF (fast)
    merged_doc = fitz.open()
    merged_doc.insert_pdf(fitz.open(summary_pdf_path))
    if annotated_new_path and os.path.exists(annotated_new_path):
        merged_doc.insert_pdf(fitz.open(annotated_new_path))
    elif annotated_old_path and os.path.exists(annotated_old_path):
        merged_doc.insert_pdf(fitz.open(annotated_old_path))
    merged_doc.save(merged_path)
    merged_doc.close()

    # write a small textual summary
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
        fh.write(f"Old: {old_path}\nNew: {new_path}\n\n")
        fh.write(f"Totals — Added: {global_changes['added']}  Removed: {global_changes['removed']}  Changed: {global_changes['changed']}\n\n")
        for ent in summary_entries:
            fh.write(f"Page {ent['page']}: +{ent['added']}  -{ent['removed']}  *{ent['changed']}\n")

    outputs = {
        "annotated_old": os.path.basename(annotated_old_path) if annotated_old_path else None,
        "annotated_new": os.path.basename(annotated_new_path) if annotated_new_path else None,
        "merged": os.path.basename(merged_path),
        "side_by_side": os.path.basename(side_by_side_path),
        "summary": os.path.basename(summary_path),
    }
    return outputs




