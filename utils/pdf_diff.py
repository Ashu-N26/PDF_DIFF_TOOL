# utils/pdf_diff.py
import os
import re
import fitz  # PyMuPDF
import difflib
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_sample_pdfs(old_path, new_path):
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
        c.drawString(50, 680, "MDA  280  VIS  800m   RVR  1200")
        c.drawString(50, 640, "Waypoint ABC  Alt 5200")
        c.save()

RED_FILL = (220, 40, 40, 120)
GREEN_FILL = (20, 160, 60, 120)

def _page_to_image(page, scale=1.0):
    """Render page to PIL image. Returns PIL.Image and px_per_pt."""
    mat = fitz.Matrix(scale, scale)
    try:
        pix = page.get_pixmap(matrix=mat, alpha=False)
    except Exception:
        # fallback: try with scale=1.0
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0,1.0), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    px_per_pt = pix.width / page.rect.width
    return img, px_per_pt

def _collect_rects_for_phrase(page, phrase, px_per_pt):
    rects = []
    if not phrase or not page:
        return rects
    try:
        hits = page.search_for(phrase, hit_max=16)
        for r in hits:
            rects.append((r.x0*px_per_pt, r.y0*px_per_pt, r.x1*px_per_pt, r.y1*px_per_pt))
        if rects:
            return rects
    except Exception:
        pass
    # fallback: per-word matching
    try:
        words = page.get_text("words")
        tokens = re.findall(r"\w+[\w\-/\.]*", phrase)
        used = set()
        for t in tokens:
            for i,w in enumerate(words):
                if i in used: continue
                if w[4].strip().lower() == t.lower():
                    used.add(i)
                    rects.append((w[0]*px_per_pt, w[1]*px_per_pt, w[2]*px_per_pt, w[3]*px_per_pt))
                    break
    except Exception:
        pass
    return rects

def _draw_overlay(img, rects, color):
    if not rects:
        return img
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)
    for r in rects:
        draw.rectangle([r[0], r[1], r[2], r[3]], fill=color)
    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    return out.convert("RGB")

def _save_images_as_pdf(images, out_path, dpi=150):
    if not images:
        return
    first, rest = images[0], images[1:]
    first.save(out_path, "PDF", resolution=dpi, save_all=True, append_images=rest)

def compare_pdfs(old_path, new_path, out_dir, prefix="", scale=1.0, page_limit=200):
    """
    Safer compare. scale default 1.0 (reduce memory).
    Returns dict of output filenames (basename).
    """
    ensure_dirs([out_dir])
    prefix = prefix or ""
    old_doc = fitz.open(old_path)
    new_doc = fitz.open(new_path)

    page_count = max(len(old_doc), len(new_doc))
    page_count = min(page_count, page_limit)

    annotated_old_images = []
    annotated_new_images = []
    summary_entries = []
    totals = {"added":0,"removed":0,"changed":0}

    for p in range(page_count):
        old_page = old_doc.load_page(p) if p < len(old_doc) else None
        new_page = new_doc.load_page(p) if p < len(new_doc) else None

        old_text = ""
        new_text = ""
        try:
            if old_page:
                old_text = old_page.get_text("text")
        except Exception:
            old_text = ""
        try:
            if new_page:
                new_text = new_page.get_text("text")
        except Exception:
            new_text = ""

        old_lines = [ln.strip() for ln in old_text.splitlines() if ln.strip()]
        new_lines = [ln.strip() for ln in new_text.splitlines() if ln.strip()]

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        opcodes = matcher.get_opcodes()

        old_rects = []
        new_rects = []
        page_summary = {"page": p+1, "added":0, "removed":0, "changed":0}

        # compute px_per_pt once
        px_old = None
        px_new = None
        if old_page:
            try:
                _, px_old = _page_to_image(old_page, scale=scale)
            except Exception:
                px_old = 1.0
        if new_page:
            try:
                _, px_new = _page_to_image(new_page, scale=scale)
            except Exception:
                px_new = 1.0

        for tag, i1, i2, j1, j2 in opcodes:
            old_chunk = " ".join(old_lines[i1:i2]) if i1 < i2 else ""
            new_chunk = " ".join(new_lines[j1:j2]) if j1 < j2 else ""

            if tag in ("replace", "delete"):
                if old_page and old_chunk:
                    rects = _collect_rects_for_phrase(old_page, old_chunk, px_old or 1.0)
                    if rects:
                        old_rects.extend(rects)
                        page_summary["removed"] += 1
                        totals["removed"] += 1
                    else:
                        page_summary["changed"] += 1
                        totals["changed"] += 1

            if tag in ("replace", "insert"):
                if new_page and new_chunk:
                    rects = _collect_rects_for_phrase(new_page, new_chunk, px_new or 1.0)
                    if rects:
                        new_rects.extend(rects)
                        page_summary["added"] += 1
                        totals["added"] += 1
                    else:
                        page_summary["changed"] += 1
                        totals["changed"] += 1

        try:
            if old_page:
                img_old, _ = _page_to_image(old_page, scale=scale)
                annotated_old_images.append(_draw_overlay(img_old, old_rects, RED_FILL))
            if new_page:
                img_new, _ = _page_to_image(new_page, scale=scale)
                annotated_new_images.append(_draw_overlay(img_new, new_rects, GREEN_FILL))
        except Exception:
            # if rendering fails, just create a blank placeholder (so page counts still match)
            w, h = 800, 1000
            if old_page:
                annotated_old_images.append(Image.new("RGB", (w, h), (255,255,255)))
            if new_page:
                annotated_new_images.append(Image.new("RGB", (w, h), (255,255,255)))

        summary_entries.append(page_summary)

    # file names
    annotated_old_path = os.path.join(out_dir, f"{prefix}annotated_old.pdf")
    annotated_new_path = os.path.join(out_dir, f"{prefix}annotated_new.pdf")
    side_by_side_path = os.path.join(out_dir, f"{prefix}side_by_side.pdf")
    merged_path = os.path.join(out_dir, f"{prefix}merged_report.pdf")
    summary_txt_path = os.path.join(out_dir, f"{prefix}summary.txt")

    if annotated_old_images:
        _save_images_as_pdf(annotated_old_images, annotated_old_path)
    else:
        annotated_old_path = None
    if annotated_new_images:
        _save_images_as_pdf(annotated_new_images, annotated_new_path)
    else:
        annotated_new_path = None

    # side-by-side
    side_images = []
    max_pages = max(len(annotated_old_images), len(annotated_new_images))
    for i in range(max_pages):
        left = annotated_old_images[i] if i < len(annotated_old_images) else Image.new("RGB", (800, 1000), (255,255,255))
        right = annotated_new_images[i] if i < len(annotated_new_images) else Image.new("RGB", (800, 1000), (255,255,255))
        h = max(left.height, right.height)
        def fit_h(img, h):
            if img.height == h: return img
            w = int(img.width * (h / img.height))
            return img.resize((w,h))
        L = fit_h(left, h)
        R = fit_h(right, h)
        combined = Image.new("RGB", (L.width + R.width, h), (255,255,255))
        combined.paste(L, (0,0))
        combined.paste(R, (L.width, 0))
        side_images.append(combined)
    if side_images:
        _save_images_as_pdf(side_images, side_by_side_path)

    # create summary page (reportlab)
    summary_pdf = os.path.join(out_dir, f"{prefix}summary.pdf")
    c = canvas.Canvas(summary_pdf, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 750, "PDF DIFF REPORT")
    c.setFont("Helvetica", 10)
    c.drawString(40, 730, f"Old: {os.path.basename(old_path)}")
    c.drawString(40, 715, f"New: {os.path.basename(new_path)}")
    c.drawString(40, 700, f"Totals — Added: {totals['added']}  Removed: {totals['removed']}  Changed: {totals['changed']}")
    y = 680
    for ent in summary_entries:
        c.drawString(45, y, f"Page {ent['page']}: +{ent['added']}  -{ent['removed']}  *{ent['changed']}")
        y -= 12
        if y < 80:
            c.showPage()
            y = 740
    c.save()

    # merged: summary + annotated_new (or annotated_old if new missing)
    import fitz
    merged_doc = fitz.open()
    merged_doc.insert_pdf(fitz.open(summary_pdf))
    if annotated_new_path and os.path.exists(annotated_new_path):
        merged_doc.insert_pdf(fitz.open(annotated_new_path))
    elif annotated_old_path and os.path.exists(annotated_old_path):
        merged_doc.insert_pdf(fitz.open(annotated_old_path))
    merged_doc.save(merged_path)
    merged_doc.close()

    # write summary text
    with open(summary_txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
        fh.write(f"Old: {old_path}\nNew: {new_path}\n")
        fh.write(f"Totals — Added: {totals['added']}  Removed: {totals['removed']}  Changed: {totals['changed']}\n")
        for ent in summary_entries:
            fh.write(f"Page {ent['page']}: +{ent['added']}  -{ent['removed']}  *{ent['changed']}\n")

    outputs = {
        "annotated_old": os.path.basename(annotated_old_path) if annotated_old_path else None,
        "annotated_new": os.path.basename(annotated_new_path) if annotated_new_path else None,
        "merged": os.path.basename(merged_path),
        "side_by_side": os.path.basename(side_by_side_path) if os.path.exists(side_by_side_path) else None,
        "summary": os.path.basename(summary_txt_path),
    }
    return outputs





