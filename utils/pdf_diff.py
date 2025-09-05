# Minimal PDF diff utilities using PyMuPDF (fitz) and reportlab for PDF creation.
# See README for usage and notes.

import fitz  # PyMuPDF
import os, re
from difflib import SequenceMatcher
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def extract_words(text):
    # split words but keep some punctuation as part of tokens if needed
    return re.findall(r"[\w%&/\\\-\+\.]+|\S", text, flags=re.UNICODE)

def compare_pdfs(old_pdf_path, new_pdf_path, output_dir):
    """
    Compare two PDFs and generate:
      - annotated_old.pdf (red highlights for deletions/changes)
      - annotated_new.pdf (green highlights for insertions/changes)
      - side_by_side.pdf (each page stitched left=old, right=new)
      - summary.pdf (simple summary)
    Returns dict of output filenames.
    """
    ensure_dirs([output_dir])

    doc_old = fitz.open(old_pdf_path)
    doc_new = fitz.open(new_pdf_path)

    # Make copies for annotations
    ann_old = fitz.open()  # will copy pages
    ann_new = fitz.open()

    # duplicate pages into ann_old / ann_new so we can annotate safely
    for p in range(len(doc_old)):
        ann_old.insert_pdf(doc_old, from_page=p, to_page=p)
    for p in range(len(doc_new)):
        ann_new.insert_pdf(doc_new, from_page=p, to_page=p)

    summary = []
    max_pages = max(len(doc_old), len(doc_new))

    for i in range(max_pages):
        old_text = doc_old[i].get_text("text") if i < len(doc_old) else ""
        new_text = doc_new[i].get_text("text") if i < len(doc_new) else ""

        old_words = extract_words(old_text)
        new_words = extract_words(new_text)

        sm = SequenceMatcher(None, old_words, new_words)
        ops = sm.get_opcodes()

        page_changes = {"page": i+1, "inserted":0, "deleted":0, "replaced":0, "inserted_words":[], "deleted_words":[], "replaced_pairs": []}

        # Annotate changes using rectangles located by searching the tokens on pages
        old_page = ann_old[i] if i < ann_old.page_count else None
        new_page = ann_new[i] if i < ann_new.page_count else None

        for tag, alo, ahi, blo, bhi in ops:
            if tag == "equal":
                continue
            if tag == "insert":
                for w in new_words[blo:bhi]:
                    page_changes["inserted"] += 1
                    page_changes["inserted_words"].append(w)
                    if new_page:
                        try:
                            rects = new_page.search_for(str(w), quads=False)
                            for r in rects:
                                annot = new_page.add_rect_annot(r)
                                annot.set_colors(stroke=(0,1,0), fill=(0,1,0))
                                annot.set_opacity(0.25)
                                annot.update()
                        except Exception:
                            pass
            elif tag == "delete":
                for w in old_words[alo:ahi]:
                    page_changes["deleted"] += 1
                    page_changes["deleted_words"].append(w)
                    if old_page:
                        try:
                            rects = old_page.search_for(str(w), quads=False)
                            for r in rects:
                                annot = old_page.add_rect_annot(r)
                                annot.set_colors(stroke=(1,0,0), fill=(1,0,0))
                                annot.set_opacity(0.25)
                                annot.update()
                        except Exception:
                            pass
            elif tag == "replace":
                page_changes["replaced"] += max(ahi-alo, bhi-blo)
                page_changes["replaced_pairs"].append((old_words[alo:ahi], new_words[blo:bhi]))
                if old_page:
                    for w in old_words[alo:ahi]:
                        try:
                            rects = old_page.search_for(str(w), quads=False)
                            for r in rects:
                                annot = old_page.add_rect_annot(r)
                                annot.set_colors(stroke=(1,0,0), fill=(1,0,0))
                                annot.set_opacity(0.25)
                                annot.update()
                        except Exception:
                            pass
                if new_page:
                    for w in new_words[blo:bhi]:
                        try:
                            rects = new_page.search_for(str(w), quads=False)
                            for r in rects:
                                annot = new_page.add_rect_annot(r)
                                annot.set_colors(stroke=(0,1,0), fill=(0,1,0))
                                annot.set_opacity(0.25)
                                annot.update()
                        except Exception:
                            pass

        summary.append(page_changes)

    # Save annotated PDFs
    annotated_old = os.path.join(output_dir, "annotated_old.pdf")
    annotated_new = os.path.join(output_dir, "annotated_new.pdf")
    ann_old.save(annotated_old)
    ann_new.save(annotated_new)

    # Generate side-by-side PDF
    side_by_side_path = os.path.join(output_dir, "side_by_side.pdf")
    create_side_by_side(doc_old, doc_new, side_by_side_path)

    # Generate summary PDF (simple)
    summary_path = os.path.join(output_dir, "summary.pdf")
    create_summary_pdf(summary, summary_path)

    # merged report: put summary page followed by annotated_new pages
    merged_path = os.path.join(output_dir, "merged_report.pdf")
    create_merged_report(summary_path, annotated_new, merged_path)

    return {
        "annotated_old": os.path.basename(annotated_old),
        "annotated_new": os.path.basename(annotated_new),
        "side_by_side": os.path.basename(side_by_side_path),
        "summary": os.path.basename(summary_path),
        "merged_report": os.path.basename(merged_path)
    }

def create_side_by_side(doc_old, doc_new, out_path):
    # Create a new PDF where each page is wide and old page on left, new on right.
    out = fitz.open()
    n = max(doc_old.page_count, doc_new.page_count)
    for i in range(n):
        pix_old = doc_old[i].get_pixmap(matrix=fitz.Matrix(1,1)) if i < doc_old.page_count else None
        pix_new = doc_new[i].get_pixmap(matrix=fitz.Matrix(1,1)) if i < doc_new.page_count else None

        w_old = pix_old.width if pix_old else 595
        h_old = pix_old.height if pix_old else 842
        w_new = pix_new.width if pix_new else 595
        h_new = pix_new.height if pix_new else 842

        W = w_old + w_new
        H = max(h_old, h_new)
        page = out.new_page(width=W, height=H)
        if pix_old:
            rect_old = fitz.Rect(0, 0, w_old, h_old)
            page.insert_image(rect_old, stream=pix_old.tobytes("png"))
        if pix_new:
            rect_new = fitz.Rect(w_old, 0, w_old + w_new, h_new)
            page.insert_image(rect_new, stream=pix_new.tobytes("png"))

    out.save(out_path)

def create_summary_pdf(summary, out_path):
    # simple textual summary using reportlab
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height-40, "PDF Comparison Summary")
    y = height-80
    c.setFont("Helvetica", 10)
    for s in summary:
        line = f"Page {s['page']}: +{s['inserted']} inserts, -{s['deleted']} deletes, ~{s['replaced']} replaces"
        c.drawString(30, y, line)
        y -= 14
        if y < 60:
            c.showPage()
            y = height-40
    c.save()

def create_merged_report(summary_pdf, annotated_new_pdf, out_path):
    # very simple merge: concatenates summary_pdf + annotated_new_pdf
    out = fitz.open()
    if os.path.exists(summary_pdf):
        out.insert_pdf(fitz.open(summary_pdf))
    if os.path.exists(annotated_new_pdf):
        out.insert_pdf(fitz.open(annotated_new_pdf))
    out.save(out_path)

def generate_sample_pdfs(dest_dir):
    # create two small sample PDFs using reportlab; returns paths (old, new)
    p1 = os.path.join(dest_dir, "sample_old.pdf")
    p2 = os.path.join(dest_dir, "sample_new.pdf")
    c = canvas.Canvas(p1, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(40, 750, "Sample Document - OLD")
    c.drawString(40, 730, "This is a sample PDF for testing the PDF diff tool.")
    c.drawString(40, 710, "Value: MDA 300")
    c.drawString(40, 690, "Visibility: 1600m")
    c.save()
    c2 = canvas.Canvas(p2, pagesize=letter)
    c2.setFont("Helvetica", 12)
    c2.drawString(40, 750, "Sample Document - NEW")
    c2.drawString(40, 730, "This is a sample PDF for testing the PDF diff tool.")
    c2.drawString(40, 710, "Value: MDA 320")  # changed value
    c2.drawString(40, 690, "Visibility: 1600m")  # same
    c2.drawString(40, 670, "NOTAM: New procedure added")  # inserted line
    c2.save()
    return p1, p2
