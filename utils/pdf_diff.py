import os
import re
from difflib import SequenceMatcher
import traceback
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def extract_words(text):
    return re.findall(r"[\w%&/\\\-\+\.]+|\S", text, flags=re.UNICODE)

def compare_pdfs(old_pdf_path, new_pdf_path, output_dir):
    ensure_dirs([output_dir])
    # open files
    doc_old = fitz.open(old_pdf_path)
    doc_new = fitz.open(new_pdf_path)

    # annotated copies
    ann_old = fitz.open()
    ann_new = fitz.open()
    try:
        for p in range(len(doc_old)):
            ann_old.insert_pdf(doc_old, from_page=p, to_page=p)
        for p in range(len(doc_new)):
            ann_new.insert_pdf(doc_new, from_page=p, to_page=p)
    except Exception:
        # if insert_pdf fails, fallback to original docs as annotated ones
        ann_old = doc_old
        ann_new = doc_new

    summary = []
    max_pages = max(len(doc_old), len(doc_new))

    for i in range(max_pages):
        try:
            old_text = doc_old[i].get_text("text") if i < len(doc_old) else ""
            new_text = doc_new[i].get_text("text") if i < len(doc_new) else ""
            old_words = extract_words(old_text)
            new_words = extract_words(new_text)

            sm = SequenceMatcher(None, old_words, new_words)
            ops = sm.get_opcodes()

            page_changes = {"page": i+1, "inserted":0, "deleted":0, "replaced":0, "inserted_words":[], "deleted_words":[], "replaced_pairs": []}

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
                                rects = new_page.search_for(str(w))
                                for r in rects:
                                    a = new_page.add_rect_annot(r)
                                    a.set_colors(stroke=(0,1,0), fill=(0,1,0))
                                    a.set_opacity(0.25)
                                    a.update()
                            except Exception:
                                # skip highlighting this token
                                continue
                elif tag == "delete":
                    for w in old_words[alo:ahi]:
                        page_changes["deleted"] += 1
                        page_changes["deleted_words"].append(w)
                        if old_page:
                            try:
                                rects = old_page.search_for(str(w))
                                for r in rects:
                                    a = old_page.add_rect_annot(r)
                                    a.set_colors(stroke=(1,0,0), fill=(1,0,0))
                                    a.set_opacity(0.25)
                                    a.update()
                            except Exception:
                                continue
                elif tag == "replace":
                    page_changes["replaced"] += max(ahi-alo, bhi-blo)
                    page_changes["replaced_pairs"].append((old_words[alo:ahi], new_words[blo:bhi]))
                    if old_page:
                        for w in old_words[alo:ahi]:
                            try:
                                rects = old_page.search_for(str(w))
                                for r in rects:
                                    a = old_page.add_rect_annot(r)
                                    a.set_colors(stroke=(1,0,0), fill=(1,0,0))
                                    a.set_opacity(0.25)
                                    a.update()
                            except Exception:
                                continue
                    if new_page:
                        for w in new_words[blo:bhi]:
                            try:
                                rects = new_page.search_for(str(w))
                                for r in rects:
                                    a = new_page.add_rect_annot(r)
                                    a.set_colors(stroke=(0,1,0), fill=(0,1,0))
                                    a.set_opacity(0.25)
                                    a.update()
                            except Exception:
                                continue

            summary.append(page_changes)

        except Exception:
            # capture page-level trace but continue with next page
            tb = traceback.format_exc()
            # write a small per-page error log next to the output dir
            try:
                with open(os.path.join(output_dir, f"page_error_{i+1}.txt"), "w", encoding="utf-8") as fh:
                    fh.write(tb)
            except Exception:
                pass
            continue

    # Save annotated PDFs (safe-save)
    annotated_old = os.path.join(output_dir, "annotated_old.pdf")
    annotated_new = os.path.join(output_dir, "annotated_new.pdf")
    try:
        ann_old.save(annotated_old)
    except Exception:
        try:
            doc_old.save(annotated_old)
        except Exception:
            pass
    try:
        ann_new.save(annotated_new)
    except Exception:
        try:
            doc_new.save(annotated_new)
        except Exception:
            pass

    side_by_side_path = os.path.join(output_dir, "side_by_side.pdf")
    try:
        create_side_by_side(doc_old, doc_new, side_by_side_path)
    except Exception:
        # write side_by_side error file
        with open(os.path.join(output_dir, "side_by_side_error.txt"), "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())

    summary_path = os.path.join(output_dir, "summary.pdf")
    try:
        create_summary_pdf(summary, summary_path)
    except Exception:
        with open(os.path.join(output_dir, "summary_error.txt"), "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())

    merged_path = os.path.join(output_dir, "merged_report.pdf")
    try:
        create_merged_report(summary_path, annotated_new, merged_path)
    except Exception:
        with open(os.path.join(output_dir, "merge_error.txt"), "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())

    return {
        "annotated_old": os.path.basename(annotated_old) if os.path.exists(annotated_old) else None,
        "annotated_new": os.path.basename(annotated_new) if os.path.exists(annotated_new) else None,
        "side_by_side": os.path.basename(side_by_side_path) if os.path.exists(side_by_side_path) else None,
        "summary": os.path.basename(summary_path) if os.path.exists(summary_path) else None,
        "merged_report": os.path.basename(merged_path) if os.path.exists(merged_path) else None
    }

def create_side_by_side(doc_old, doc_new, out_path):
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
    c = canvas.Canvas(out_path, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, h-40, "PDF Comparison Summary")
    y = h-80
    c.setFont("Helvetica", 10)
    for s in summary:
        line = f"Page {s['page']}: +{s['inserted']} inserts, -{s['deleted']} deletes, ~{s['replaced']} replaces"
        c.drawString(30, y, line)
        y -= 14
        if y < 60:
            c.showPage()
            y = h-40
    c.save()

def create_merged_report(summary_pdf, annotated_new_pdf, out_path):
    out = fitz.open()
    if os.path.exists(summary_pdf):
        out.insert_pdf(fitz.open(summary_pdf))
    if os.path.exists(annotated_new_pdf):
        out.insert_pdf(fitz.open(annotated_new_pdf))
    out.save(out_path)

def generate_sample_pdfs(dest_dir):
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
    c2.drawString(40, 710, "Value: MDA 320")
    c2.drawString(40, 690, "Visibility: 1600m")
    c2.drawString(40, 670, "NOTAM: New procedure added")
    c2.save()
    return p1, p2

