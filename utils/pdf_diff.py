import os
import io
import fitz  # pymupdf
import difflib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def normalize_text(s):
    return " ".join(s.split())

def extract_words_by_page(pdf_path):
    """
    Returns list of pages; each page is list of (word_text, rect)
    rect is (x0,y0,x1,y1) in points (PDF coordinate)
    """
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        words = p.get_text("words")  # list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        page_words = []
        for w in words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            page_words.append((text, (x0, y0, x1, y1)))
        pages.append(page_words)
    doc.close()
    return pages

def page_text_list(page_words):
    return [w[0] for w in page_words]

def find_word_rects(page_obj, word):
    # uses search_for to be safer for multi occurrences
    rects = page_obj.search_for(str(word))
    return rects

def annotate_pdf(base_pdf_path, highlights_by_page, color_rgb=(1,0,0), opacity=0.35, out_path=None, cover_full_page_for_missing=False):
    """
    highlights_by_page: dict[page_index] -> list of (rect, label)
    rect is fitz.Rect or (x0,y0,x1,y1)
    """
    doc = fitz.open(base_pdf_path)
    for p_idx in highlights_by_page:
        if p_idx < 0 or p_idx >= doc.page_count:
            continue
        page = doc[p_idx]
        for rect, label in highlights_by_page[p_idx]:
            r = fitz.Rect(rect)
            # draw semi-transparent rectangle
            annot = page.add_rect_annot(r)
            annot.set_colors(stroke=None, fill=color_rgb)
            annot.set_opacity(opacity)
            annot.update()
            # add a popup text in popup note (so the PDF viewer can see text)
            # We'll also add a small text annotation on top-left as plain text
            # Add optional text label
            if label:
                # small text at top-left of rect
                tx = fitz.Point(r.x0 + 2, r.y0 + 8)
                page.insert_text(tx, str(label), fontsize=6, color=(0,0,0))
    if out_path:
        doc.save(out_path, garbage=4, deflate=True)
        doc.close()
        return out_path
    else:
        out_buf = doc.write()
        doc.close()
        return out_buf

def build_summary_page(summary_items, out_path):
    """
    summary_items: list of dict rows, e.g. [{"page":1,"type":"changed","old":"X","new":"Y"}, ...]
    Creates a one-page PDF with summary table appended.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "PDF Comparison Summary")
    c.setFont("Helvetica", 10)
    y = height - 70
    row_h = 16
    c.drawString(40, y, "Page")
    c.drawString(80, y, "Type")
    c.drawString(140, y, "Old snippet")
    c.drawString(360, y, "New snippet")
    y -= row_h
    for it in summary_items:
        if y < 60:
            c.showPage()
            y = height - 40
        c.drawString(40, y, str(it.get("page","-")))
        c.drawString(80, y, it.get("type",""))
        old = (it.get("old","")[:40] + "...") if len(it.get("old",""))>43 else it.get("old","")
        new = (it.get("new","")[:40] + "...") if len(it.get("new",""))>43 else it.get("new","")
        c.drawString(140, y, old)
        c.drawString(360, y, new)
        y -= row_h
    c.showPage()
    c.save()
    buffer.seek(0)
    with open(out_path, "wb") as f:
        f.write(buffer.read())
    return out_path

def merge_pdfs(paths, out_path):
    out_doc = fitz.open()
    for p in paths:
        doc = fitz.open(p)
        out_doc.insert_pdf(doc)
        doc.close()
    out_doc.save(out_path)
    out_doc.close()
    return out_path

def compare_pdfs(old_pdf_path, new_pdf_path, output_dir, prefix=""):
    """
    Returns: annotated_merged_path, side_by_side_pdf_path, summary_pdf_path, preview_data
    preview_data: JSON-ready dict with page-wise old_text/new_text and diffs for browser preview
    """
    ensure_dirs([output_dir])
    old_pages = extract_words_by_page(old_pdf_path)
    new_pages = extract_words_by_page(new_pdf_path)

    # Basic approach: compare text by page index; if pages differ in count, handle gracefully
    max_pages = max(len(old_pages), len(new_pages))

    doc_old = fitz.open(old_pdf_path)
    doc_new = fitz.open(new_pdf_path)

    # highlights structures: page_index -> list[(rect,label)]
    highlights_old = {}
    highlights_new = {}
    summary_items = []
    preview = {"pages": []}

    for i in range(max_pages):
        old_page_words = old_pages[i] if i < len(old_pages) else []
        new_page_words = new_pages[i] if i < len(new_pages) else []

        old_texts = page_text_list(old_page_words)
        new_texts = page_text_list(new_page_words)

        # perform a sequence matcher on word lists
        sm = difflib.SequenceMatcher(a=old_texts, b=new_texts, autojunk=False)
        old_page_obj = doc_old[i] if i < doc_old.page_count else None
        new_page_obj = doc_new[i] if i < doc_new.page_count else None

        page_preview = {"page_index": i, "old_text": " ".join(old_texts), "new_text": " ".join(new_texts), "diffs": []}

        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag == "equal":
                # nothing to highlight
                continue
            elif tag == "replace":
                # old words removed/changed -> highlight on old in RED
                # new words inserted/changed -> highlight on new in GREEN
                # collect snippet for summary
                old_snip = " ".join(old_texts[a0:a1])
                new_snip = " ".join(new_texts[b0:b1])
                summary_items.append({"page": i+1, "type": "replace", "old": old_snip, "new": new_snip})
                page_preview["diffs"].append({"type": "replace", "old": old_snip, "new": new_snip})

                # find rects for old words
                if old_page_obj:
                    for w in old_texts[a0:a1]:
                        rects = old_page_obj.search_for(str(w))
                        for r in rects:
                            highlights_old.setdefault(i, []).append(((r.x0, r.y0, r.x1, r.y1), f"OLD:{w}"))
                if new_page_obj:
                    for w in new_texts[b0:b1]:
                        rects = new_page_obj.search_for(str(w))
                        for r in rects:
                            highlights_new.setdefault(i, []).append(((r.x0, r.y0, r.x1, r.y1), f"NEW:{w}"))

            elif tag == "delete":
                old_snip = " ".join(old_texts[a0:a1])
                summary_items.append({"page": i+1, "type": "delete", "old": old_snip, "new": ""})
                page_preview["diffs"].append({"type": "delete", "old": old_snip})
                if old_page_obj:
                    for w in old_texts[a0:a1]:
                        rects = old_page_obj.search_for(str(w))
                        for r in rects:
                            highlights_old.setdefault(i, []).append(((r.x0, r.y0, r.x1, r.y1), f"OLD:{w}"))
            elif tag == "insert":
                new_snip = " ".join(new_texts[b0:b1])
                summary_items.append({"page": i+1, "type": "insert", "old": "", "new": new_snip})
                page_preview["diffs"].append({"type": "insert", "new": new_snip})
                if new_page_obj:
                    for w in new_texts[b0:b1]:
                        rects = new_page_obj.search_for(str(w))
                        for r in rects:
                            highlights_new.setdefault(i, []).append(((r.x0, r.y0, r.x1, r.y1), f"NEW:{w}"))

        preview["pages"].append(page_preview)

    # Create annotated PDFs
    annotated_old = os.path.join(output_dir, f"{prefix}annotated_old.pdf")
    annotated_new = os.path.join(output_dir, f"{prefix}annotated_new.pdf")
    # annotate old (red) and new (green)
    annotate_pdf(old_pdf_path, highlights_old, color_rgb=(1,0.2,0.2), opacity=0.45, out_path=annotated_old)
    annotate_pdf(new_pdf_path, highlights_new, color_rgb=(0.2,0.8,0.3), opacity=0.45, out_path=annotated_new)

    # merged output: summary page + annotated new (so analyst sees summary then pages)
    summary_pdf = os.path.join(output_dir, f"{prefix}summary.pdf")
    build_summary_page(summary_items, summary_pdf)

    merged_path = os.path.join(output_dir, f"{prefix}merged_report.pdf")
    merge_pdfs([summary_pdf, annotated_new], merged_path)

    # side-by-side PDF (two-column pages)
    side_by_side_path = os.path.join(output_dir, f"{prefix}side_by_side.pdf")
    create_side_by_side_pdf(old_pdf_path, new_pdf_path, side_by_side_path)

    # We'll return the merged annotated new PDF as 'annotated' for convenience
    return merged_path, side_by_side_path, summary_pdf, preview

def create_side_by_side_pdf(old_pdf, new_pdf, out_path):
    """
    Build side-by-side pages by rasterizing each page and placing left/right
    """
    doc_old = fitz.open(old_pdf)
    doc_new = fitz.open(new_pdf)
    max_pages = max(doc_old.page_count, doc_new.page_count)
    out_doc = fitz.open()

    for i in range(max_pages):
        # render images
        if i < doc_old.page_count:
            pix_old = doc_old[i].get_pixmap(matrix=fitz.Matrix(2,2))
        else:
            # blank
            pix_old = fitz.Pixmap(fitz.csRGB, 1, 1, 255)
        if i < doc_new.page_count:
            pix_new = doc_new[i].get_pixmap(matrix=fitz.Matrix(2,2))
        else:
            pix_new = fitz.Pixmap(fitz.csRGB, 1, 1, 255)

        # create new page wide enough
        w = pix_old.width + pix_new.width
        h = max(pix_old.height, pix_new.height)
        page = out_doc.new_page(width=w, height=h)
        page.insert_image(fitz.Rect(0,0,pix_old.width, pix_old.height), pixmap=pix_old)
        page.insert_image(fitz.Rect(pix_old.width,0, pix_old.width+pix_new.width, pix_new.height), pixmap=pix_new)

    out_doc.save(out_path)
    out_doc.close()
    doc_old.close()
    doc_new.close()
    return out_path

# Optional helper to generate sample PDFs for testing
def generate_sample_pdfs(outdir):
    ensure_dirs([outdir])
    sample_old = os.path.join(outdir, "sample_old.pdf")
    sample_new = os.path.join(outdir, "sample_new.pdf")
    # create minimal PDFs
    c = canvas.Canvas(sample_old, pagesize=letter)
    c.drawString(72, 700, "Instrument Approach Chart - OLD")
    c.drawString(72, 660, "MDA: 600")
    c.drawString(72, 640, "VIS: 5km")
    c.save()
    c = canvas.Canvas(sample_new, pagesize=letter)
    c.drawString(72, 700, "Instrument Approach Chart - NEW")
    c.drawString(72, 660, "MDA: 620")  # changed
    c.drawString(72, 640, "VIS: 5km")
    c.save()
    return sample_old, sample_new



