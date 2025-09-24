# app.py
"""
Streamlit frontend for PDF_DIFF tool
- Requires pdf_diff.py in same folder exposing:
    generate_annotated_pdf(old_pdf_path, new_pdf_path, output_path, created_by, highlight_opacity)
    generate_side_by_side_pdf(old_pdf_path, new_pdf_path, output_path, zoom=1.8)
    build_summary_pdf(summary_rows, out_path, created_by)
"""

import streamlit as st
import tempfile
import os
import io
import fitz  # PyMuPDF (used only for rendering previews)
import pandas as pd

# Import the functions from pdf_diff.py (must be in same directory)
from pdf_diff import (
    generate_annotated_pdf,
    generate_side_by_side_pdf,
    build_summary_pdf,
)

st.set_page_config(page_title="PDF_DIFF Tool (Ashutosh Nanaware)", layout="wide")

st.title("📄 PDF_DIFF Tool — Token-level PDF comparison")
st.caption("Created by Ashutosh Nanaware")

st.write(
    "Upload an **Old PDF** and a **New PDF**. "
    "You can preview Side-by-Side and the Annotated PDF (with highlights + summary) before downloading."
)

# Sidebar: uploads and options
with st.sidebar:
    st.header("Uploads & Options")
    old_pdf_file = st.file_uploader("Old PDF (left)", type=["pdf"])
    new_pdf_file = st.file_uploader("New PDF (right)", type=["pdf"])
    st.markdown("---")
    st.write("Highlight settings:")
    highlight_opacity = st.slider("Highlight opacity (applies to generated annotated new PDF)", 0.1, 1.0, 0.5, 0.05)
    preview_zoom = st.slider("Preview render zoom (higher = sharper images, more memory)", 1.0, 3.0, 1.8, 0.1)
    st.markdown("---")
    st.caption("Note: For scanned PDFs (no selectable text) run OCR before upload (e.g., OCRmyPDF).")

# Helper: render a PDF page to PNG bytes using fitz
def render_pdf_page_to_png_bytes(pdf_path: str, page_number: int = 0, zoom: float = 1.8) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page_number = max(0, min(page_number, doc.page_count - 1))
        page = doc.load_page(page_number)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("png")
        return img_bytes
    finally:
        doc.close()

# Helper: get page count
def get_pdf_page_count(pdf_path: str) -> int:
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0

# Main workflow
if old_pdf_file and new_pdf_file:
    st.success("Both PDFs uploaded. Configure options in the sidebar then press Generate.")
    col_generate = st.columns([1, 1])
    if col_generate[0].button("▶️ Generate Previews & Annotated PDF"):
        # Create temp directory to store intermediate files
        tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
        old_path = os.path.join(tmpdir, "old.pdf")
        new_path = os.path.join(tmpdir, "new.pdf")
        annotated_out = os.path.join(tmpdir, "annotated_with_summary.pdf")
        side_by_side_out = os.path.join(tmpdir, "side_by_side.pdf")
        summary_only_out = os.path.join(tmpdir, "summary_panel.pdf")
        summary_csv_out = os.path.join(tmpdir, "summary_table.csv")

        # Write uploaded files to disk
        try:
            with open(old_path, "wb") as f:
                f.write(old_pdf_file.getbuffer())
            with open(new_path, "wb") as f:
                f.write(new_pdf_file.getbuffer())
        except Exception as e:
            st.error(f"Failed to write uploaded files to disk: {e}")
            raise

        # Run generation inside spinner
        try:
            with st.spinner("Running token-level diff & generating annotated PDF..."):
                final_annotated_path, summary_rows = generate_annotated_pdf(
                    old_path,
                    new_path,
                    annotated_out,
                    created_by="Ashutosh Nanaware",
                    highlight_opacity=float(highlight_opacity),
                )
            st.success("Annotated PDF generated.")

            # Build separate summary PDF for download (and CSV)
            try:
                build_summary_pdf(summary_rows, summary_only_out, created_by="Ashutosh Nanaware")
                # build CSV
                try:
                    df_summary = pd.DataFrame(summary_rows)
                    df_summary.to_csv(summary_csv_out, index=False)
                except Exception:
                    df_summary = pd.DataFrame(summary_rows)
            except Exception as e:
                st.warning(f"Could not build separate summary PDF: {e}")
                df_summary = pd.DataFrame(summary_rows)

            # Create side-by-side (rendered) PDF
            with st.spinner("Generating Side-by-Side PDF..."):
                generate_side_by_side_pdf(old_path, final_annotated_path, side_by_side_out, zoom=preview_zoom)
            st.success("Side-by-side PDF generated.")

        except Exception as e:
            st.error(f"Processing failed: {e}")
            # show more context in logs
            st.exception(e)
            raise

        # Previews and download area
        st.markdown("## 🔎 Previews (review before download)")

        # Tabs: Side-by-side preview and Annotated preview
        tab1, tab2 = st.tabs(["Side-by-Side Preview", "Annotated Preview & Summary"])

        # Side-by-side preview
        with tab1:
            st.markdown("**Side-by-side (Old | New)** — use page selector below.")
            ss_page_count = get_pdf_page_count(side_by_side_out)
            if ss_page_count == 0:
                st.info("Side-by-side PDF has 0 pages (unexpected).")
            else:
                page_idx = st.select_slider("Select page to preview", options=list(range(1, ss_page_count + 1)), value=1)
                try:
                    img_bytes = render_pdf_page_to_png_bytes(side_by_side_out, page_idx - 1, zoom=preview_zoom)
                    st.image(img_bytes, use_column_width=True)
                except Exception as e:
                    st.error(f"Failed rendering side-by-side preview: {e}")

            # Download button
            try:
                with open(side_by_side_out, "rb") as f:
                    sb_bytes = f.read()
                st.download_button("⬇️ Download Side-by-Side PDF", sb_bytes, file_name="side_by_side.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Side-by-side file unavailable for download: {e}")

        # Annotated preview and summary
        with tab2:
            st.markdown("**Annotated NEW PDF (highlights shown)** — review pages and the Summary panel below.")
            ann_page_count = get_pdf_page_count(final_annotated_path)
            if ann_page_count == 0:
                st.info("Annotated PDF has 0 pages (unexpected).")
            else:
                # Page selector (allow previewing either document pages OR last pages for summary)
                page_idx2 = st.slider("Select page to preview", 1, ann_page_count, 1)
                try:
                    img_bytes2 = render_pdf_page_to_png_bytes(final_annotated_path, page_idx2 - 1, zoom=preview_zoom)
                    st.image(img_bytes2, use_column_width=True)
                except Exception as e:
                    st.error(f"Failed rendering annotated PDF page: {e}")

            # Download annotated PDF (with summary appended)
            try:
                with open(final_annotated_path, "rb") as f:
                    ann_bytes = f.read()
                st.download_button("⬇️ Download Annotated PDF (with Summary)", ann_bytes, file_name="Annotated_with_Summary.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Annotated PDF unavailable for download: {e}")

            # Show a short human-friendly summary table inline (top 200 rows truncated)
            st.markdown("### Summary (short view)")
            if summary_rows:
                try:
                    df = pd.DataFrame(summary_rows)
                    # Keep columns in friendly order
                    cols = ["page", "change_type", "old_snippet", "new_snippet", "old_val", "new_val", "note"]
                    display_cols = [c for c in cols if c in df.columns]
                    st.dataframe(df[display_cols].head(200))
                except Exception as e:
                    st.warning(f"Could not display summary table: {e}")
            else:
                st.info("No differences detected or summary empty.")

            # Separate Summary PDF & CSV downloads
            try:
                with open(summary_only_out, "rb") as f:
                    summary_pdf_bytes = f.read()
                st.download_button("⬇️ Download Summary Panel (PDF)", summary_pdf_bytes, file_name="summary_panel.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Summary PDF unavailable: {e}")

            try:
                with open(summary_csv_out, "rb") as f:
                    csv_bytes = f.read()
                st.download_button("⬇️ Download Summary (CSV)", csv_bytes, file_name="summary.csv", mime="text/csv")
            except Exception as e:
                # fallback: allow CSV from dataframe
                if 'df' in locals():
                    st.download_button("⬇️ Download Summary (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="summary.csv", mime="text/csv")
                else:
                    st.warning("Summary CSV not available.")

        st.markdown("---")
        st.success("✅ All previews generated. Use the download buttons above to retrieve your files.")
        st.caption("Footer: Created by Ashutosh Nanaware")

else:
    st.info("Upload both Old and New PDF files using the sidebar to enable comparison.")
    st.caption("Tip: For scanned PDFs without selectable text, run OCR (e.g., OCRmyPDF) before uploading.")
