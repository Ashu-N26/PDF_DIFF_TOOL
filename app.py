# app.py
"""
Streamlit web UI for PDF_DIFF_TOOL
Uses the functions provided in pdf_diff.py:
  - process_and_generate(old_input, new_input, workdir=None, highlight_opacity=0.5, created_by=...)
This UI:
  - Upload OLD and NEW PDFs
  - Run compare pipeline (annotated + side-by-side + summary)
  - Preview results and provide downloads
"""

import os
import io
import tempfile
import base64
from typing import List, Dict, Any

import streamlit as st

# Import the real APIs from your pdf_diff.py
# pdf_diff.py must export process_and_generate
try:
    from pdf_diff import process_and_generate
except Exception as e:
    # Re-raise with clearer message for logs (Streamlit shows full traceback)
    raise ImportError(f"Failed to import process_and_generate from pdf_diff.py: {e}") from e

# Try to import fitz for on-page previews (optional but recommended)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# Utilities
def _save_uploaded_tmp(uploaded_file) -> str:
    """Save Streamlit UploadedFile to a temp file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def _pdf_page_to_png_bytes(pdf_path: str, page_idx: int = 0, zoom: float = 1.25) -> bytes:
    """Render a PDF page to PNG bytes using fitz. Returns PNG bytes."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed in the runtime; cannot render preview images.")
    doc = fitz.open(pdf_path)
    if page_idx < 0 or page_idx >= doc.page_count:
        doc.close()
        raise IndexError("page index out of range")
    page = doc.load_page(page_idx)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes

def _display_pdf_inline(pdf_path: str, height: int = 700):
    """Embed a PDF inline using a data URI for browsers that support it."""
    with open(pdf_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    pdf_html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px"></iframe>'
    st.components.v1.html(pdf_html, height=height)

def _summary_to_dataframe(summary_rows: List[Dict[str, Any]]):
    """Convert the summary rows into a simple table structure (list of dicts) for display."""
    import pandas as pd
    if not summary_rows:
        return pd.DataFrame(columns=["page", "change_type", "old_snippet", "new_snippet"])
    df = pd.DataFrame(summary_rows)
    # Standardize columns
    expected = ["page", "change_type", "old_snippet", "new_snippet"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df[expected]

# Streamlit UI
st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")
st.markdown("<h1 style='text-align:center;'>PDF_DIFF TOOL</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:gray;'>Created by <b>Ashutosh Nanaware</b></div>", unsafe_allow_html=True)
st.write("")  # spacing

# Upload area
col1, col2 = st.columns(2)
with col1:
    old_pdf = st.file_uploader("Upload OLD PDF", type=["pdf"], key="old_uploader")
with col2:
    new_pdf = st.file_uploader("Upload NEW PDF", type=["pdf"], key="new_uploader")

st.markdown("---")

if old_pdf and new_pdf:
    st.success("Both PDFs uploaded successfully ✅")
    # Options
    st.sidebar.header("Comparison Options")
    highlight_opacity = st.sidebar.slider("Highlight opacity (%)", 10, 100, 50) / 100.0
    created_by = st.sidebar.text_input("Created by label", value="Ashutosh Nanaware")
    run_button = st.button("🔍 Compare PDFs")

    if run_button:
        # Use spinner during processing
        with st.spinner("Running comparison pipeline — this can take a while for large PDFs..."):
            try:
                # Pass file-like objects directly to process_and_generate (pdf_diff handles file-like)
                annotated_path, side_by_side_path, summary_rows = process_and_generate(
                    old_pdf, new_pdf, highlight_opacity=highlight_opacity, created_by=created_by
                )
            except Exception as e:
                st.error(f"Processing failed: {e}")
                # Print log guidance
                st.info("Check server logs (Render or Streamlit logs) for full traceback.")
                raise

        st.success("Comparison finished ✅")
        st.markdown("### Outputs")

        # Summary panel
        st.subheader("📊 Summary Panel")
        df = _summary_to_dataframe(summary_rows)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Summary CSV", data=csv, file_name="summary.csv", mime="text/csv")

        st.markdown("---")
        # Side-by-side preview (image-based)
        st.subheader("🖼 Side-by-Side Preview")
        if os.path.exists(side_by_side_path):
            try:
                # Show first page preview (image)
                if fitz is not None:
                    png_bytes = _pdf_page_to_png_bytes(side_by_side_path, page_idx=0, zoom=1.25)
                    st.image(png_bytes, use_column_width=True)
                else:
                    st.info("PyMuPDF not available to render inline preview; download the Side-by-Side PDF instead.")
                with open(side_by_side_path, "rb") as f:
                    sb_bytes = f.read()
                st.download_button("⬇️ Download Side-by-Side PDF", data=sb_bytes, file_name="side_by_side.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Could not render side-by-side preview: {e}")
                # Offer download anyway
                try:
                    with open(side_by_side_path, "rb") as f:
                        st.download_button("⬇️ Download Side-by-Side PDF", data=f.read(), file_name="side_by_side.pdf", mime="application/pdf")
                except Exception:
                    st.error("Side-by-side PDF not found for download.")
        else:
            st.warning("Side-by-side PDF was not generated.")

        st.markdown("---")
        # Annotated PDF preview and download
        st.subheader("🖍 Annotated PDF (NEW PDF with highlights + Summary panel appended)")
        if os.path.exists(annotated_path):
            try:
                # Render first page preview of annotated PDF
                if fitz is not None:
                    png_bytes = _pdf_page_to_png_bytes(annotated_path, page_idx=0, zoom=1.25)
                    st.image(png_bytes, use_column_width=True)
                else:
                    st.info("PyMuPDF not available to render annotated preview; download the Annotated PDF instead.")
                with open(annotated_path, "rb") as f:
                    a_bytes = f.read()
                st.download_button("⬇️ Download Annotated PDF", data=a_bytes, file_name="annotated_with_summary.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Could not render annotated preview: {e}")
                try:
                    with open(annotated_path, "rb") as f:
                        st.download_button("⬇️ Download Annotated PDF", data=f.read(), file_name="annotated_with_summary.pdf", mime="application/pdf")
                except Exception:
                    st.error("Annotated PDF not found for download.")
        else:
            st.warning("Annotated PDF was not generated.")

        st.markdown("---")
        st.info("If previews are not visible, download the generated PDFs and open them locally.")
else:
    st.info("Upload both OLD and NEW PDFs to begin comparison.")

st.markdown("<hr><div style='text-align:center; color:gray;'>© 2025 Created by Ashutosh Nanaware. All rights reserved.</div>", unsafe_allow_html=True)


