# app.py
"""
Streamlit UI for PDF_DIFF_TOOL — robust importer & fallback-aware.
This app will try to use process_and_generate() if available in pdf_diff.py,
otherwise will fall back to generate_annotated_pdf + generate_side_by_side_pdf.
"""

import os
import io
import tempfile
import base64
from typing import Any, Dict, List, Tuple

import streamlit as st

# Try to import common pipeline functions from pdf_diff.py in a safe way.
process_and_generate = None
generate_annotated_pdf = None
generate_side_by_side_pdf = None

try:
    # Preferred: single pipeline function that accepts file-like objects
    from pdf_diff import process_and_generate  # type: ignore
except Exception:
    process_and_generate = None

if process_and_generate is None:
    try:
        # Fallback: older API where separate functions exist (paths expected)
        from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf  # type: ignore
    except Exception:
        generate_annotated_pdf = None
        generate_side_by_side_pdf = None

# Try optional preview dependency
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -------------------------
# Helpers
# -------------------------
def _save_uploaded_to_tmp(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temporary file and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def _render_pdf_page_to_png_bytes(pdf_path: str, page_idx: int = 0, zoom: float = 1.25) -> bytes:
    """Render a PDF page to PNG bytes using PyMuPDF. Raises RuntimeError if fitz not available."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed; cannot render previews.")
    doc = fitz.open(pdf_path)
    if page_idx < 0 or page_idx >= doc.page_count:
        doc.close()
        raise IndexError("page index out of range")
    page = doc.load_page(page_idx)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png = pix.tobytes("png")
    doc.close()
    return png

def _df_from_summary(summary_rows: List[Dict[str, Any]]):
    """Convert summary rows into a pandas DataFrame for display and CSV download."""
    try:
        import pandas as pd
    except Exception:
        return None
    if not summary_rows:
        return pd.DataFrame(columns=["page", "change_type", "old_snippet", "new_snippet"])
    df = pd.DataFrame(summary_rows)
    # ensure standardized columns
    for c in ["page", "change_type", "old_snippet", "new_snippet"]:
        if c not in df.columns:
            df[c] = ""
    return df[["page", "change_type", "old_snippet", "new_snippet"]]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")
st.markdown("<h1 style='text-align:center'>PDF_DIFF TOOL</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:gray;'>Created by <b>Ashutosh Nanaware</b></div>", unsafe_allow_html=True)
st.write("")

# Sidebar minimal options (no 'Created by' input)
with st.sidebar:
    st.header("Options")
    highlight_opacity_pct = st.slider("Highlight opacity (%)", 10, 100, 50)
    highlight_opacity = highlight_opacity_pct / 100.0
    debug_mode = st.checkbox("Show debug state", value=False)

# Uploaders with session_state persistence
col1, col2 = st.columns(2)
with col1:
    uploaded_old = st.file_uploader("Upload OLD PDF", type=["pdf"], key="old_uploader")
with col2:
    uploaded_new = st.file_uploader("Upload NEW PDF", type=["pdf"], key="new_uploader")

# persist upload objects into session_state to avoid losing them across reruns
if uploaded_old is not None:
    st.session_state["old_file_obj"] = uploaded_old
    st.session_state["old_file_name"] = uploaded_old.name
if uploaded_new is not None:
    st.session_state["new_file_obj"] = uploaded_new
    st.session_state["new_file_name"] = uploaded_new.name

old_present = bool(st.session_state.get("old_file_obj"))
new_present = bool(st.session_state.get("new_file_obj"))

# Show attached file info and allow removal
if old_present:
    st.success(f"OLD attached: {st.session_state.get('old_file_name')}")
    if st.button("Remove OLD PDF"):
        st.session_state.pop("old_file_obj", None)
        st.session_state.pop("old_file_name", None)
        old_present = False
if new_present:
    st.success(f"NEW attached: {st.session_state.get('new_file_name')}")
    if st.button("Remove NEW PDF"):
        st.session_state.pop("new_file_obj", None)
        st.session_state.pop("new_file_name", None)
        new_present = False

if debug_mode:
    st.subheader("Debug session_state")
    st.json({k: (str(type(v))) for k, v in st.session_state.items()})

st.markdown("---")

# If neither expected API exists, show user-friendly help immediately (no crash)
if (process_and_generate is None) and (generate_annotated_pdf is None or generate_side_by_side_pdf is None):
    st.error(
        "pdf_diff.py does not expose an API that this UI can use.\n\n"
        "Expected one of these: \n"
        "  • process_and_generate(old, new, workdir=None, highlight_opacity=0.5, created_by=...)  (preferred), OR\n"
        "  • generate_annotated_pdf(old_path, new_path, out_path, highlight_opacity=0.5, created_by=...) AND generate_side_by_side_pdf(old_path, new_path, out_path)\n\n"
        "Please update pdf_diff.py to export these functions. Check the server logs for details."
    )
    st.stop()

# Show helper when uploads missing
if not (old_present and new_present):
    st.info("Upload both OLD and NEW PDFs to begin comparison.")
else:
    # Compare button
    if st.button("🔍 Compare PDFs"):
        # Save uploaded files to temp paths (safe for functions that expect paths)
        old_obj = st.session_state["old_file_obj"]
        new_obj = st.session_state["new_file_obj"]
        tmpdir = tempfile.mkdtemp(prefix="pdfdiff_run_")
        old_tmp_path = os.path.join(tmpdir, "old.pdf")
        new_tmp_path = os.path.join(tmpdir, "new.pdf")
        with open(old_tmp_path, "wb") as f:
            # uploaded file may have .read() consumed earlier; ensure we get original UploadedFile from session_state
            obj = st.session_state["old_file_obj"]
            f.write(obj.read())
        with open(new_tmp_path, "wb") as f:
            obj2 = st.session_state["new_file_obj"]
            f.write(obj2.read())

        annotated_path = None
        side_by_side_path = None
        summary_rows: List[Dict[str, Any]] = []

        with st.spinner("Running comparison — this may take a while for large PDFs..."):
            try:
                if process_and_generate is not None:
                    # Preferred pipeline accepts file-like objects too
                    # We can pass the file paths (old_tmp_path/new_tmp_path) or the UploadedFile objects
                    try:
                        # Try calling with file-like objects first (some implementations accept either)
                        annotated_path, side_by_side_path, summary_rows = process_and_generate(
                            st.session_state["old_file_obj"], st.session_state["new_file_obj"],
                            workdir=tmpdir, highlight_opacity=highlight_opacity, created_by="Ashutosh Nanaware"
                        )
                    except Exception:
                        # fallback: pass paths if file-like call fails
                        annotated_path, side_by_side_path, summary_rows = process_and_generate(
                            old_tmp_path, new_tmp_path, workdir=tmpdir, highlight_opacity=highlight_opacity, created_by="Ashutosh Nanaware"
                        )
                else:
                    # Use separate functions
                    if generate_annotated_pdf is None or generate_side_by_side_pdf is None:
                        raise RuntimeError("Fallback functions are not available in pdf_diff.py")
                    # annotated function may return path or (path, summary_rows)
                    annotated_tmp = os.path.join(tmpdir, "annotated_out.pdf")
                    res = generate_annotated_pdf(old_tmp_path, new_tmp_path, annotated_tmp, highlight_opacity=highlight_opacity, created_by="Ashutosh Nanaware")
                    if isinstance(res, tuple):
                        # older implementation might return (path, summary_rows)
                        annotated_path, summary_rows = res
                    else:
                        annotated_path = res
                        summary_rows = []
                    side_by_side_tmp = os.path.join(tmpdir, "side_by_side.pdf")
                    generate_side_by_side_pdf(old_tmp_path, new_tmp_path, side_by_side_tmp)
                    side_by_side_path = side_by_side_tmp

            except Exception as e:
                st.error(f"Comparison failed: {e}")
                st.exception(e)
                st.stop()

        st.success("Comparison finished ✅")

        # Show summary
        st.subheader("Summary of Changes")
        df = _df_from_summary(summary_rows)
        if df is not None:
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download Summary CSV", df.to_csv(index=False).encode("utf-8"), file_name="summary.csv", mime="text/csv")
        else:
            st.write("Summary (rows):")
            st.write(summary_rows)

        st.markdown("---")

        # Side-by-side preview & download
        st.subheader("Side-by-Side Preview (Old | New)")
        if side_by_side_path and os.path.exists(side_by_side_path):
            try:
                if fitz is not None:
                    png = _render_pdf_page_to_png_bytes(side_by_side_path, page_idx=0, zoom=1.25)
                    st.image(png, use_column_width=True)
                with open(side_by_side_path, "rb") as f:
                    st.download_button("⬇️ Download Side-by-Side PDF", f.read(), file_name="side_by_side.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Could not render Side-by-Side preview: {e}")
                try:
                    with open(side_by_side_path, "rb") as f:
                        st.download_button("⬇️ Download Side-by-Side PDF", f.read(), file_name="side_by_side.pdf", mime="application/pdf")
                except Exception:
                    st.error("Side-by-side file not available for download.")
        else:
            st.info("Side-by-side PDF was not produced.")

        st.markdown("---")

        # Annotated PDF preview & download
        st.subheader("Annotated NEW PDF (highlights + summary)")
        if annotated_path and os.path.exists(annotated_path):
            try:
                if fitz is not None:
                    png = _render_pdf_page_to_png_bytes(annotated_path, page_idx=0, zoom=1.25)
                    st.image(png, use_column_width=True)
                with open(annotated_path, "rb") as f:
                    st.download_button("⬇️ Download Annotated PDF", f.read(), file_name="annotated_with_summary.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Could not render annotated preview: {e}")
                try:
                    with open(annotated_path, "rb") as f:
                        st.download_button("⬇️ Download Annotated PDF", f.read(), file_name="annotated_with_summary.pdf", mime="application/pdf")
                except Exception:
                    st.error("Annotated file not available for download.")
        else:
            st.info("Annotated PDF was not produced.")

st.markdown("<hr><div style='text-align:center; color:gray;'>© 2025 Created by Ashutosh Nanaware. All rights reserved.</div>", unsafe_allow_html=True)





