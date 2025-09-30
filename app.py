# app.py
"""
Streamlit UI for PDF_DIFF_TOOL
Robust file upload handling using session_state so uploaded files do not "disappear".
Calls process_and_generate(old_file, new_file, ...) from pdf_diff.py
"""

import os
import io
import tempfile
import base64
from typing import Any, Dict, List

import streamlit as st

# Import the pipeline function from your pdf_diff implementation
try:
    from pdf_diff import process_and_generate
except Exception as e:
    # Fail fast with a clear message for logs if pdf_diff.py is missing or broken
    raise ImportError(f"Could not import process_and_generate from pdf_diff.py: {e}") from e

# Optional: preview renderer (PyMuPDF)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -------------------------
# Utility helpers
# -------------------------
def save_streamlit_uploaded_to_tmp(uploaded_file) -> str:
    """Save Streamlit UploadedFile to a temp file and return path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def render_pdf_page_as_png_bytes(pdf_path: str, page_idx: int = 0, zoom: float = 1.25) -> bytes:
    """Render the first page of a PDF to PNG bytes for inline image preview (requires PyMuPDF)."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed in the runtime. Install PyMuPDF to enable page previews.")
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

def dataframe_from_summary(summary_rows: List[Dict[str, Any]]):
    """Small helper to convert summary rows to dataframe for download/preview."""
    import pandas as pd
    if not summary_rows:
        return pd.DataFrame(columns=["page", "change_type", "old_snippet", "new_snippet"])
    df = pd.DataFrame(summary_rows)
    # Ensure standard column order
    cols = ["page", "change_type", "old_snippet", "new_snippet"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")
st.markdown("<h1 style='text-align:center'>PDF_DIFF TOOL</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:gray'>Created by <b>Ashutosh Nanaware</b></div>", unsafe_allow_html=True)
st.write("")

# Sidebar options
with st.sidebar:
    st.header("Options")
    highlight_opacity_percent = st.slider("Highlight opacity (%)", min_value=10, max_value=100, value=50)
    highlight_opacity = highlight_opacity_percent / 100.0
    created_by_text = st.text_input("Created by label", value="Ashutosh Nanaware")
    show_debug = st.checkbox("Show debug state (session_state)", value=False)
    st.markdown("---")
    st.markdown("Tip: If upload seems to fail, check debug state for session contents and check browser console for network errors.")

# Main upload area
col1, col2 = st.columns(2)

with col1:
    st.write("**Upload OLD PDF**")
    # Use a stable key; saved object will be in st.session_state['old_uploader']
    old_uploader = st.file_uploader("Old PDF", type=["pdf"], key="old_uploader")
    # Persist to session_state wrapper to avoid being lost across reruns
    if old_uploader is not None:
        st.session_state["old_file_obj"] = old_uploader
        st.session_state["old_file_name"] = getattr(old_uploader, "name", None)
        st.session_state["old_file_size"] = getattr(old_uploader, "size", None)
    # Show attached file info and allow removal
    if st.session_state.get("old_file_obj"):
        name = st.session_state.get("old_file_name")
        size = st.session_state.get("old_file_size")
        st.success(f"Attached: {name} ({size} bytes)")
        if st.button("Remove OLD PDF"):
            st.session_state.pop("old_file_obj", None)
            st.session_state.pop("old_file_name", None)
            st.session_state.pop("old_file_size", None)

with col2:
    st.write("**Upload NEW PDF**")
    new_uploader = st.file_uploader("New PDF", type=["pdf"], key="new_uploader")
    if new_uploader is not None:
        st.session_state["new_file_obj"] = new_uploader
        st.session_state["new_file_name"] = getattr(new_uploader, "name", None)
        st.session_state["new_file_size"] = getattr(new_uploader, "size", None)
    if st.session_state.get("new_file_obj"):
        name = st.session_state.get("new_file_name")
        size = st.session_state.get("new_file_size")
        st.success(f"Attached: {name} ({size} bytes)")
        if st.button("Remove NEW PDF"):
            st.session_state.pop("new_file_obj", None)
            st.session_state.pop("new_file_name", None)
            st.session_state.pop("new_file_size", None)

st.markdown("---")

# Show debug session state if asked
if show_debug:
    st.subheader("Session State Debug")
    safe_keys = {k: (type(v).__name__ if k.startswith("old") or k.startswith("new") else repr(v)) for k,v in st.session_state.items()}
    st.json(safe_keys)

# Validate upload presence
old_present = "old_file_obj" in st.session_state and st.session_state["old_file_obj"] is not None
new_present = "new_file_obj" in st.session_state and st.session_state["new_file_obj"] is not None

if not old_present or not new_present:
    st.info("Upload both OLD and NEW PDFs to begin comparison.")
else:
    # show a small summary card
    st.success("Both PDFs attached. You can now run the comparison.")
    # Provide a "Compare" button so user explicitly starts long job
    if st.button("🔍 Compare PDFs"):
        # Save uploaded files to temp files (process_and_generate accepts file-like OR path, but writing to disk is stable)
        try:
            old_uploaded = st.session_state["old_file_obj"]
            new_uploaded = st.session_state["new_file_obj"]

            # Save to temp paths so the diff engine can open them repeatedly without stream exhaustion
            old_tmp_path = save_streamlit_uploaded_to_tmp = None
        except Exception:
            old_tmp_path = None

        # Save to filesystem (robust)
        try:
            old_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            old_tmp.write(st.session_state["old_file_obj"].read())
            old_tmp.flush()
            old_tmp.close()
            old_tmp_path = old_tmp.name

            new_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            new_tmp.write(st.session_state["new_file_obj"].read())
            new_tmp.flush()
            new_tmp.close()
            new_tmp_path = new_tmp.name
        except Exception as e:
            st.error(f"Failed to write uploaded files to disk: {e}")
            raise

        # Run pipeline
        with st.spinner("Running PDF comparison — this may take a minute for larger files..."):
            try:
                annotated_out, side_by_side_out, summary_rows = process_and_generate(
                    old_tmp_path,
                    new_tmp_path,
                    highlight_opacity=highlight_opacity,
                    created_by=created_by_text
                )
            except Exception as e:
                st.error(f"Comparison failed: {e}")
                st.exception(e)
                raise

        st.success("Comparison completed ✅")

        # Summary table
        st.header("Summary of Changes")
        df = dataframe_from_summary(summary_rows)
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download summary CSV", data=csv_bytes, file_name="summary.csv", mime="text/csv")

        st.markdown("---")

        # Side-by-side preview & download
        st.subheader("Side-by-Side (Old | New)")
        if os.path.exists(side_by_side_out):
            try:
                if fitz is not None:
                    png = render_pdf_page_as_png_bytes(side_by_side_out, page_idx=0, zoom=1.25)
                    st.image(png, use_column_width=True)
                with open(side_by_side_out, "rb") as f:
                    st.download_button("⬇ Download Side-by-Side PDF", data=f.read(), file_name="side_by_side.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Preview failed: {e}")
                # still allow download
                try:
                    with open(side_by_side_out, "rb") as f:
                        st.download_button("⬇ Download Side-by-Side PDF", data=f.read(), file_name="side_by_side.pdf", mime="application/pdf")
                except Exception:
                    st.error("Side-by-side PDF not available.")
        else:
            st.warning("Side-by-side PDF not generated.")

        st.markdown("---")

        # Annotated PDF preview & download
        st.subheader("Annotated NEW PDF (highlights + summary appended)")
        if os.path.exists(annotated_out):
            try:
                if fitz is not None:
                    png = render_pdf_page_as_png_bytes(annotated_out, page_idx=0, zoom=1.25)
                    st.image(png, use_column_width=True)
                with open(annotated_out, "rb") as f:
                    st.download_button("⬇ Download Annotated PDF", data=f.read(), file_name="annotated_with_summary.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"Annotated preview failed: {e}")
                try:
                    with open(annotated_out, "rb") as f:
                        st.download_button("⬇ Download Annotated PDF", data=f.read(), file_name="annotated_with_summary.pdf", mime="application/pdf")
                except Exception:
                    st.error("Annotated PDF not available.")
        else:
            st.warning("Annotated PDF not generated.")

        st.markdown("---")
        st.info("If previews look empty, download PDFs and open them locally in Acrobat/Reader for best fidelity.")

# Footer
st.markdown("<hr><div style='text-align:center; color:gray;'>© 2025 Created by Ashutosh Nanaware. All rights reserved.</div>", unsafe_allow_html=True)



