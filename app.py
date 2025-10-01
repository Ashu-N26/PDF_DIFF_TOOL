# app.py
"""
Streamlit frontend for PDF_DIFF_TOOL (full-featured, robust).
- Upload OLD and NEW PDFs
- Configure options in sidebar
- Run advanced comparison pipeline (process_and_generate preferred)
- Preview and download:
    * Side-by-side PDF
    * Annotated PDF (NEW with highlights + summary appended)
    * Summary CSV
- Uses st.session_state to persist uploads & results across reruns
- Hides the "Created by" label under the title, keeps footer only
Author: Assistant (adapted to user's requirements)
"""

import os
import io
import tempfile
import time
import base64
import traceback
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Try to import pipeline(s) from pdf_diff.py with graceful fallback
process_and_generate = None
generate_annotated_pdf = None
generate_side_by_side_pdf = None
try:
    # Preferred single-pipeline API
    from pdf_diff import process_and_generate  # type: ignore
except Exception:
    process_and_generate = None

if process_and_generate is None:
    try:
        # Fallback to separate functions (older API)
        from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf  # type: ignore
    except Exception:
        generate_annotated_pdf = None
        generate_side_by_side_pdf = None

# Optional preview dependency
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# Optional pandas for summary
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

# ----------------------------
# Utility helpers
# ----------------------------
def make_temp_file_from_uploaded(uploaded_file: Any, suffix: str = ".pdf") -> str:
    """
    Save a Streamlit UploadedFile to a temporary file and return path.
    This function reads uploaded_file.read() once and writes bytes to disk.
    """
    if uploaded_file is None:
        raise ValueError("No uploaded file provided.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        # If uploaded_file is a stream-like object (Streamlit UploadedFile),
        # read bytes and write.
        data = uploaded_file.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def render_pdf_embed_bytes(pdf_bytes: bytes, height: int = 720) -> None:
    """
    Embed PDF bytes into Streamlit via a base64 iframe.
    Browser must support data URIs for PDFs.
    """
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px"></iframe>'
    st.components.v1.html(html, height=height)


def render_pdf_first_page_png(pdf_path: str, zoom: float = 1.25) -> Optional[bytes]:
    """
    Render the first page of a PDF into PNG bytes using PyMuPDF (if available).
    Return None if fitz is not available or rendering fails.
    """
    if fitz is None:
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png = pix.tobytes("png")
        doc.close()
        return png
    except Exception:
        return None


def df_from_summary_rows(summary_rows: List[Dict[str, Any]]):
    """
    Convert summary_rows returned by process_and_generate into a pandas DataFrame
    to display in Streamlit and to export as CSV. Returns None if pandas not installed.
    """
    if pd is None:
        return None
    if not summary_rows:
        return pd.DataFrame(columns=["page", "change_type", "old_snippet", "new_snippet"])
    df = pd.DataFrame(summary_rows)
    # Ensure columns exist
    for col in ["page", "change_type", "old_snippet", "new_snippet"]:
        if col not in df.columns:
            df[col] = ""
    return df[["page", "change_type", "old_snippet", "new_snippet"]]


def safe_remove_file(path: Optional[str]):
    if not path:
        return
    try:
        os.remove(path)
    except Exception:
        pass


# ----------------------------
# Streamlit page layout & state
# ----------------------------
st.set_page_config(page_title="PDF_DIFF TOOL", page_icon="📄", layout="wide")
# Title — intentionally no 'Created by' under the title
st.markdown("<h1 style='text-align:center;'>PDF_DIFF TOOL</h1>", unsafe_allow_html=True)
st.write("")  # spacing

# Sidebar options
with st.sidebar:
    st.header("Options")
    highlight_opacity_pct = st.slider("Highlight opacity (%)", min_value=10, max_value=100, value=50)
    highlight_opacity = float(highlight_opacity_pct) / 100.0
    use_ocr = st.checkbox("Enable OCR fallback (pytesseract)", value=True, help="If pages are scanned images, OCR will be used to extract text.")
    preview_with_images = st.checkbox("Render inline page preview (PyMuPDF)", value=True, help="Requires PyMuPDF installed on the server.")
    show_debug = st.checkbox("Show debug panel", value=False)
    st.markdown("---")
    st.markdown("Upload both OLD and NEW PDFs and click **Compare** to run the pipeline.")

# Initialize session_state slots
if "old_file_path" not in st.session_state:
    st.session_state["old_file_path"] = None
if "new_file_path" not in st.session_state:
    st.session_state["new_file_path"] = None
if "annotated_path" not in st.session_state:
    st.session_state["annotated_path"] = None
if "side_by_side_path" not in st.session_state:
    st.session_state["side_by_side_path"] = None
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []
if "last_run_time" not in st.session_state:
    st.session_state["last_run_time"] = None
if "error" not in st.session_state:
    st.session_state["error"] = None

# Upload columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload OLD PDF")
    uploaded_old = st.file_uploader("Drag and drop OLD PDF", type=["pdf"], key="old_uploader")
    if uploaded_old is not None:
        # Save immediately to a temp file and store path in session_state
        try:
            path = make_temp_file_from_uploaded(uploaded_old)
            # remove previous path if present
            old_prev = st.session_state.get("old_file_path")
            if old_prev and os.path.exists(old_prev):
                safe_remove_file(old_prev)
            st.session_state["old_file_path"] = path
            st.success(f"OLD attached: {os.path.basename(path)}")
        except Exception as e:
            st.error(f"Failed to save OLD upload: {e}")

with col2:
    st.subheader("Upload NEW PDF")
    uploaded_new = st.file_uploader("Drag and drop NEW PDF", type=["pdf"], key="new_uploader")
    if uploaded_new is not None:
        try:
            path = make_temp_file_from_uploaded(uploaded_new)
            new_prev = st.session_state.get("new_file_path")
            if new_prev and os.path.exists(new_prev):
                safe_remove_file(new_prev)
            st.session_state["new_file_path"] = path
            st.success(f"NEW attached: {os.path.basename(path)}")
        except Exception as e:
            st.error(f"Failed to save NEW upload: {e}")

st.markdown("---")

# Buttons row: Compare, Reset, Show existing outputs
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col1:
    compare_clicked = st.button("🔍 Compare")
with btn_col2:
    reset_clicked = st.button("♻️ Reset")
with btn_col3:
    show_previous = st.button("📂 Show last results")

# Reset action
if reset_clicked:
    # Remove temp files and clear session_state results
    safe_remove_file(st.session_state.get("old_file_path"))
    safe_remove_file(st.session_state.get("new_file_path"))
    safe_remove_file(st.session_state.get("annotated_path"))
    safe_remove_file(st.session_state.get("side_by_side_path"))
    st.session_state["old_file_path"] = None
    st.session_state["new_file_path"] = None
    st.session_state["annotated_path"] = None
    st.session_state["side_by_side_path"] = None
    st.session_state["summary_rows"] = []
    st.session_state["last_run_time"] = None
    st.session_state["error"] = None
    st.experimental_rerun()

# Helper: show last results if requested
if show_previous and st.session_state.get("annotated_path"):
    st.info("Showing results from last run.")
    # fall through to display below

# Validate presence of files
old_path = st.session_state.get("old_file_path")
new_path = st.session_state.get("new_file_path")

if compare_clicked:
    # Basic validation
    if not old_path or not new_path:
        st.error("Please upload both OLD and NEW PDFs before clicking Compare.")
    else:
        st.session_state["error"] = None
        # Run pipeline in try/except and show spinner
        with st.spinner("Running comparison pipeline — this can take a while depending on PDF size..."):
            start_time = time.time()
            try:
                # Preferred API
                if process_and_generate is not None:
                    # process_and_generate should accept file-like or paths; try passing paths
                    try:
                        annotated_out, sbs_out, summary_rows = process_and_generate(
                            old_path, new_path, workdir=None, highlight_opacity=highlight_opacity, created_by="Ashutosh Nanaware"
                        )
                    except Exception as e:
                        # Some implementations accept file-like UploadedFiles — but we have paths, so try again with paths
                        # fallback already attempted; if it fails, raise.
                        raise
                else:
                    # Use separate functions fallback
                    if generate_annotated_pdf is None or generate_side_by_side_pdf is None:
                        raise RuntimeError("pdf_diff.py does not expose process_and_generate or the fallback functions. Check pdf_diff.py.")
                    # annotated may return (path, summary) or just path
                    tmp_annot = os.path.join(tempfile.mkdtemp(prefix="annot_"), "annotated_out.pdf")
                    res = generate_annotated_pdf(old_path, new_path, tmp_annot, highlight_opacity=highlight_opacity, created_by="Ashutosh Nanaware")
                    if isinstance(res, tuple):
                        annotated_out, summary_rows = res
                    else:
                        annotated_out = res
                        summary_rows = []
                    tmp_sbs = os.path.join(tempfile.mkdtemp(prefix="sbs_"), "side_by_side.pdf")
                    generate_side_by_side_pdf(old_path, new_path, tmp_sbs)
                    sbs_out = tmp_sbs

                # store outputs in session_state
                st.session_state["annotated_path"] = annotated_out
                st.session_state["side_by_side_path"] = sbs_out
                st.session_state["summary_rows"] = summary_rows or []
                st.session_state["last_run_time"] = time.time()
                elapsed = time.time() - start_time
                st.success(f"Comparison finished in {elapsed:.1f}s ✅")

            except Exception as err:
                st.session_state["error"] = str(err)
                st.error(f"Comparison failed: {err}")
                # show traceback in debug mode
                if show_debug:
                    st.text(traceback.format_exc())

# If we have outputs, display in tabs
if st.session_state.get("annotated_path") or st.session_state.get("side_by_side_path"):
    tab1, tab2, tab3 = st.tabs(["Summary", "Side-by-Side Preview", "Annotated PDF"])

    # ---------- Summary Tab ----------
    with tab1:
        st.header("Summary Panel")
        summary_rows = st.session_state.get("summary_rows", [])
        if summary_rows:
            df = df_from_summary_rows(summary_rows)
            if df is not None:
                st.dataframe(df, use_container_width=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Summary CSV", data=csv_bytes, file_name="summary.csv", mime="text/csv")
            else:
                # pandas not available, show raw
                st.write(summary_rows)
        else:
            st.info("No summary rows available. If you ran comparison, make sure pdf_diff returned summary data.")

    # ---------- Side-by-Side Preview ----------
    with tab2:
        st.header("Side-by-Side Comparison")
        sbs = st.session_state.get("side_by_side_path")
        if sbs and os.path.exists(sbs):
            # Render preview as first-page PNG if requested & available
            if preview_with_images and fitz is not None:
                png = render_pdf_first_page_png(sbs, zoom=1.25)
                if png:
                    st.image(png, caption="Side-by-Side first page preview", use_column_width=True)
                else:
                    st.info("Preview rendering failed; download the PDF to view full content.")
            else:
                st.info("Inline preview disabled or PyMuPDF not available. Use the download button to view.")
            with open(sbs, "rb") as f:
                sbs_bytes = f.read()
            st.download_button("⬇ Download Side-by-Side PDF", data=sbs_bytes, file_name="side_by_side.pdf", mime="application/pdf")
        else:
            st.info("Side-by-side PDF not yet generated. Run the comparison first.")

    # ---------- Annotated PDF Tab ----------
    with tab3:
        st.header("Annotated PDF (NEW with highlights + Summary appended)")
        ann = st.session_state.get("annotated_path")
        if ann and os.path.exists(ann):
            # Show preview image if available
            if preview_with_images and fitz is not None:
                png = render_pdf_first_page_png(ann, zoom=1.25)
                if png:
                    st.image(png, caption="Annotated PDF first page preview", use_column_width=True)
            else:
                st.info("Inline preview disabled or PyMuPDF not available.")
            with open(ann, "rb") as f:
                ann_bytes = f.read()
            st.download_button("⬇ Download Annotated PDF", data=ann_bytes, file_name="annotated_with_summary.pdf", mime="application/pdf")
        else:
            st.info("Annotated PDF not yet generated. Run the comparison first.")
else:
    # No outputs yet; show helpful tips or prior error
    if st.session_state.get("error"):
        st.warning("Previous run failed. Fix the issue or try again.")
    else:
        st.info("Upload both PDFs and click Compare to generate annotated & side-by-side outputs.")

# Debug panel (optional)
if show_debug:
    st.markdown("---")
    st.subheader("Debug")
    st.write("session_state keys:", list(st.session_state.keys()))
    st.write("old_file_path:", st.session_state.get("old_file_path"))
    st.write("new_file_path:", st.session_state.get("new_file_path"))
    st.write("annotated_path:", st.session_state.get("annotated_path"))
    st.write("side_by_side_path:", st.session_state.get("side_by_side_path"))
    st.write("last_run_time:", st.session_state.get("last_run_time"))
    st.write("error:", st.session_state.get("error"))

# Footer (keep)
st.markdown(
    "<div style='text-align:center; color: #888; margin-top: 30px;'>© 2025 Created by Ashutosh Nanaware. All rights reserved.</div>",
    unsafe_allow_html=True,
)







