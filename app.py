# app.py
"""
Memory-optimized Streamlit UI for PDF_DIFF_TOOL.

Key behavior:
- Auto-detects whether OCR is required and only runs it as fallback per-page.
- Uses low-resolution rendering for previews to keep memory usage low.
- Freely cleans up pixmaps and temporary files to avoid memory leaks.
- Integrates with pdf_diff.py: prefers `process_and_generate`, falls back to `generate_annotated_pdf` + `generate_side_by_side_pdf`.
- Keeps UI simple: Upload OLD/NEW PDFs, Compare, Preview, Download annotated & side-by-side outputs, Summary CSV.

Author: Assistant (adapted for user)
"""

import os
import io
import tempfile
import time
import base64
import traceback
import gc
from typing import Any, Dict, List, Optional, Tuple, Callable

import streamlit as st

# Attempt to import optional runtime libs (safe fail)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import pandas as pd
except Exception:
    pd = None

# Try to import pipeline(s) from pdf_diff.py with fallback handling
process_and_generate = None
generate_annotated_pdf = None
generate_side_by_side_pdf = None

try:
    from pdf_diff import process_and_generate  # type: ignore
except Exception:
    process_and_generate = None

if process_and_generate is None:
    try:
        from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf  # type: ignore
    except Exception:
        generate_annotated_pdf = None
        generate_side_by_side_pdf = None

# ---------- Utility helpers ----------

def save_uploaded_to_temp(uploaded_file: Any, suffix: str = ".pdf") -> str:
    """Save a Streamlit UploadedFile to disk and return path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        data = uploaded_file.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


def safe_remove(path: Optional[str]):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def detect_text_presence(path: str, max_pages_to_sample: int = 3) -> bool:
    """
    Return True if the PDF appears to contain extractable text.
    Strategy:
      - Prefer PyPDF2 text extraction (lightweight).
      - If PyPDF2 not present but fitz available, use fitz.get_text() (still light for sampling).
      - If neither available, assume False (scanned) to be safe.
    """
    try:
        if PyPDF2 is not None:
            reader = PyPDF2.PdfReader(path)
            num_pages = len(reader.pages)
            sample_count = min(max_pages_to_sample, num_pages)
            nonempty = 0
            for i in range(sample_count):
                try:
                    page = reader.pages[i]
                    txt = page.extract_text()
                    if txt and txt.strip():
                        nonempty += 1
                except Exception:
                    continue
            # if at least one sample page has text, assume document has text
            return nonempty > 0
        elif fitz is not None:
            doc = fitz.open(path)
            sample_count = min(max_pages_to_sample, len(doc))
            nonempty = 0
            for i in range(sample_count):
                try:
                    p = doc.load_page(i)
                    t = p.get_text().strip()
                    if t:
                        nonempty += 1
                except Exception:
                    continue
            doc.close()
            return nonempty > 0
    except Exception:
        # any unexpected error -> treat as no text to be conservative
        return False
    return False


def render_pdf_page_png_bytes(pdf_path: str, page_idx: int = 0, zoom: float = 1.0) -> Optional[bytes]:
    """Render a PDF page to PNG bytes using PyMuPDF. Keep zoom small for memory safety."""
    if fitz is None:
        return None
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= doc.page_count:
            doc.close()
            return None
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        # free resources
        del pix
        del page
        doc.close()
        gc.collect()
        return png_bytes
    except Exception:
        return None


def dataframe_from_summary(summary_rows: List[Dict[str, Any]]):
    """Return pandas DataFrame or None if pandas not installed"""
    if pd is None:
        return None
    if not summary_rows:
        return pd.DataFrame(columns=["page", "change_type", "old_snippet", "new_snippet"])
    df = pd.DataFrame(summary_rows)
    for c in ["page", "change_type", "old_snippet", "new_snippet"]:
        if c not in df.columns:
            df[c] = ""
    return df[["page", "change_type", "old_snippet", "new_snippet"]]


def try_call_with_kwargs_fallback(func: Callable, *args, **kwargs):
    """
    Call func with kwargs. If TypeError due to unexpected kwargs, call with args only.
    Returns whatever func returns.
    """
    try:
        return func(*args, **kwargs)
    except TypeError as e:
        # likely unexpected kwargs - try positional only
        return func(*args)


# ---------- Streamlit UI ----------

st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")
st.markdown("<h1 style='text-align:center;'>PDF_DIFF TOOL</h1>", unsafe_allow_html=True)
st.write("")  # spacing

# Sidebar: memory-conscious defaults
with st.sidebar:
    st.header("Options")
    highlight_opacity_pct = st.slider("Highlight opacity (%)", 10, 100, 50)
    highlight_opacity = float(highlight_opacity_pct) / 100.0
    auto_detect_ocr = st.checkbox("Auto-detect OCR (recommended)", value=True, help="Tool will only run OCR on pages that lack extractable text.")
    enable_preview = st.checkbox(" Enable inline preview (uses more memory)", value=False)
    preview_zoom = st.selectbox("Preview zoom (memory vs clarity)", options=[0.75, 1.0, 1.25], index=1)
    show_debug = st.checkbox("Show debug panel", value=False)

st.markdown("---")

# Upload area
col_old, col_new = st.columns(2)
with col_old:
    st.subheader("Upload OLD PDF")
    uploaded_old = st.file_uploader("Drag and drop OLD PDF", type=["pdf"], key="old_uploader")
with col_new:
    st.subheader("Upload NEW PDF")
    uploaded_new = st.file_uploader("Drag and drop NEW PDF", type=["pdf"], key="new_uploader")

# Save uploaded files to temp paths and persist in session_state
if uploaded_old is not None:
    try:
        tmp_old = save_uploaded_to_temp(uploaded_old)
        prev_old = st.session_state.get("old_path")
        if prev_old and prev_old != tmp_old:
            safe_remove(prev_old)
        st.session_state["old_path"] = tmp_old
        st.success(f"OLD attached: {os.path.basename(tmp_old)}")
    except Exception as e:
        st.error(f"Unable to save OLD upload: {e}")

if uploaded_new is not None:
    try:
        tmp_new = save_uploaded_to_temp(uploaded_new)
        prev_new = st.session_state.get("new_path")
        if prev_new and prev_new != tmp_new:
            safe_remove(prev_new)
        st.session_state["new_path"] = tmp_new
        st.success(f"NEW attached: {os.path.basename(tmp_new)}")
    except Exception as e:
        st.error(f"Unable to save NEW upload: {e}")

old_path: Optional[str] = st.session_state.get("old_path")
new_path: Optional[str] = st.session_state.get("new_path")

st.markdown("---")

# Buttons row
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    compare_btn = st.button("🔍 Compare")
with c2:
    reset_btn = st.button("♻️ Reset")
with c3:
    show_last = st.button("📂 Show last results")

# Reset behavior
if reset_btn:
    # remove temp files and outputs
    safe_remove(st.session_state.get("old_path"))
    safe_remove(st.session_state.get("new_path"))
    safe_remove(st.session_state.get("annotated_path"))
    safe_remove(st.session_state.get("side_by_side_path"))
    st.session_state["old_path"] = None
    st.session_state["new_path"] = None
    st.session_state["annotated_path"] = None
    st.session_state["side_by_side_path"] = None
    st.session_state["summary_rows"] = []
    st.session_state["last_run_time"] = None
    st.session_state["error"] = None
    st.experimental_rerun()

# Helper to show last results if exist
if show_last and (st.session_state.get("annotated_path") or st.session_state.get("side_by_side_path")):
    st.success("Showing previously generated outputs (from last run).")

# If compare button clicked, run pipeline
if compare_btn:
    if not old_path or not new_path:
        st.error("Please upload both OLD and NEW PDFs before clicking Compare.")
    else:
        st.session_state["error"] = None
        # Step 1: detect whether OCR is needed for either doc (if auto_detect_ocr enabled)
        needs_ocr_old = False
        needs_ocr_new = False
        if auto_detect_ocr:
            try:
                needs_ocr_old = not detect_text_presence(old_path, max_pages_to_sample=3)
                needs_ocr_new = not detect_text_presence(new_path, max_pages_to_sample=3)
            except Exception:
                # if detection fails, be conservative and assume no text -> OCR required
                needs_ocr_old = True
                needs_ocr_new = True

        needs_ocr = needs_ocr_old or needs_ocr_new

        # Inform the user briefly
        if needs_ocr:
            st.info("Auto-detect: one or both PDFs appear to be scanned / image-based. OCR will be used where needed.")
        else:
            st.info("Auto-detect: documents contain extractable text. OCR will be skipped.")

        # Step 2: run pipeline with memory-safety measures
        out_annot = None
        out_sbs = None
        summary_rows: List[Dict[str, Any]] = []

        # Provide progress UI
        progress = st.progress(0)
        start_time = time.time()
        try:
            # Preferred: process_and_generate
            if process_and_generate is not None:
                # Build kwargs we want to pass. We will try with kwargs, fallback to positional if needed.
                kwargs = {
                    "workdir": None,
                    "highlight_opacity": highlight_opacity,
                    "created_by": "Ashutosh Nanaware",
                }
                # Many implementations accept a use_ocr or ocr flag - only include if we detected OCR needed
                if needs_ocr:
                    # include both common names; the wrapper will attempt and fall back
                    kwargs["use_ocr"] = True
                    kwargs["ocr"] = True

                # Call with reasonable attempt: prefer file paths (safer memory-wise)
                try:
                    progress.progress(10)
                    res = try_call_with_kwargs_fallback(process_and_generate, old_path, new_path, **kwargs)
                except Exception as e:
                    # second attempt: maybe the function expects file-like. pass open files but ensure we close.
                    try:
                        with open(old_path, "rb") as f_old, open(new_path, "rb") as f_new:
                            progress.progress(20)
                            res = try_call_with_kwargs_fallback(process_and_generate, f_old, f_new, **kwargs)
                    except Exception:
                        raise

                # Expect process_and_generate to return (annotated_path, side_by_side_path, summary_rows)
                if isinstance(res, tuple) and len(res) >= 2:
                    out_annot = res[0]
                    out_sbs = res[1] if len(res) > 1 else None
                    summary_rows = res[2] if len(res) > 2 else []
                else:
                    # some implementations might return only annotated path and summary, try to adapt
                    out_annot = res
                    out_sbs = None
                    summary_rows = []
            else:
                # fallback to separate functions
                if generate_annotated_pdf is None or generate_side_by_side_pdf is None:
                    raise RuntimeError("pdf_diff.py does not expose 'process_and_generate' or fallback functions. Update pdf_diff.py accordingly.")
                progress.progress(10)
                tmp_dir = tempfile.mkdtemp(prefix="pdfdiff_run_")
                annotated_out = os.path.join(tmp_dir, "annotated_out.pdf")
                side_by_side_out = os.path.join(tmp_dir, "side_by_side.pdf")

                # Try calling annotated generator with kwargs fallback
                kwargs = {"highlight_opacity": highlight_opacity, "created_by": "Ashutosh Nanaware"}
                if needs_ocr:
                    kwargs["use_ocr"] = True
                    kwargs["ocr"] = True
                try:
                    progress.progress(20)
                    res = try_call_with_kwargs_fallback(generate_annotated_pdf, old_path, new_path, annotated_out, **kwargs)
                except Exception:
                    # last resort: positional only
                    res = generate_annotated_pdf(old_path, new_path, annotated_out)
                # res might be annotated path or (annotated_path, summary_rows)
                if isinstance(res, tuple):
                    out_annot, summary_rows = res[0], res[1]
                else:
                    out_annot = res
                    summary_rows = []

                # generate side-by-side
                try:
                    progress.progress(50)
                    try_call_with_kwargs_fallback(generate_side_by_side_pdf, old_path, new_path, side_by_side_out)
                except Exception as e:
                    # ignore side-by-side if it fails, continue with annotated
                    side_by_side_out = None
                out_sbs = side_by_side_out

            elapsed = time.time() - start_time
            progress.progress(100)

            # Persist outputs
            st.session_state["annotated_path"] = out_annot
            st.session_state["side_by_side_path"] = out_sbs
            st.session_state["summary_rows"] = summary_rows or []
            st.session_state["last_run_time"] = time.time()
            st.success(f"Comparison finished — took {elapsed:.1f}s")

        except Exception as err:
            st.session_state["error"] = str(err)
            st.error(f"Comparison failed: {err}")
            if show_debug:
                st.text(traceback.format_exc())

# ---------- Display results (if any) ----------
annotated_path = st.session_state.get("annotated_path")
side_by_side_path = st.session_state.get("side_by_side_path")
summary_rows = st.session_state.get("summary_rows", [])

if annotated_path or side_by_side_path:
    t1, t2, t3 = st.tabs(["Summary", "Side-by-Side", "Annotated PDF"])

    # Summary tab
    with t1:
        st.header("Summary Panel")
        if summary_rows:
            df = dataframe_from_summary(summary_rows)
            if df is not None:
                st.dataframe(df, use_container_width=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Summary CSV", data=csv_bytes, file_name="summary.csv", mime="text/csv")
            else:
                st.write(summary_rows)
        else:
            st.info("No summary available. The comparison function may not produce structured summary rows.")

    # Side-by-Side tab
    with t2:
        st.header("Side-by-Side Comparison")
        if side_by_side_path and os.path.exists(side_by_side_path):
            if enable_preview and fitz is not None:
                png = render_pdf_page_png_bytes(side_by_side_path, page_idx=0, zoom=float(preview_zoom))
                if png:
                    st.image(png, use_column_width=True)
                else:
                    st.info("Preview not available for Side-by-Side.")
            with open(side_by_side_path, "rb") as f:
                data = f.read()
            st.download_button("⬇ Download Side-by-Side PDF", data=data, file_name="side_by_side.pdf", mime="application/pdf")
        else:
            st.info("Side-by-side PDF not generated.")

    # Annotated tab
    with t3:
        st.header("Annotated NEW PDF (highlights + summary appended)")
        if annotated_path and os.path.exists(annotated_path):
            if enable_preview and fitz is not None:
                png = render_pdf_page_png_bytes(annotated_path, page_idx=0, zoom=float(preview_zoom))
                if png:
                    st.image(png, use_column_width=True)
                else:
                    st.info("Annotated preview not available.")
            with open(annotated_path, "rb") as f:
                data = f.read()
            st.download_button("⬇ Download Annotated PDF", data=data, file_name="annotated_with_summary.pdf", mime="application/pdf")
        else:
            st.info("Annotated PDF not generated.")

else:
    # no outputs yet
    if st.session_state.get("error"):
        st.warning("Previous run failed. See error above.")
    else:
        st.info("Upload both PDFs and click Compare to generate outputs.")

# Debug panel
if show_debug:
    st.markdown("---")
    st.subheader("Debug information")
    st.write("old_path:", st.session_state.get("old_path"))
    st.write("new_path:", st.session_state.get("new_path"))
    st.write("annotated_path:", st.session_state.get("annotated_path"))
    st.write("side_by_side_path:", st.session_state.get("side_by_side_path"))
    st.write("last_run_time:", st.session_state.get("last_run_time"))
    st.write("error:", st.session_state.get("error"))

# Footer (kept)
st.markdown(
    "<div style='text-align:center; color: #888; margin-top: 30px;'>© 2025 Created by Ashutosh Nanaware. All rights reserved.</div>",
    unsafe_allow_html=True,
)








