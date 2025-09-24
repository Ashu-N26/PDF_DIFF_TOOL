# streamlit_app.py
"""
Streamlit app: PDF diff + annotated PDF generator
Created to be deployed on Streamlit Cloud.
Author/Branding in UI: Created by Ashutosh Nanaware
"""

import streamlit as st
import tempfile
import os
import io
from pdf_diff_utils import (
    compare_pdfs_and_annotate,
    create_side_by_side_pdf,
    merge_summary_into_pdf,
)
from datetime import datetime

st.set_page_config(page_title="PDF Diff Tool (Ashutosh Nanaware)", layout="wide")

# ---- UI ----
st.title("PDF Diff Tool — Annotate & Compare PDFs")
st.markdown("**Created by Ashutosh Nanaware**")

st.sidebar.markdown("### Upload PDFs")
old_pdf = st.sidebar.file_uploader("Old PDF (left)", type=["pdf"])
new_pdf = st.sidebar.file_uploader("New PDF (right)", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.markdown("Options")
enable_ocr = st.sidebar.checkbox("Enable OCR fallback (if PDF has no selectable text)", value=False)
opacity = st.sidebar.slider("Highlight opacity (for annotations)", 0.05, 0.9, 0.28, 0.01)
numeric_detection = st.sidebar.checkbox("Try numeric detection & increase/decrease coloring", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Output files will be generated and available to download after processing.")

if old_pdf and new_pdf:
    # Save uploaded files temporarily
    tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
    old_path = os.path.join(tmpdir, "old.pdf")
    new_path = os.path.join(tmpdir, "new.pdf")
    with open(old_path, "wb") as f:
        f.write(old_pdf.getbuffer())
    with open(new_path, "wb") as f:
        f.write(new_pdf.getbuffer())

    if st.button("Compare PDFs and Generate Outputs"):
        st.info("Starting comparison — this might take a few seconds for large PDFs.")
        try:
            # compute diffs and produce annotated PDFs (old annotated and new annotated)
            annotated_old_path, annotated_new_path, diff_summary = compare_pdfs_and_annotate(
                old_path,
                new_path,
                tmpdir,
                enable_ocr=enable_ocr,
                opacity=opacity,
                numeric_detection=numeric_detection,
            )

            # Create summary page PDF and merge into annotated new PDF
            summary_pdf_path = os.path.join(tmpdir, "summary_panel.pdf")
            merge_summary_into_pdf(diff_summary, summary_pdf_path, created_by="Ashutosh Nanaware")

            final_single_pdf = os.path.join(tmpdir, f"annotated_with_summary_{int(datetime.now().timestamp())}.pdf")
            # append summary panel to annotated_new_path -> final_single_pdf
            from pdf_diff_utils import append_pdf

            append_pdf(annotated_new_path, summary_pdf_path, final_single_pdf)

            # Create side-by-side PDF (old left, new right with highlights). No summary panel on side-by-side
            side_by_side_pdf = os.path.join(tmpdir, f"side_by_side_{int(datetime.now().timestamp())}.pdf")
            create_side_by_side_pdf(annotated_old_path, annotated_new_path, side_by_side_pdf)

            st.success("Comparison finished — download files below.")

            # Provide downloads
            with open(final_single_pdf, "rb") as f:
                st.download_button("Download single annotated PDF (with summary)", f.read(), file_name="annotated_with_summary.pdf")

            with open(side_by_side_pdf, "rb") as f:
                st.download_button("Download side-by-side PDF", f.read(), file_name="side_by_side.pdf")

            with open(annotated_old_path, "rb") as f:
                st.download_button("Download annotated OLD PDF (red highlights for removed/reduced)", f.read(), file_name="annotated_old.pdf")

            with open(annotated_new_path, "rb") as f:
                st.download_button("Download annotated NEW PDF (green highlights for inserted/increased)", f.read(), file_name="annotated_new.pdf")

            # Show summary table inline (top 10 rows)
            import pandas as pd

            if diff_summary:
                df = pd.DataFrame(diff_summary)
                st.markdown("### Diff Summary (top rows)")
                st.dataframe(df.head(20))
            else:
                st.info("No differences detected (or diff summary is empty).")

        except Exception as e:
            st.error(f"Error while processing PDFs: {e}")
            raise

else:
    st.warning("Upload both Old and New PDFs in the sidebar to enable comparison.")

st.markdown("---")
st.caption("This tool highlights changes: Red = removed/reduced, Green = inserted/increased. Created by Ashutosh Nanaware")
