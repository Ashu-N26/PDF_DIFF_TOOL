# streamlit_app.py
"""
Streamlit frontend for PDF_DIFF_TOOL
Created by Ashutosh Nanaware
"""

import streamlit as st
import tempfile
import os
from pdf_diff_utils import (
    process_and_annotate_pdfs,
    create_side_by_side_pdf
)

st.set_page_config(page_title="PDF_DIFF_TOOL", layout="wide")
st.title("PDF_DIFF_TOOL — Token-level PDF diff (Created by Ashutosh Nanaware)")
st.markdown("**Rules:** Inserted → Green (50% opacity). Changed → Red (50% opacity). Removed → Listed only in Summary panel.")

with st.sidebar:
    st.header("Upload PDFs")
    old_pdf = st.file_uploader("Old PDF (left)", type="pdf")
    new_pdf = st.file_uploader("New PDF (right)", type="pdf")
    st.markdown("---")
    st.write("Options")
    enable_ocr = st.checkbox("Enable OCR fallback (not implemented here; preprocess scanned PDFs with OCR)", value=False)
    st.markdown("If your PDFs are scanned images, run OCR (e.g. OCRmyPDF) before uploading.")
    st.markdown("---")
    st.caption("Created by Ashutosh Nanaware")

if old_pdf and new_pdf:
    if st.button("Compare & Generate Annotated PDFs"):
        st.info("Processing — token-level diffing and annotations in progress. This may take a moment for large PDFs.")
        tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
        old_path = os.path.join(tmpdir, "old.pdf")
        new_path = os.path.join(tmpdir, "new.pdf")
        with open(old_path, "wb") as f:
            f.write(old_pdf.getbuffer())
        with open(new_path, "wb") as f:
            f.write(new_pdf.getbuffer())

        try:
            # process
            annotated_new_path, summary_pdf_path, summary_rows = process_and_annotate_pdfs(
                old_path, new_path, tmpdir, highlight_opacity=0.5
            )

            # final combined PDF: annotated new + summary appended
            annotated_with_summary = os.path.join(tmpdir, "annotated_new_with_summary.pdf")
            # process_and_annotate_pdfs already wrote annotated_new_path and summary; combine here
            from pdf_diff_utils import append_pdf
            append_pdf(annotated_new_path, summary_pdf_path, annotated_with_summary)

            # side-by-side
            side_by_side_path = os.path.join(tmpdir, "side_by_side.pdf")
            create_side_by_side_pdf(old_path, annotated_new_path, side_by_side_path)

            st.success("Files generated. Download below.")

            with open(annotated_with_summary, "rb") as f:
                st.download_button("Download Annotated NEW PDF (with Summary Panel)", f.read(), file_name="Annotated_NEW_with_Summary.pdf", mime="application/pdf")

            with open(side_by_side_path, "rb") as f:
                st.download_button("Download Side-by-Side PDF (Old | New)", f.read(), file_name="Side_by_Side.pdf", mime="application/pdf")

            # Also provide raw annotated new and summary separately
            with open(annotated_new_path, "rb") as f:
                st.download_button("Download Annotated NEW PDF (no summary)", f.read(), file_name="Annotated_NEW.pdf", mime="application/pdf")

            with open(summary_pdf_path, "rb") as f:
                st.download_button("Download Summary Panel (PDF)", f.read(), file_name="Summary_Panel.pdf", mime="application/pdf")

            # Show summary table inline (short)
            if summary_rows:
                import pandas as pd
                df = pd.DataFrame(summary_rows)
                st.markdown("### Summary (short view)")
                st.dataframe(df.head(50))

        except Exception as e:
            st.error(f"Processing failed: {e}")
            raise
else:
    st.info("Upload both Old and New PDFs in the sidebar to enable comparison.")


