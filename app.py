# app.py
import streamlit as st
from pdf_diff import process_and_generate
import fitz
import os
import pandas as pd

st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")
st.markdown("<h1 style='text-align:center;'>📑 PDF_DIFF TOOL</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:gray;'>Created by <b>Ashutosh Nanaware</b></div>", unsafe_allow_html=True)
st.write("")  # spacing

# Uploaders
col1, col2 = st.columns(2)
with col1:
    old_file = st.file_uploader("Upload OLD PDF", type=["pdf"], key="old")
with col2:
    new_file = st.file_uploader("Upload NEW PDF", type=["pdf"], key="new")

st.markdown("---")

if old_file and new_file:
    st.success("✅ Both PDFs uploaded")
    if st.button("🔍 Compare & Generate"):
        with st.spinner("Processing — extracting text, running OCR if required, annotating... This may take a while for large PDFs"):
            try:
                # process_and_generate accepts file-like objects or paths
                annotated_path, side_by_side_path, summary_rows = process_and_generate(old_file, new_file)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                # Show a short hint to check logs
                st.stop()

        st.success("✅ Generated outputs")

        # --- Side-by-side preview ---
        st.subheader("Side-by-Side Preview (Old left | New right)")
        try:
            doc = fitz.open(side_by_side_path)
            page_count = doc.page_count
            page_idx = st.slider("Preview page", 1, page_count, 1, key="sbspage")
            pix = doc.load_page(page_idx - 1).get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            st.image(pix.tobytes("png"), use_column_width=True)
            doc.close()
        except Exception as e:
            st.warning(f"Could not render side-by-side preview: {e}")

        # Side-by-side download
        try:
            with open(side_by_side_path, "rb") as f:
                sbytes = f.read()
            st.download_button("⬇️ Download Side-by-Side PDF", sbytes, file_name="side_by_side.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"Could not load side-by-side file for download: {e}")

        st.markdown("---")

        # --- Annotated preview ---
        st.subheader("Annotated NEW PDF (highlights) + Summary")
        try:
            adoc = fitz.open(annotated_path)
            a_page_count = adoc.page_count
            a_idx = st.slider("Annotated page", 1, a_page_count, 1, key="annpage")
            apix = adoc.load_page(a_idx - 1).get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            st.image(apix.tobytes("png"), use_column_width=True)
            adoc.close()
        except Exception as e:
            st.warning(f"Could not render annotated preview: {e}")

        # Annotated download
        try:
            with open(annotated_path, "rb") as f:
                abytes = f.read()
            st.download_button("⬇️ Download Annotated PDF (with Summary)", abytes, file_name="annotated_with_summary.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"Could not load annotated file for download: {e}")

        st.markdown("---")

        # --- Summary table ---
        st.subheader("Summary of Changes")
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            # Ensure columns are present and ordered
            expected_cols = ["page", "change_type", "old_snippet", "new_snippet"]
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[expected_cols]
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download Summary CSV", df.to_csv(index=False).encode("utf-8"), file_name="summary.csv", mime="text/csv")
        else:
            st.info("No differences detected or summary is empty.")

else:
    st.info("Upload both OLD and NEW PDFs to start the comparison.")









