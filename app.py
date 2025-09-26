# app.py
import streamlit as st
from pdf_diff import process_and_generate
import tempfile
import os
from PIL import Image
import fitz
import io

st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")

st.markdown("<h1 style='text-align:center;'>📑 PDF_DIFF TOOL</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    old_file = st.file_uploader("Upload OLD PDF", type="pdf")
with col2:
    new_file = st.file_uploader("Upload NEW PDF", type="pdf")

st.markdown("---")

if old_file and new_file:
    st.success("✅ Both PDFs uploaded successfully")
    if st.button("🔍 Compare & Generate"):
        with st.spinner("Processing — extracting text, running OCR if required, annotating..."):
            tmpdir = tempfile.mkdtemp(prefix="pdfdiff_")
            try:
                annotated_path, side_by_side_path, summary_rows = process_and_generate(old_file, new_file, workdir=tmpdir, highlight_opacity=0.5)
            except Exception as e:
                st.error(f"Processing failed: {e}")
                raise

        st.success("✅ Generated Annotated PDF and Side-by-Side preview")

        # Side-by-Side Preview (render first page as image)
        st.subheader("Side-by-Side Preview")
        try:
            doc = fitz.open(side_by_side_path)
            page_count = doc.page_count
            page_idx = st.slider("Select page", 1, page_count, 1)
            pix = doc.load_page(page_idx-1).get_pixmap(matrix=fitz.Matrix(1.25,1.25), alpha=False)
            st.image(pix.tobytes("png"))
            doc.close()
        except Exception as e:
            st.warning(f"Could not render side-by-side preview: {e}")

        st.download_button("⬇️ Download Side-by-Side PDF", open(side_by_side_path,"rb").read(), file_name="side_by_side.pdf", mime="application/pdf")

        # Annotated PDF preview first page image
        st.subheader("Annotated NEW PDF (highlights)")
        try:
            doc2 = fitz.open(annotated_path)
            page_count2 = doc2.page_count
            page_idx2 = st.slider("Select annotated page", 1, page_count2, 1, key="annpage")
            pix2 = doc2.load_page(page_idx2-1).get_pixmap(matrix=fitz.Matrix(1.25,1.25), alpha=False)
            st.image(pix2.tobytes("png"))
            doc2.close()
        except Exception as e:
            st.warning(f"Could not render annotated preview: {e}")

        st.download_button("⬇️ Download Annotated PDF (with Summary)", open(annotated_path,"rb").read(), file_name="annotated_with_summary.pdf", mime="application/pdf")

        # Show summary table
        st.subheader("Summary (short)")
        import pandas as pd
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            st.dataframe(df)
            st.download_button("⬇️ Download Summary CSV", df.to_csv(index=False).encode("utf-8"), file_name="summary.csv", mime="text/csv")
        else:
            st.info("No differences detected.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 Created by Ashutosh Nanaware</div>", unsafe_allow_html=True)






