import streamlit as st
from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf
import os

st.set_page_config(page_title="PDF Diff Tool", layout="wide")

st.markdown("<h2 style='text-align: center;'>PDF Diff Tool</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Ashutosh Nanaware</p>", unsafe_allow_html=True)

# Upload PDFs
col1, col2 = st.columns(2)
with col1:
    old_pdf = st.file_uploader("Upload OLD PDF", type="pdf")
with col2:
    new_pdf = st.file_uploader("Upload NEW PDF", type="pdf")

if old_pdf and new_pdf:
    st.success("Both PDFs uploaded successfully ✅")

    if st.button("🔍 Compare PDFs"):
        with st.spinner("Processing... Please wait"):
            annotated_path = "annotated_diff.pdf"
            side_by_side_path = "side_by_side_diff.pdf"

            generate_annotated_pdf(old_pdf, new_pdf, annotated_path)
            generate_side_by_side_pdf(old_pdf, new_pdf, side_by_side_path)

        st.success("Comparison complete ✅")

        # Show preview
        st.subheader("📑 Preview (Side by Side)")
        st.markdown("Old (Left) vs New (Right)")
        st.download_button("⬇️ Download Annotated PDF", data=open(annotated_path, "rb"), file_name="annotated_diff.pdf")
        st.download_button("⬇️ Download Side by Side PDF", data=open(side_by_side_path, "rb"), file_name="side_by_side_diff.pdf")



