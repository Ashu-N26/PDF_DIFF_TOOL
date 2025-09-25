import streamlit as st
from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf
import os

# --- Page config ---
st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")

# --- Header ---
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>📑 PDF_DIFF TOOL</h1>
    """,
    unsafe_allow_html=True
)

# --- File upload section ---
col1, col2 = st.columns(2)

with col1:
    old_pdf = st.file_uploader("Upload OLD PDF", type=["pdf"])

with col2:
    new_pdf = st.file_uploader("Upload NEW PDF", type=["pdf"])

if old_pdf and new_pdf:
    st.success("✅ Both PDFs uploaded successfully")

    if st.button("🔍 Compare PDFs"):
        with st.spinner("Processing... This may take a while for large PDFs"):
            annotated_output = "annotated_diff.pdf"
            side_by_side_output = "side_by_side.pdf"

            # Save uploaded PDFs temporarily
            with open("old_temp.pdf", "wb") as f:
                f.write(old_pdf.read())
            with open("new_temp.pdf", "wb") as f:
                f.write(new_pdf.read())

            # Generate PDFs using helper functions
            generate_annotated_pdf("old_temp.pdf", "new_temp.pdf", annotated_output)
            generate_side_by_side_pdf("old_temp.pdf", "new_temp.pdf", side_by_side_output)

            st.success("✅ Comparison completed!")

            # Download buttons
            with open(annotated_output, "rb") as f:
                st.download_button(
                    "📥 Download Annotated PDF",
                    f,
                    file_name="Annotated_Diff.pdf"
                )

            with open(side_by_side_output, "rb") as f:
                st.download_button(
                    "📥 Download Side-by-Side PDF",
                    f,
                    file_name="SideBySide_Diff.pdf"
                )

# --- Footer ---
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: grey;'>
        © 2025 Created by Ashutosh Nanaware. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)





