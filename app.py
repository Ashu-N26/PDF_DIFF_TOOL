import streamlit as st
from pdf_diff import (
    compare_pdfs,
    save_side_by_side_pdf,
    save_annotated_pdf,
)

# =========================
# Streamlit UI Setup
# =========================
st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")

# Header
st.title("📄 PDF_DIFF TOOL")
st.caption("Created by Ashutosh Nanaware")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    old_pdf = st.file_uploader("Upload Old PDF", type=["pdf"])
with col2:
    new_pdf = st.file_uploader("Upload New PDF", type=["pdf"])

# Comparison button
if old_pdf and new_pdf:
    if st.button("🔍 Compare PDFs"):
        with st.spinner("Processing and comparing... Please wait ⏳"):
            # Run diff
            differences, annotated_pdf = compare_pdfs(old_pdf, new_pdf)

        # =========================
        # Results Section
        # =========================
        st.success("✅ Comparison completed!")

        # Show summary
        st.subheader("📊 Differences Summary")
        if differences:
            for diff in differences:
                st.write(diff)
        else:
            st.write("No differences found.")

        # -------------------------
        # Side-by-Side PDF
        # -------------------------
        st.subheader("📑 Side-by-Side Preview")
        side_by_side_pdf = save_side_by_side_pdf(old_pdf, new_pdf)

        if side_by_side_pdf:
            with open(side_by_side_pdf, "rb") as f:
                st.download_button(
                    label="⬇️ Download Side-by-Side PDF",
                    data=f,
                    file_name="side_by_side_comparison.pdf",
                    mime="application/pdf"
                )

        # -------------------------
        # Annotated PDF
        # -------------------------
        st.subheader("🖍 Annotated Comparison PDF")
        annotated_output = save_annotated_pdf(annotated_pdf, differences)

        if annotated_output:
            with open(annotated_output, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated PDF",
                    data=f,
                    file_name="annotated_comparison.pdf",
                    mime="application/pdf"
                )








