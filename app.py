import streamlit as st
import os
from pdf_diff import (
    generate_annotated_pdf,
    generate_side_by_side_pdf,
)

# -------------------- Streamlit App -------------------- #
st.set_page_config(page_title="PDF_DIFF TOOL", layout="wide")

st.title("📑 PDF_DIFF TOOL")

st.markdown("""
Upload **two PDF files** (Old & New) to compare.
- **Side-by-Side Preview** → Old vs New for quick review.
- **Annotated PDF with Summary** → Highlights insertions (🟢), changes (🔴), and removed text.
""")

# File Upload
col1, col2 = st.columns(2)
with col1:
    old_pdf = st.file_uploader("Upload Old PDF", type=["pdf"], key="old")
with col2:
    new_pdf = st.file_uploader("Upload New PDF", type=["pdf"], key="new")

# Process Button
if old_pdf and new_pdf:
    if st.button("🔍 Run Comparison"):
        with st.spinner("Processing and analyzing differences..."):
            old_path = os.path.join("uploads", old_pdf.name)
            new_path = os.path.join("uploads", new_pdf.name)

            os.makedirs("uploads", exist_ok=True)

            with open(old_path, "wb") as f:
                f.write(old_pdf.read())
            with open(new_path, "wb") as f:
                f.write(new_pdf.read())

            # --- Side-by-Side PDF ---
            side_by_side_path = generate_side_by_side_pdf(old_path, new_path)

            # --- Annotated PDF ---
            annotated_path = generate_annotated_pdf(old_path, new_path)

        st.success("✅ Comparison completed!")

        # Preview + Download Buttons
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📘 Side-by-Side Preview")
            with open(side_by_side_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Side-by-Side PDF",
                    f,
                    file_name="side_by_side.pdf",
                    mime="application/pdf"
                )
        with col4:
            st.subheader("📕 Annotated PDF with Summary")
            with open(annotated_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Annotated Comparison PDF",
                    f,
                    file_name="annotated_comparison.pdf",
                    mime="application/pdf"
                )
else:
    st.info("⬆️ Please upload both Old and New PDFs to continue.")







