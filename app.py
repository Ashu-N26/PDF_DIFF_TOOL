import streamlit as st
from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf
import os

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="PDF Diff Tool", layout="wide")

st.markdown("<h2 style='text-align: center;'>📑 Advanced PDF Diff Tool</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by <b>Ashutosh Nanaware</b></p>", unsafe_allow_html=True)

# ----------------------------
# File Upload Section
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    old_pdf = st.file_uploader("Upload OLD PDF", type="pdf")

with col2:
    new_pdf = st.file_uploader("Upload NEW PDF", type="pdf")

# ----------------------------
# Action Button
# ----------------------------
if old_pdf and new_pdf:
    st.success("✅ Both PDFs uploaded successfully")

    if st.button("🔍 Compare PDFs"):
        with st.spinner("Processing... Please wait ⏳"):

            annotated_path = "annotated_diff.pdf"
            side_by_side_path = "side_by_side_diff.pdf"

            try:
                generate_annotated_pdf(old_pdf, new_pdf, annotated_path)
                # Reset pointer for re-read (since .read() consumes file stream)
                old_pdf.seek(0)
                new_pdf.seek(0)
                generate_side_by_side_pdf(old_pdf, new_pdf, side_by_side_path)

                st.success("✅ Comparison complete")

                # ----------------------------
                # Preview Section
                # ----------------------------
                st.subheader("📑 Preview: Side by Side View")
                st.markdown("Old PDF (Left) vs New PDF (Right)")

                with open(side_by_side_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Side by Side PDF",
                        data=f,
                        file_name="side_by_side_diff.pdf",
                        mime="application/pdf",
                    )

                with open(annotated_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Annotated PDF (with Highlights + Summary)",
                        data=f,
                        file_name="annotated_diff.pdf",
                        mime="application/pdf",
                    )

            except Exception as e:
                st.error(f"❌ Error while comparing PDFs: {str(e)}")

else:
    st.info("⬆️ Please upload both OLD and NEW PDF to continue.")

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 13px;">
        © 2025 Created by <b>Ashutosh Nanaware</b>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)




