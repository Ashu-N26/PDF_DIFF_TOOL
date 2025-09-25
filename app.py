import streamlit as st
from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf

st.set_page_config(page_title="PDF Diff Tool", layout="wide")
st.title("📄 PDF Difference Highlighter Tool")

st.markdown(
    """
    Upload **Old** and **New** PDFs.  
    The tool will:
    - 🟥 Highlight changed text in **red** (50% opacity)  
    - 🟩 Highlight inserted text in **green** (50% opacity)  
    - 📌 Summarize removed text in a final summary panel  
    """
)

# Upload PDFs
col1, col2 = st.columns(2)
with col1:
    old_pdf = st.file_uploader("Upload Old PDF", type=["pdf"])
with col2:
    new_pdf = st.file_uploader("Upload New PDF", type=["pdf"])

if old_pdf and new_pdf:
    try:
        with st.spinner("🔍 Comparing PDFs..."):
            annotated_bytes, summary_text = generate_annotated_pdf(old_pdf, new_pdf)
            side_by_side_bytes = generate_side_by_side_pdf(old_pdf, new_pdf)

        st.success("✅ Comparison complete!")

        # Side by Side Download
        st.subheader("📊 Side-by-Side Comparison")
        st.download_button(
            "⬇️ Download Side-by-Side PDF",
            side_by_side_bytes,
            file_name="side_by_side.pdf",
            mime="application/pdf",
        )

        # Summary
        st.subheader("📝 Summary of Removed Data")
        if summary_text.strip():
            st.text_area("Removed Text", summary_text, height=200)
        else:
            st.info("No data removed.")

        # Annotated Final PDF
        st.subheader("📌 Final Annotated PDF")
        st.download_button(
            "⬇️ Download Annotated PDF",
            annotated_bytes,
            file_name="annotated_diff.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("Please upload both PDFs to start.")


