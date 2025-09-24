import streamlit as st
from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf

# Streamlit App Title
st.set_page_config(page_title="PDF Difference Highlighter", layout="wide")
st.title("📄 PDF Difference Highlighter Tool")

st.markdown(
    """
    Upload two PDF files (Old vs New).  
    The tool will:
    - Highlight **inserted text** in 🟩 Green (50% opacity)  
    - Highlight **changed text** in 🟥 Red (50% opacity)  
    - List **removed text** in a summary panel  
    - Show **side-by-side preview** before downloading the final annotated PDF  
    """
)

# File Upload
col1, col2 = st.columns(2)
with col1:
    old_pdf = st.file_uploader("Upload Old PDF", type=["pdf"], key="old_pdf")
with col2:
    new_pdf = st.file_uploader("Upload New PDF", type=["pdf"], key="new_pdf")

# Processing
if old_pdf and new_pdf:
    try:
        with st.spinner("🔍 Comparing PDFs..."):
            # Generate annotated diff PDF and summary
            annotated_bytes, summary_text = generate_annotated_pdf(old_pdf, new_pdf)

            # Generate side-by-side comparison view
            side_by_side_bytes = generate_side_by_side_pdf(old_pdf, new_pdf)

        st.success("✅ PDF comparison complete!")

        # Side by Side Preview
        st.subheader("📊 Side-by-Side Preview")
        st.download_button(
            label="⬇️ Download Side-by-Side PDF",
            data=side_by_side_bytes,
            file_name="side_by_side_preview.pdf",
            mime="application/pdf"
        )
        st.write("You can open the preview PDF to visually check differences before final download.")

        # Summary Panel
        st.subheader("📝 Summary of Removed Data")
        if summary_text.strip():
            st.text_area("Removed Text Summary", summary_text, height=200)
        else:
            st.info("No data was removed from the Old PDF.")

        # Final Annotated PDF
        st.subheader("📌 Final Annotated PDF")
        st.download_button(
            label="⬇️ Download Annotated PDF (with highlights + summary)",
            data=annotated_bytes,
            file_name="annotated_diff.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        st.info("Check that both PDFs are valid and try again.")
else:
    st.info("Please upload **both Old and New PDFs** to start.")

