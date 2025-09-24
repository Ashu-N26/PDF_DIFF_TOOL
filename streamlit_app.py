import streamlit as st
from pdf_diff import generate_annotated_pdf, generate_side_by_side_pdf
import os

st.set_page_config(page_title="PDF Comparison Tool", layout="wide")

st.title("📑 Advanced PDF Comparison Tool")
st.caption("Created by Ashutosh Nanaware")

st.markdown("""
Upload **Old PDF** and **New PDF** below to compare changes.  
- ✅ Side-by-Side Preview (Old vs New)  
- ✅ Annotated PDF (Highlights + Summary Panel)  
""")

col1, col2 = st.columns(2)
with col1:
    old_file = st.file_uploader("Upload Old PDF", type="pdf")
with col2:
    new_file = st.file_uploader("Upload New PDF", type="pdf")

if old_file and new_file:
    old_path = os.path.join("old.pdf")
    new_path = os.path.join("new.pdf")

    with open(old_path, "wb") as f:
        f.write(old_file.read())
    with open(new_path, "wb") as f:
        f.write(new_file.read())

    st.success("✅ Files uploaded successfully!")

    # Preview Options
    st.subheader("🔍 Choose Comparison Mode")
    option = st.radio("Select View:", ["Side-by-Side Preview", "Annotated PDF with Summary"])

    if st.button("▶️ Generate Comparison"):
        if option == "Side-by-Side Preview":
            output_path = "side_by_side.pdf"
            generate_side_by_side_pdf(old_path, new_path, output_path)
            st.success("✅ Side-by-Side PDF Generated")
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download Side-by-Side PDF", f, file_name="SideBySide.pdf")

        elif option == "Annotated PDF with Summary":
            output_path = "annotated.pdf"
            generate_annotated_pdf(old_path, new_path, output_path)
            st.success("✅ Annotated PDF Generated")
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download Annotated PDF", f, file_name="Annotated.pdf")



