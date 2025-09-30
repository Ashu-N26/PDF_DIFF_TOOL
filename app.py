import streamlit as st
import os
from pdf_diff import (
    PDFComparator,
    PDFSummaryPanel,
    PDFAnnotatedExporter,
    PDFSideBySideExporter,
)

# ===============================
# Streamlit Page Configuration
# ===============================
st.set_page_config(
    page_title="PDF_DIFF TOOL",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# Custom Styling
# ===============================
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #0e1117;
    }
    .sidebar .sidebar-content {
        background-color: #111827;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #f5f5f5;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============================
# Title
# ===============================
st.title("📑 PDF_DIFF TOOL")
st.markdown("#### Created by **Ashutosh Nanaware**")

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("⚙️ Options")
    debug = st.checkbox("Show debug state")  # removed "Created by" + tip

# ===============================
# File Upload Section
# ===============================
col1, col2 = st.columns(2)

with col1:
    old_pdf = st.file_uploader(
        "Upload OLD PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload the baseline (old) version of the PDF",
    )

with col2:
    new_pdf = st.file_uploader(
        "Upload NEW PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload the revised (new) version of the PDF",
    )

# ===============================
# Session State
# ===============================
if "results" not in st.session_state:
    st.session_state.results = None

# ===============================
# Comparison
# ===============================
if old_pdf and new_pdf:
    comparator = PDFComparator(old_pdf, new_pdf)
    results = comparator.compare()
    st.session_state.results = results

    # Display Tabs for Outputs
    tabs = st.tabs(["📊 Summary", "📝 Annotated Export", "📖 Side-by-Side"])

    # --- Tab 1: Summary Panel ---
    with tabs[0]:
        st.subheader("Comparison Summary")
        panel = PDFSummaryPanel(results)
        st.markdown(panel.render_summary(), unsafe_allow_html=True)

    # --- Tab 2: Annotated Export ---
    with tabs[1]:
        st.subheader("Annotated PDF Export")
        exporter = PDFAnnotatedExporter(results)
        annotated_path = exporter.export("annotated_diff.pdf")
        with open(annotated_path, "rb") as f:
            st.download_button(
                "⬇️ Download Annotated PDF",
                f,
                file_name="annotated_diff.pdf",
                mime="application/pdf",
            )
        st.info("This PDF highlights insertions and deletions inline.")

    # --- Tab 3: Side-by-Side Export ---
    with tabs[2]:
        st.subheader("Side-by-Side Comparison PDF")
        side_by_side = PDFSideBySideExporter(results)
        sbs_path = side_by_side.export("side_by_side_diff.pdf")
        with open(sbs_path, "rb") as f:
            st.download_button(
                "⬇️ Download Side-by-Side PDF",
                f,
                file_name="side_by_side_diff.pdf",
                mime="application/pdf",
            )
        st.info("This PDF shows OLD and NEW versions next to each other.")

else:
    st.warning("Upload both OLD and NEW PDFs to begin comparison.")

# ===============================
# Debug State (Optional)
# ===============================
if debug:
    st.subheader("🔍 Debug State")
    st.json(
        {
            "old_pdf_uploaded": bool(old_pdf),
            "new_pdf_uploaded": bool(new_pdf),
            "results_ready": st.session_state.results is not None,
        }
    )

# ===============================
# Footer
# ===============================
st.markdown(
    """
    <div style='text-align: center; padding: 20px; font-size: 13px; color: #aaa;'>
        © 2025 Created by <b>Ashutosh Nanaware</b>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)




