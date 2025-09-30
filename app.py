import os
import io
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# Import our advanced PDF diff engine
from pdf_diff import (
    PDFComparator,
    DiffResult,
    SummaryStats,
    AnnotatedPDFExporter,
    SideBySideExporter
)

# ------------------------------
# Utility Functions
# ------------------------------

def save_uploaded_file(uploaded_file, suffix=".pdf") -> str:
    """Save uploaded file to a temporary path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def render_summary_panel(stats: SummaryStats):
    """Render summary panel with statistics and metrics."""
    st.subheader("📊 Summary Panel")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Pages", stats.total_pages)
    col2.metric("Pages Changed", stats.pages_changed)
    col3.metric("Glyph Differences", stats.glyph_diffs)
    col4.metric("Layout Adjustments", stats.layout_changes)

    st.write("#### Page-by-Page Diff Stats")
    for pg, detail in stats.page_details.items():
        st.markdown(f"- **Page {pg}**: {detail}")


def render_side_by_side(diff_result: DiffResult):
    """Render side-by-side comparison view."""
    st.subheader("🖼️ Side-by-Side Comparison")

    try:
        exporter = SideBySideExporter(diff_result)
        side_by_side_pdf = exporter.export()

        st.download_button(
            label="⬇️ Download Side-by-Side PDF",
            data=side_by_side_pdf,
            file_name="side_by_side_comparison.pdf",
            mime="application/pdf"
        )

        st.success("Side-by-side PDF generated successfully!")
    except Exception as e:
        st.error(f"Error generating side-by-side PDF: {e}")


def render_annotated(diff_result: DiffResult):
    """Render annotated PDF export view."""
    st.subheader("✏️ Annotated Export")

    try:
        exporter = AnnotatedPDFExporter(diff_result)
        annotated_pdf = exporter.export()

        st.download_button(
            label="⬇️ Download Annotated PDF",
            data=annotated_pdf,
            file_name="annotated_diff.pdf",
            mime="application/pdf"
        )

        st.success("Annotated PDF generated successfully!")
    except Exception as e:
        st.error(f"Error generating annotated PDF: {e}")


# ------------------------------
# Streamlit App Layout
# ------------------------------

def main():
    st.set_page_config(page_title="Advanced PDF Diff Tool", layout="wide")
    st.title("📑 Advanced PDF Diff Tool")
    st.caption("Compare two PDFs with glyph-level precision, layout normalization, and Adobe-like rendering")

    with st.sidebar:
        st.header("⚙️ Upload PDFs")
        file1 = st.file_uploader("Upload First PDF", type=["pdf"])
        file2 = st.file_uploader("Upload Second PDF", type=["pdf"])

        normalize_layout = st.checkbox("Normalize Layout", value=True)
        per_char_diff = st.checkbox("Enable Per-Character Glyph Diffs", value=True)
        resize_pages = st.checkbox("Resize Pages Before Diff", value=True)

        st.markdown("---")
        st.markdown("**Export Options**")
        want_side_by_side = st.checkbox("Generate Side-by-Side PDF", value=True)
        want_annotated = st.checkbox("Generate Annotated PDF", value=True)

    if file1 and file2:
        path1 = save_uploaded_file(file1)
        path2 = save_uploaded_file(file2)

        st.info("🔍 Processing PDFs...")
        try:
            comparator = PDFComparator(
                normalize_layout=normalize_layout,
                per_char=per_char_diff,
                resize_pages=resize_pages
            )

            diff_result: DiffResult = comparator.compare(path1, path2)

            # Tabs for navigation
            tab1, tab2, tab3 = st.tabs(["Summary Panel", "Side-by-Side Export", "Annotated Export"])

            with tab1:
                render_summary_panel(diff_result.stats)

            with tab2:
                if want_side_by_side:
                    render_side_by_side(diff_result)
                else:
                    st.info("Side-by-Side export is disabled in options.")

            with tab3:
                if want_annotated:
                    render_annotated(diff_result)
                else:
                    st.info("Annotated export is disabled in options.")

        except Exception as e:
            st.error(f"❌ Comparison failed: {e}")
    else:
        st.warning("Please upload two PDF files to start comparison.")


# ------------------------------
# Entrypoint
# ------------------------------

if __name__ == "__main__":
    main()

