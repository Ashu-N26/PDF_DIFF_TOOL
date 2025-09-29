#!/usr/bin/env python3
"""
app.py - Frontend CLI wrapper for the advanced PDF diff engine (pdf_diff.py)

This script provides a high-level application interface for analysts and QA teams
to run structured PDF comparisons using the rich set of diffing tools.

Features:
    - Command line interface for configuring comparisons
    - Integration with SummaryPanel, AnnotatedExport, and SideBySideExport
    - Verbose logging and graceful error handling
    - Analyst-friendly reporting output
    - Configurable thresholds for diff sensitivity

Author: Ashutosh Nanaware
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Import core diffing classes from pdf_diff.py
try:
    from pdf_diff import (
        PDFComparator,
        SummaryPanel,
        AnnotatedExporter,
        SideBySideExporter,
        DiffMode,
        DiffResult,
    )
except ImportError as e:
    print("Error: Could not import from pdf_diff.py. Make sure it is installed or in the same directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
def configure_logging(verbosity: int):
    """Setup logging based on user verbosity level."""
    log_level = logging.WARNING
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity >= 2:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser with detailed options."""
    parser = argparse.ArgumentParser(
        description="Advanced PDF Difference Analyzer - Analyst Edition",
        epilog="Example: python app.py file1.pdf file2.pdf --mode overlay --out result.pdf"
    )

    parser.add_argument("pdf_a", help="Path to the first PDF (baseline/reference)")
    parser.add_argument("pdf_b", help="Path to the second PDF (comparison/revision)")

    parser.add_argument(
        "--mode",
        choices=["text", "image", "overlay", "hybrid"],
        default="overlay",
        help="Comparison mode: text = semantic text diff, image = render & pixel diff, "
             "overlay = graphical overlay (Adobe-like), hybrid = combine text+visual"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="pdf_diff_output.pdf",
        help="Output file path for annotated or exported PDF"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate a summary report of differences"
    )

    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Export results in side-by-side comparison format"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Threshold for difference sensitivity (0.0 - 1.0, smaller is stricter)"
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum number of pages to compare (0 = all)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)"
    )

    return parser


# ---------------------------------------------------------------------------
# Application Workflow
# ---------------------------------------------------------------------------
def run_app(args):
    """Main execution workflow for the PDF diff application."""

    # Validate files
    pdf_a_path = Path(args.pdf_a)
    pdf_b_path = Path(args.pdf_b)

    if not pdf_a_path.exists() or not pdf_b_path.exists():
        logging.error("One or both PDF files do not exist.")
        sys.exit(1)

    logging.info(f"Loading PDFs: {pdf_a_path} vs {pdf_b_path}")

    # Initialize comparator
    comparator = PDFComparator(
        file_a=str(pdf_a_path),
        file_b=str(pdf_b_path),
        mode=DiffMode(args.mode),
        threshold=args.threshold,
        max_pages=args.max_pages if args.max_pages > 0 else None,
    )

    # Perform diff
    logging.info("Running PDF comparison...")
    result: DiffResult = comparator.compare()
    logging.info("Comparison complete.")

    # Generate summary
    if args.summary:
        summary = SummaryPanel(result)
        report = summary.generate()
        print("\n=== SUMMARY PANEL ===")
        print(report)
        print("=====================\n")

    # Export annotated output
    if result.has_differences():
        if args.side_by_side:
            exporter = SideBySideExporter(result)
            exporter.export(args.out)
            logging.info(f"Side-by-side diff exported to {args.out}")
        else:
            exporter = AnnotatedExporter(result)
            exporter.export(args.out)
            logging.info(f"Annotated diff exported to {args.out}")
    else:
        logging.info("No differences found; no output file created.")

    print("\nTask completed at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)

    try:
        run_app(args)
    except Exception as e:
        logging.exception("Fatal error during PDF diff process")
        sys.exit(1)


if __name__ == "__main__":
    main()
