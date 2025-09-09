import os
import io
from flask import Flask, render_template, request, send_file, jsonify
import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
from difflib import HtmlDiff
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------------------
# Utility: Extract text from PDF
# ---------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # handles scanned PDFs (returns None if no text)
    return text.strip()

# ---------------------------
# Utility: Diff text into HTML
# ---------------------------
def generate_text_diff(text1, text2):
    diff = HtmlDiff(wrapcolumn=80)
    return diff.make_table(
        text1.splitlines(), text2.splitlines(),
        fromdesc="Old PDF", todesc="New PDF"
    )

# ---------------------------
# Utility: Generate downloadable PDF report
# ---------------------------
def generate_diff_pdf(diff_html):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    textobject = c.beginText(40, 750)

    # Strip HTML tags (keep only raw text for PDF report)
    clean_text = (
        diff_html.replace("<td>", " ")
                 .replace("</td>", " ")
                 .replace("<tr>", "\n")
                 .replace("<br>", "\n")
                 .replace("&nbsp;", " ")
    )

    for line in clean_text.splitlines():
        textobject.textLine(line[:100])  # trim long lines
    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    try:
        if "pdf1" not in request.files or "pdf2" not in request.files:
            return jsonify({"error": "Please upload two PDF files"}), 400

        pdf1 = request.files["pdf1"]
        pdf2 = request.files["pdf2"]

        filename1 = secure_filename(pdf1.filename)
        filename2 = secure_filename(pdf2.filename)

        path1 = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
        path2 = os.path.join(app.config["UPLOAD_FOLDER"], filename2)

        pdf1.save(path1)
        pdf2.save(path2)

        # Extract text
        text1 = extract_text_from_pdf(path1)
        text2 = extract_text_from_pdf(path2)

        if not text1 and not text2:
            return jsonify({"error": "Both PDFs seem to be scanned images with no extractable text."}), 400

        diff_html = generate_text_diff(text1, text2)
        pdf_buffer = generate_diff_pdf(diff_html)

        return send_file(pdf_buffer, as_attachment=True, download_name="diff_report.pdf")

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)









