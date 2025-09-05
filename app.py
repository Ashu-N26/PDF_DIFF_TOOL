from flask import Flask, render_template, request, send_file
import os
from pdf_diff import compare_pdfs

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    old_pdf = request.files["old_pdf"]
    new_pdf = request.files["new_pdf"]

    old_path = os.path.join(UPLOAD_FOLDER, old_pdf.filename)
    new_path = os.path.join(UPLOAD_FOLDER, new_pdf.filename)

    old_pdf.save(old_path)
    new_pdf.save(new_path)

    annotated_pdf, side_by_side_pdf, summary_pdf = compare_pdfs(old_path, new_path, OUTPUT_FOLDER)

    return render_template("result.html",
                           annotated=annotated_pdf,
                           side_by_side=side_by_side_pdf,
                           summary=summary_pdf)

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
