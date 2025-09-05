import os
from flask import Flask, render_template, request, send_from_directory
from pdf_diff import compare_pdfs

app = Flask(__name__)

# Ensure required folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/compare", methods=["POST"])
def compare():
    if "old_pdf" not in request.files or "new_pdf" not in request.files:
        return "Please upload both PDFs.", 400

    old_pdf = request.files["old_pdf"]
    new_pdf = request.files["new_pdf"]

    old_path = os.path.join("uploads", old_pdf.filename)
    new_path = os.path.join("uploads", new_pdf.filename)

    old_pdf.save(old_path)
    new_pdf.save(new_path)

    # Run comparison (outputs annotated PDFs + summary)
    output_files = compare_pdfs(old_path, new_path, "output")

    return render_template("result.html", outputs=output_files)


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory("output", filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


