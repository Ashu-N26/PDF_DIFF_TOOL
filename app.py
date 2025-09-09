# app.py
import os
import time
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from utils.pdf_diff import compare_pdfs, ensure_dirs

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXT = {".pdf"}

ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER])

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB cap

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    """
    Accepts two files: 'old_pdf' and 'new_pdf'
    Optional form flags:
      - ocr (on/off)
      - annotate_pages (on/off) -> heavy; off by default
    Returns: zipped files or single merged_report.pdf
    """
    try:
        if "old_pdf" not in request.files or "new_pdf" not in request.files:
            return jsonify({"error": "Upload both old and new PDF files (fields: old_pdf, new_pdf)."}), 400

        oldf = request.files["old_pdf"]
        newf = request.files["new_pdf"]

        def allowed(fn):
            return os.path.splitext(fn)[1].lower() in ALLOWED_EXT

        ofn = secure_filename(oldf.filename)
        nfn = secure_filename(newf.filename)
        if not allowed(ofn) or not allowed(nfn):
            return jsonify({"error": "Only PDF files are allowed."}), 400

        ts = int(time.time())
        old_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{ts}_OLD_{ofn}")
        new_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{ts}_NEW_{nfn}")

        oldf.save(old_path)
        newf.save(new_path)

        # parse options
        do_ocr = request.form.get("ocr", "off") == "on"
        annotate_pages = request.form.get("annotate_pages", "off") == "on"

        # call compare routine
        result = compare_pdfs(
            old_path,
            new_path,
            app.config["OUTPUT_FOLDER"],
            do_ocr=do_ocr,
            enable_page_annotations=annotate_pages,
            prefix=f"{ts}_"
        )

        # result contains paths to generated files
        return jsonify({"success": True, "outputs": result})

    except Exception as e:
        # Return useful traceback info to logs while giving user-friendly message
        app.logger.exception("Compare failed")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == "__main__":
    # local dev: debug on
    app.run(host="0.0.0.0", port=5000, debug=True)










