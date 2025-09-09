import os
import time
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.pdf_diff import compare_pdfs, ensure_dirs, generate_sample_pdfs

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXT = {".pdf"}

ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER, "tmp", "sample"])

app = Flask(__name__)
app.secret_key = "pdfdiff-secret"  # change for prod
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

# generate demo PDFs if none present (useful for quick tests)
generate_sample_pdfs("sample/old_demo.pdf", "sample/new_demo.pdf")

@app.route("/")
def index():
    return render_template("index.html", sample_old="sample/old_demo.pdf", sample_new="sample/new_demo.pdf")

def allowed(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

@app.route("/compare", methods=["POST"])
def compare():
    # Accept files or use sample
    old_file = request.files.get("old_pdf")
    new_file = request.files.get("new_pdf")
    use_sample = request.form.get("use_sample") == "on"

    if use_sample:
        old_path = os.path.abspath("sample/old_demo.pdf")
        new_path = os.path.abspath("sample/new_demo.pdf")
    else:
        if not old_file or not new_file:
            flash("Upload both old and new PDF files (or use sample).")
            return redirect(url_for("index"))
        if not allowed(old_file.filename) or not allowed(new_file.filename):
            flash("Only .pdf allowed.")
            return redirect(url_for("index"))
        t = int(time.time())
        old_name = f"{t}_old_{secure_filename(old_file.filename)}"
        new_name = f"{t}_new_{secure_filename(new_file.filename)}"
        old_path = os.path.join(app.config["UPLOAD_FOLDER"], old_name)
        new_path = os.path.join(app.config["UPLOAD_FOLDER"], new_name)
        old_file.save(old_path)
        new_file.save(new_path)

    # run compare (synchronous). For heavy docs increase gunicorn timeout (see README)
    try:
        outputs = compare_pdfs(old_path, new_path, app.config["OUTPUT_FOLDER"])
    except Exception as e:
        app.logger.exception("Compare failed")
        flash(f"Error during compare: {e}")
        return redirect(url_for("index"))

    # outputs is dict with keys: annotated_old, annotated_new, merged, side_by_side, summary_page
    return render_template("index.html", result=outputs, sample_old="sample/old_demo.pdf", sample_new="sample/new_demo.pdf")

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    # debug server (local)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)







