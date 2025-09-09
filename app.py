# app.py
import os
import sys
import time
import json
import logging
import subprocess
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.pdf_diff import ensure_dirs, generate_sample_pdfs

# Config
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXT = {".pdf"}
MAX_FILE_BYTES = 150 * 1024 * 1024  # 150 MB file limit
SUBPROCESS_TIMEOUT = 240  # seconds (adjust if needed)

ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER, "tmp", "sample"])

app = Flask(__name__)
app.secret_key = "pdfdiff-secret"  # change for production
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB global cap

# generate tiny demo PDFs if missing
generate_sample_pdfs("sample/old_demo.pdf", "sample/new_demo.pdf")

logging.basicConfig(level=logging.INFO)

def allowed(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

@app.route("/")
def index():
    return render_template("index.html", sample_old="sample/old_demo.pdf", sample_new="sample/new_demo.pdf")

@app.route("/compare", methods=["POST"])
def compare():
    use_sample = request.form.get("use_sample") == "on"
    if use_sample:
        old_path = os.path.abspath("sample/old_demo.pdf")
        new_path = os.path.abspath("sample/new_demo.pdf")
    else:
        old_file = request.files.get("old_pdf")
        new_file = request.files.get("new_pdf")
        if not old_file or not new_file:
            flash("Please upload both Old and New PDF files (or choose 'Use sample').")
            return redirect(url_for("index"))
        if not allowed(old_file.filename) or not allowed(new_file.filename):
            flash("Only .pdf files are allowed.")
            return redirect(url_for("index"))

        t = int(time.time())
        old_name = f"{t}_old_{secure_filename(old_file.filename)}"
        new_name = f"{t}_new_{secure_filename(new_file.filename)}"
        old_path = os.path.join(app.config["UPLOAD_FOLDER"], old_name)
        new_path = os.path.join(app.config["UPLOAD_FOLDER"], new_name)
        old_file.save(old_path)
        new_file.save(new_path)

        # enforce file size limit
        if os.path.getsize(old_path) > MAX_FILE_BYTES or os.path.getsize(new_path) > MAX_FILE_BYTES:
            os.remove(old_path)
            os.remove(new_path)
            flash(f"One of the uploads exceeds max allowed size ({MAX_FILE_BYTES // (1024*1024)} MB).")
            return redirect(url_for("index"))

    # call subprocess worker (this isolates crashes)
    prefix = f"{int(time.time())}_"
    worker_script = os.path.join("utils", "worker_compare.py")
    cmd = [sys.executable, worker_script, old_path, new_path, app.config["OUTPUT_FOLDER"], prefix]

    logging.info("Starting compare subprocess: %s", cmd)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT)
    except subprocess.TimeoutExpired:
        logging.exception("Compare subprocess timed out")
        flash("Comparison timed out (too large or slow). Try smaller PDFs or increase timeout/plan.")
        return redirect(url_for("index"))
    except Exception as e:
        logging.exception("Failed to run compare subprocess")
        flash(f"Failed to run compare: {e}")
        return redirect(url_for("index"))

    if proc.returncode != 0:
        logging.error("Compare subprocess failed; stdout=%s stderr=%s", proc.stdout, proc.stderr)
        # attempt to read outputs.json to show useful error, else generic
        outputs_json = os.path.join(app.config["OUTPUT_FOLDER"], f"{prefix}outputs.json")
        if os.path.exists(outputs_json):
            try:
                with open(outputs_json, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    flash("Compare failed: " + data.get("error", "unknown"))
            except Exception:
                flash("Compare failed (see server logs).")
        else:
            flash("Compare failed (see server logs).")
        return redirect(url_for("index"))

    # read outputs JSON (worker writes it)
    outputs_json = os.path.join(app.config["OUTPUT_FOLDER"], f"{prefix}outputs.json")
    if not os.path.exists(outputs_json):
        logging.error("Outputs JSON not found at %s", outputs_json)
        flash("Internal error (no outputs). Please check logs.")
        return redirect(url_for("index"))

    with open(outputs_json, "r", encoding="utf-8") as fh:
        outputs = json.load(fh)

    # pass filenames for download
    return render_template("index.html", result=outputs, sample_old="sample/old_demo.pdf", sample_new="sample/new_demo.pdf")

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    # debug server for local testing
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)








