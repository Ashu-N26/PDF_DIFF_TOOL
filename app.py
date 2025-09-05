import os
import logging
import traceback
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash

# local import
from utils.pdf_diff import compare_pdfs, ensure_dirs, generate_sample_pdfs

# --- configuration ---
BASE_DIR = os.path.abspath(os.getcwd())
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(BASE_DIR, "app.log")
ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER])

# --- logging ---
logger = logging.getLogger("pdf_diff_app")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# stream handler (appears in Render logs)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# file handler
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(formatter)
logger.addHandler(fh)

# --- flask app ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    # robust top-level try/except to capture anything
    try:
        logger.info("Compare request started")
        if 'pdf1' not in request.files or 'pdf2' not in request.files:
            flash('Both PDF files are required (old and new).')
            return redirect(url_for('index'))

        f1 = request.files['pdf1']
        f2 = request.files['pdf2']

        if f1.filename == '' or f2.filename == '':
            flash('Both PDF files are required (old and new).')
            return redirect(url_for('index'))

        # save files
        t = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        fname1 = f"old_{t}_" + os.path.basename(f1.filename)
        fname2 = f"new_{t}_" + os.path.basename(f2.filename)
        old_path = os.path.join(app.config['UPLOAD_FOLDER'], fname1)
        new_path = os.path.join(app.config['UPLOAD_FOLDER'], fname2)

        f1.save(old_path)
        f2.save(new_path)
        logger.info(f"Saved uploaded PDFs: {old_path}, {new_path}")

        # run comparison (this is the heavy operation)
        outputs = compare_pdfs(old_path, new_path, app.config['OUTPUT_FOLDER'])
        logger.info("Comparison finished successfully. Outputs: %s", outputs)

        return render_template("result.html", outputs=outputs)

    except Exception as e:
        # capture full traceback to both logs and file for easy diagnosis
        tb = traceback.format_exc()
        logger.error("Exception during /compare:\n%s", tb)

        # write last_error for convenient retrieval
        last_error_path = os.path.join(app.config['OUTPUT_FOLDER'], "last_error.txt")
        try:
            with open(last_error_path, "w", encoding="utf-8") as fh:
                fh.write(tb)
        except Exception as w:
            logger.error("Failed to write last_error.txt: %s", str(w))

        # user-visible but not the full trace (we wrote it to output/last_error.txt)
        flash("Server error while comparing PDFs. Full traceback written to output/last_error.txt and app.log.")
        return redirect(url_for('index'))

@app.route("/sample")
def sample():
    sample_dir = os.path.join(BASE_DIR, "sample_pdfs")
    os.makedirs(sample_dir, exist_ok=True)
    a,b = generate_sample_pdfs(sample_dir)
    flash(f"Generated sample PDFs in {sample_dir}")
    logger.info("Sample PDFs generated: %s, %s", a, b)
    return redirect(url_for('index'))

@app.route("/output/<path:filename>")
def download_file(filename):
    # serve generated outputs
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# friendly health route
@app.route("/_healthz")
def health():
    return "ok", 200

if __name__ == "__main__":
    # local dev - the server will show tracebacks; but Render will use gunicorn
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug)




