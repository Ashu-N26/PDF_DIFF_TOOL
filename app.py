import os
import time
import logging
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Markup
from werkzeug.utils import secure_filename
from utils.advanced_pdf_diff import compare_pdfs_advanced, ensure_dirs

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-diff")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER])

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'old_pdf' not in request.files or 'new_pdf' not in request.files:
        return "Please provide both old and new PDF files", 400

    old = request.files['old_pdf']
    new = request.files['new_pdf']
    if not old or not new:
        return "Missing files", 400

    ts = time.strftime("%Y%m%d%H%M%S")
    old_fn = secure_filename(f"old_{ts}_{old.filename}")
    new_fn = secure_filename(f"new_{ts}_{new.filename}")
    old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_fn)
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_fn)
    old.save(old_path)
    new.save(new_path)
    logger.info("Saved files: %s , %s", old_path, new_path)

    try:
        outputs = compare_pdfs_advanced(old_path, new_path, app.config['OUTPUT_FOLDER'], prefix=f"{ts}_")
    except Exception as e:
        logger.exception("Comparison failed")
        return f"Internal server error while comparing: {e}", 500

    # Render side-by-side HTML into result page
    side_html = ""
    if outputs.get('side_by_side_html') and os.path.exists(outputs['side_by_side_html']):
        with open(outputs['side_by_side_html'], 'r', encoding='utf-8') as fh:
            side_html = fh.read()

    return render_template('result.html',
                           side_by_side=Markup(side_html),
                           summary_file=os.path.basename(outputs.get('summary_txt', '')) if outputs.get('summary_txt') else None,
                           annotated_new=os.path.basename(outputs.get('annotated_new_pdf', '')) if outputs.get('annotated_new_pdf') else None,
                           merged_report=os.path.basename(outputs.get('merged_report', '')) if outputs.get('merged_report') else None,
                           changed_pages=outputs.get('changed_pages', []))

@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))













