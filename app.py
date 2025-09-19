import os
import time
import logging
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Markup
from werkzeug.utils import secure_filename
from utils.pdf_diff import compare_pdfs, ensure_dirs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER])

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB upload cap (adjust if needed)

@app.route('/')
def index():
    return render_template('index.html', author="Ashutosh Nanaware")

@app.route('/compare', methods=['POST'])
def compare():
    if 'old_pdf' not in request.files or 'new_pdf' not in request.files:
        return "Please upload both old and new PDF files.", 400

    old_f = request.files['old_pdf']
    new_f = request.files['new_pdf']
    if old_f.filename == '' or new_f.filename == '':
        return "Missing filename.", 400

    ts = time.strftime("%Y%m%d%H%M%S")
    old_fn = secure_filename(f"old_{ts}_{old_f.filename}")
    new_fn = secure_filename(f"new_{ts}_{new_f.filename}")
    old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_fn)
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_fn)
    old_f.save(old_path)
    new_f.save(new_path)
    logger.info("Saved uploaded PDFs: %s, %s", old_path, new_path)

    # Compare (this is the core function)
    try:
        outputs = compare_pdfs(old_path, new_path, app.config['OUTPUT_FOLDER'], prefix=f"{ts}_")
    except Exception as e:
        logger.exception("Comparison failed")
        return f"Internal server error while comparing PDFs: {e}", 500

    # outputs contains keys: side_by_side_html, summary_txt, changed_pages (list)
    side_html_path = outputs.get('side_by_side_html')
    summary_path = outputs.get('summary_txt')
    changed_pages = outputs.get('changed_pages', [])

    # Read side_by_side HTML and show inside a template
    side_html_content = ""
    if side_html_path and os.path.exists(side_html_path):
        with open(side_html_path, 'r', encoding='utf-8') as fh:
            side_html_content = fh.read()

    return render_template('result.html',
                           side_by_side=Markup(side_html_content),
                           summary_path=os.path.basename(summary_path) if summary_path else None,
                           changed_pages=changed_pages,
                           author="Ashutosh Nanaware")

@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))












