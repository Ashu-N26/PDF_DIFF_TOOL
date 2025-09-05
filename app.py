from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from utils.pdf_diff import compare_pdfs, ensure_dirs, generate_sample_pdfs
from datetime import datetime

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "output")
ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'dev-secret-key'  # change in production

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    # accept two files pdf1 (old) and pdf2 (new)
    if 'pdf1' not in request.files or 'pdf2' not in request.files:
        flash('Both PDF files are required (old and new).')
        return redirect(url_for('index'))

    f1 = request.files['pdf1']
    f2 = request.files['pdf2']

    if f1.filename == '' or f2.filename == '':
        flash('Both PDF files are required (old and new).')
        return redirect(url_for('index'))

    fname1 = secure_filename(f1.filename)
    fname2 = secure_filename(f2.filename)
    t = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    old_path = os.path.join(app.config['UPLOAD_FOLDER'], f"old_{t}_{fname1}")
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], f"new_{t}_{fname2}")
    f1.save(old_path)
    f2.save(new_path)

    # run comparison
    try:
        outputs = compare_pdfs(old_path, new_path, app.config['OUTPUT_FOLDER'])
    except Exception as e:
        # For debugging, write error to a log file and show a friendly message
        with open(os.path.join(app.config['OUTPUT_FOLDER'], 'last_error.txt'), 'w') as fh:
            fh.write(str(e))
        flash('Server error while comparing PDFs. Check output/last_error.txt for details.')
        return redirect(url_for('index'))

    return render_template('result.html', outputs=outputs)

@app.route('/sample')
def sample():
    # Generate two sample PDFs and redirect to index with info
    sample_dir = os.path.join(os.getcwd(), "sample_pdfs")
    os.makedirs(sample_dir, exist_ok=True)
    a,b = generate_sample_pdfs(sample_dir)
    flash('Sample PDFs generated. Use Upload -> choose these sample files from sample_pdfs folder.')
    return redirect(url_for('index'))

@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # local dev server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)



