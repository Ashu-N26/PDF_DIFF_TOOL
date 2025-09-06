import os
import time
from flask import Flask, render_template, request, send_from_directory, jsonify
from utils.pdf_diff import compare_pdfs, ensure_dirs, generate_sample_pdfs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
ensure_dirs([UPLOAD_FOLDER, OUTPUT_FOLDER])

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    OUTPUT_FOLDER=OUTPUT_FOLDER,
    MAX_CONTENT_LENGTH=200 * 1024 * 1024
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    # expects form fields 'oldpdf' and 'newpdf'
    old = request.files.get("oldpdf")
    new = request.files.get("newpdf")
    if not old or not new:
        return jsonify({"error": "Both files required"}), 400

    ts = int(time.time())
    old_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{ts}_old.pdf")
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{ts}_new.pdf")
    old.save(old_path)
    new.save(new_path)

    try:
        # compare_pdfs returns: annotated_path, side_by_side_path, summary_pdf_path, preview_data
        annotated_path, side_by_side_path, summary_pdf_path, preview = compare_pdfs(
            old_path, new_path, app.config['OUTPUT_FOLDER'], prefix=f"{ts}_"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # preview contains side-by-side text content and word-diff info for browser preview
    response = {
        "annotated": os.path.basename(annotated_path),
        "side_by_side": os.path.basename(side_by_side_path),
        "summary": os.path.basename(summary_pdf_path),
        "preview": preview
    }
    return jsonify(response), 200

@app.route("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))






