# app.py
import os
import time
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from pdf_diff import compare_pdfs

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "pdfdiffsecret")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    if "old_pdf" not in request.files or "new_pdf" not in request.files:
        flash("Please upload both old and new PDF files.")
        return redirect(url_for("index"))

    old_file = request.files["old_pdf"]
    new_file = request.files["new_pdf"]
    if old_file.filename == "" or new_file.filename == "":
        flash("Please choose files to upload.")
        return redirect(url_for("index"))

    ts = int(time.time())
    old_path = os.path.join(UPLOAD_FOLDER, f"old_{ts}.pdf")
    new_path = os.path.join(UPLOAD_FOLDER, f"new_{ts}.pdf")
    old_file.save(old_path)
    new_file.save(new_path)

    try:
        annotated, side_by_side, summary, merged = compare_pdfs(old_path, new_path, OUTPUT_FOLDER, prefix=f"{ts}_")
    except Exception as e:
        flash(f"Error while processing PDFs: {e}")
        return redirect(url_for("index"))

    return render_template("result.html",
                           annotated=annotated,
                           side_by_side=side_by_side,
                           summary=summary,
                           merged=merged)

@app.route("/download/<path:filename>")
def download(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        flash("File not found.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    # local debug only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

