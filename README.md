# PDF_DIFF_TOOL (Render Web)
Created by Ashutosh Nanaware

## What this repo contains
- Minimal Flask web app that compares two PDFs and highlights changes.
- Uses PyMuPDF (fitz) to extract text and map changed words to bounding boxes.
- Generates the following outputs in `output/`:
  - `annotated_old.pdf` (red boxes for removed/changed words)
  - `annotated_new.pdf` (green boxes for inserted/changed words)
  - `side_by_side.pdf` (old|new pages side-by-side)
  - `summary.pdf` (simple page summarizing changes)
  - `merged_report.pdf` (summary + annotated_new)

## Deploy to Render (quick)
1. Push these files to a GitHub repo (root of repo).
2. On Render: New → Web Service → Connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Deploy. Open the public URL and upload old/new PDFs.

## Notes and troubleshooting
- If you see an Internal Server Error on Render: check `output/last_error.txt` and Render logs.
- This implementation uses a pragmatic text-search strategy that works best for
  digital (text-based) PDFs. Scanned PDFs (images) require OCR first.
- For production, secure secret key, sanitize inputs, and add size limits.

