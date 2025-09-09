# PDF_DIFF_TOOL

Files in repo:
- app.py (Flask web app)
- utils/pdf_diff.py (compare + annotate logic)
- templates/index.html
- static/style.css
- requirements.txt
- .render.yaml (Render deploy config)

## Local run
1. Create venv & activate:
   python3 -m venv venv
   source venv/bin/activate
2. Install:
   pip install -r requirements.txt
3. Run:
   python app.py
4. Open http://localhost:5000

## Deploy on Render
1. Create a new Web Service -> Connect to this GitHub repo.
2. Render will run `pip install -r requirements.txt` (as set in .render.yaml)
3. In service settings set start command (if not using .render.yaml):
   `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300`
4. Upload PDFs via the UI and wait for output links.

## Notes & Troubleshooting
- If you need OCR for scanned PDFs, install `tesseract` and add `pytesseract` & `pdf2image` to requirements. OCR requires system packages (poppler + tesseract).
- Worker timeouts: increase the gunicorn timeout in the start command if you compare very large PDFs.
- If memory issues appear on Render, reduce the page render scale in `utils/pdf_diff.py` (`scale=1.5`) or upgrade plan.

Created by Ashutosh Nanaware


