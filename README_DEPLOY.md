# PDF_DIFF_TOOL — Deploy notes (Render Docker)

Use the provided Dockerfile and push this repo to GitHub. Then, in Render:
- Create New -> Web Service
- Connect GitHub repo and select branch `main`
- Environment: Docker
- Leave default build command (Render will build Dockerfile)
- Start command is defined in Dockerfile (gunicorn app:app ...)

Important:
- The Docker image installs `poppler-utils` (provides `pdftoppm`) and `tesseract-ocr` (for OCR).
- If you plan to enable `page-level annotations` in the web UI, make sure your Render instance has >=1GB RAM. Heavy operations can be slow and CPU/memory hungry.
- If you see `Internal Server Error` after clicking compare:
  1. Check the Render service logs. Look for `Exception` or `Memory` signs.
  2. If OCR fails: ensure tesseract is installed (Dockerfile does that). If you deployed without Docker, install tesseract/poppler on the host.
- By default the app produces:
  - `output/{ts}_annotated_text_report.pdf` (textual summary with colored lines)
  - `output/{ts}_side_by_side.pdf` (page images)
  - `output/{ts}_merged_report.pdf` (summary + side-by-side)
