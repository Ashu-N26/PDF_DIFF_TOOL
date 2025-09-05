# PDF_DIFF_TOOL (Render Web)

Created by Ashutosh Nanaware

## Deploy to Render (quick)
1. Push these files to a GitHub repo (root of repo).
2. On Render: New → Web Service → Connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Deploy. Open the public URL and upload old/new PDFs.

Outputs: annotated_new.pdf, side_by_side.pdf, summary.pdf, merged_report.pdf (in `output/`).
