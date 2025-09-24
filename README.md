# PDF_DIFF_TOOL (Token-level highlighting)
Created by Ashutosh Nanaware

## What it does
- Token-level inline highlights in new PDF:
  - Inserted tokens → Green highlight (50% opacity)
  - Changed tokens → Red highlight (50% opacity)
  - Removed tokens → NOT highlighted; listed in appended Summary panel
- Side-by-side PDF (Old | New)
- Summary panel appended to annotated NEW PDF

## Run locally
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py

