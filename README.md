# PDF Diff Tool (Ashutosh Nanaware)

## What it does
- Compares an Old PDF and a New PDF.
- Produces:
  - Annotated NEW PDF (green for inserted/increased)
  - Annotated OLD PDF (red for removed/reduced)
  - Single annotated PDF with appended summary panel
  - Side-by-side PDF (old left, new right — right is annotated)
- Summary panel page appended to the annotated single PDF.

## How to deploy to Streamlit Cloud
1. Create a new GitHub repository and add these files.
2. Push to GitHub.
3. Sign in to https://streamlit.io/cloud and click "New app".
4. Connect your GitHub repo and choose branch + `streamlit_app.py` as the main file.
5. Deploy. Streamlit Cloud will install dependencies from `requirements.txt`.

## Local run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
