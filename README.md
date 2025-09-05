# PDF_DIFF_TOOL (Render Web Version)
Created by Ashutosh Nanaware

## Deployment on Render
1. Go to [Render](https://render.com)
2. Create a new **Web Service**
3. Upload this ZIP or connect GitHub repo
4. Set **Start Command** = `python app.py`
5. Render auto-installs from `requirements.txt`

App will be available at your Render URL.

## Features
- Annotated PDF (Red = removed, Green = added)
- Side-by-Side PDF (old vs new)
- Summary panel PDF (DA/MDA, VIS, RVR changes)
