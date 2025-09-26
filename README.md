# 📄 PDF_DIFF TOOL

A web-based tool to compare **two PDFs** (Old vs New) with:
- ✅ Side-by-Side Preview
- ✅ Annotated PDF highlighting:
  - 🟢 Inserted text (green, 50% opacity)
  - 🔴 Changed text (red, 50% opacity)
  - Removed text → captured in **Summary Panel**

## 🚀 Features
- OCR support → no text is missed
- Character-level (sub-token) diff highlighting
- Proper resizing & alignment for different layouts
- Download options:
  - Side-by-Side Comparison PDF
  - Annotated Comparison PDF with Summary

## 🛠 Deployment
This repo is Dockerized for **Render** deployment.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py



