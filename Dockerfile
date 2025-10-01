# -------------------------------
# PDF_DIFF_TOOL Dockerfile for Render
# -------------------------------

# Base Python image (small + stable)
FROM python:3.11-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OCR, PDF processing and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    ghostscript \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first (to leverage Docker layer caching)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip wheel setuptools
RUN pip install -r requirements.txt

# Copy full app source code
COPY . /app

# Expose Streamlit’s default port
EXPOSE 8501

# Run Streamlit app with proper network binding for Render
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]






