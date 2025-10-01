# Dockerfile (for Render or any Docker host)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Force rebuild cache when needed (bump the build-arg)
ARG CACHEBUST=1

# Install system packages required for Tesseract, Poppler, compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    poppler-utils \
    tesseract-ocr \
    libleptonica-dev \
    libtesseract-dev \
    libjpeg-dev \
    zlib1g-dev \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to use Docker layer caching
COPY requirements.txt /app/requirements.txt

# Uninstall any wrongly installed 'fitz' package (safe guard), upgrade pip then install deps
RUN pip install --upgrade pip wheel setuptools && \
    pip uninstall -y fitz || true && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Expose Streamlit port expected on Render (or default 8501 if using Streamlit Cloud)
EXPOSE 10000

# Start Streamlit app on port 10000 (Render maps that port)
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]







