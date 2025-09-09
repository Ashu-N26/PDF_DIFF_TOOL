# Dockerfile
FROM python:3.11-slim

# install system deps: poppler (pdf->image), tesseract (OCR), build essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        tesseract-ocr \
        libtesseract-dev \
        pkg-config \
        libjpeg-dev \
        zlib1g-dev \
        poppler-utils \
        poppler-data \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# copy repo
COPY . /app

# create folders
RUN mkdir -p uploads output

# expose port (Render uses $PORT at runtime)
ENV PORT 10000

# Use gunicorn with single worker and larger timeout (avoid worker timeout on long comparisons)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--threads", "2", "--timeout", "300"]
