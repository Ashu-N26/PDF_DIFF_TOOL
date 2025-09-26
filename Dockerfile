# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system packages including tesseract for OCR and fonts/libs for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    libopenjp2-7-dev \
    ghostscript \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]

