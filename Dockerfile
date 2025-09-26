# Dockerfile (use this exact content)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system packages needed by PyMuPDF, Tesseract OCR, and Streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Expose port (Render sets $PORT at runtime; we expose an arbitrary port)
EXPOSE 8000

# Streamlit envs to ensure proper binding and headless mode
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# Use sh -c so $PORT expands correctly at runtime
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8000} --server.address=0.0.0.0"]




