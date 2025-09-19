# Use a stable slim Python image
FROM python:3.10-slim

# Prevent prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install system packages required to build MuPDF/PyMuPDF and for OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      wget \
      git \
      pkg-config \
      libjpeg-dev \
      libfreetype6-dev \
      libopenjp2-7-dev \
      libharfbuzz-dev \
      libtiff5-dev \
      zlib1g-dev \
      libleptonica-dev \
      libpng-dev \
      libxcb1-dev \
      libx11-dev \
      libxext-dev \
      libssl-dev \
      poppler-utils \
      tesseract-ocr \
      libtesseract-dev \
      ca-certificates \
      curl \
      locales \
    && rm -rf /var/lib/apt/lists/*

# Set locale (optional, helpful for OCR languages)
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip + install wheelsupport
RUN python -m pip install --upgrade pip setuptools wheel

# Create app dir
WORKDIR /app

# Copy requirements file first for pip layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (this will compile PyMuPDF if wheel not available)
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . /app

# Expose port
EXPOSE 8000

# Create runtime folders
RUN mkdir -p /app/uploads /app/output

# Start using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4"]

