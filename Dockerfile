FROM python:3.11-slim

# Install system deps for tesseract + basic build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip wheel
RUN pip install -r requirements.txt

# Copy app source
COPY . /app

# Streamlit run (example)
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]





