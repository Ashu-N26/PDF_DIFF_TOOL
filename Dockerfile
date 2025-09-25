# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by PyMuPDF & reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker cache use)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app source code into the container
COPY . .

# Expose Render’s dynamic port
EXPOSE 8000

# Streamlit entrypoint for Render (PORT is set by Render automatically)
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
