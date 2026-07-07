FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-dan \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender1

# Copy app
WORKDIR /app
COPY . .

# Install Python deps
RUN pip install -r requirements.txt

# Start server
CMD ["python3", "server.py"]

