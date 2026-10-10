# Use minimal Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system runtime libraries required by OpenCV & Video4Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leveraging Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONNX models, source code, and data
COPY . .

# Ensure storage directories exist inside the container
RUN mkdir -p recordings data/residents

# Expose backend API and streaming port
EXPOSE 8000

# Launch Uvicorn with single worker to prevent duplicate camera descriptor locks
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]