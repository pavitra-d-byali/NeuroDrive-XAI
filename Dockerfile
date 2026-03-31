FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . /app/

# Expose API Port
EXPOSE 8000

# Start Uvicorn Server
CMD ["uvicorn", "deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
