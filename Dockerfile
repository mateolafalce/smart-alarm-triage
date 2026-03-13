FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .
RUN pip install -e . --no-deps

# Create data directories
RUN mkdir -p data/raw data/processed models reports/figures

# Default entrypoint: training
ENTRYPOINT ["python", "scripts/train.py"]
