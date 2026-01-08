# Use a lightweight Python base
FROM python:3.10-slim

# Keep Python from buffering stdout (so logs show up immediately)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (needed for some hardware libs)
RUN apt-get update && apt-get install -y \
    i2c-tools \
    libgpiod2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files first (better caching)
COPY shared/pyproject.toml shared/pyproject.toml
COPY robot/requirements.txt robot/requirements.txt
COPY operator/requirements.txt operator/requirements.txt
COPY ui/requirements.txt ui/requirements.txt
COPY vision/requirements.txt vision/requirements.txt

# Install dependencies
# Note: We merge them to create one "super-environment" for simplicity
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r robot/requirements.txt && \
    pip install -r operator/requirements.txt && \
    pip install -r ui/requirements.txt && \
    pip install -r vision/requirements.txt

# Copy the rest of the code
COPY . .

# Default command (can be overridden in docker-compose)
CMD ["python3"]