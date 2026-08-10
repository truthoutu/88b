# Production Dockerfile for Render & Cloud Deployment
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Create output, log, and model directories
RUN mkdir -p /app/scraped_data /app/logs /app/models

# Expose server port
EXPOSE 8000

# Launch Uvicorn server dynamically reading PORT environment variable
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
