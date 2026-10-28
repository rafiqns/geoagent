# Use lightweight official Python runtime
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables for Python optimization & logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for geospatial libraries (GEOS, GDAL, PROJ)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and static frontend
COPY . .

# Expose FastAPI application port
EXPOSE 8088

# Healthcheck to verify service availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8088/api/health || exit 1

# Command to launch GeoAgent via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]
