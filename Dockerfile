FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libfontconfig1 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ASTAP (plate solving)
RUN wget https://www.hnsky.org/astap_cli.tgz \
    && tar -xzf astap_cli.tgz \
    && mv astap_cli /usr/local/bin/astap \
    && chmod +x /usr/local/bin/astap \
    && rm astap_cli.tgz

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/detections /app/reports

# Set environment variables
ENV DISPLAY=:99
ENV QT_QPA_PLATFORM=offscreen
ENV PYTHONPATH=/app

# Expose port for web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Default command
CMD ["python", "pulse_gui.py"]
