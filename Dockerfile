# ─── Stage 1: Base ────────────────────────────────────────────────────────────
FROM python:3.10-slim AS base

# System dependencies for OpenCV + DeepFace + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 2: Dependencies ────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first (smaller image), then rest
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ─── Stage 3: App ─────────────────────────────────────────────────────────────
COPY . .

# Create output directories
RUN mkdir -p data/videos models outputs logs

# Pre-download YOLOv8 weights (optional — comment out to keep image small)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
