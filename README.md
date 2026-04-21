# 🎣 Fisherman Facial Recognition & Working Hour Estimation

An end-to-end video analytics pipeline for multi-person facial recognition, re-identification, and automated working hour estimation from surveillance footage.

---

## 📋 Overview

This system processes surveillance video footage to:
- **Detect** faces using YOLOv8
- **Track** individuals across frames using DeepSORT
- **Re-identify** persons after occlusion or re-entry using ArcFace/Facenet embeddings
- **Estimate working hours** for each tracked individual
- **Visualize results** through an interactive Streamlit dashboard

Originally designed for fisherman workforce monitoring at docking stations, the pipeline generalizes to any surveillance scenario requiring attendance and activity tracking.

---

## 🏗️ Architecture

```
Input Video
     │
     ▼
┌─────────────┐
│ YOLOv8 Face │  ← Detects face bounding boxes per frame
│  Detector   │
└──────┬──────┘
       │ boxes + crops
       ▼
┌─────────────┐
│  DeepSORT   │  ← Assigns consistent track IDs across frames
│   Tracker   │
└──────┬──────┘
       │ track IDs + bboxes
       ▼
┌─────────────┐
│  ArcFace /  │  ← Extracts 512-d / 128-d identity embeddings
│   Facenet   │
│  Embedder   │
└──────┬──────┘
       │ embeddings
       ▼
┌─────────────┐
│   Re-ID     │  ← Cosine similarity matching against gallery
│   Engine    │  ← Prevents identity switches on re-entry
└──────┬──────┘
       │ person labels
       ▼
┌─────────────┐
│   Working   │  ← Accumulates time per identity
│Hour Estimator│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Streamlit  │  ← Interactive dashboard with charts + downloads
│  Dashboard  │
└─────────────┘
```

---

## ✨ Key Features

- **Multi-person tracking** — handles 500+ unique individuals per video
- **Re-identification** — recognizes persons after occlusion or scene re-entry
- **Identity switch prevention** — embedding-based drift detection reduces false merges
- **Working hour estimation** — per-person timestamps with HH:MM:SS precision
- **Interactive dashboard** — bar charts, Gantt timeline, hourly heatmap, CSV/video downloads
- **GPU acceleration** — YOLO runs on CUDA for faster processing
- **Resumable sessions** — identity gallery persists across runs

---

## 📊 Results

Tested on a 16-minute street surveillance video (640×360, 30 FPS):

| Metric | Value |
|--------|-------|
| Unique individuals tracked | 610 |
| Total frames processed | 4,905 |
| Detection model | YOLOv8n-face |
| Embedding model | ArcFace (512-d) |
| Tracking algorithm | DeepSORT |
| Re-ID threshold | 0.72 cosine similarity |

---

## 🖥️ Dashboard Screenshots

The Streamlit dashboard provides:
- **Results Dashboard** — KPI metrics, working hours bar chart, data table
- **Presence Timeline** — Gantt-style per-person activity timeline  
- **Hourly Heatmap** — activity intensity by hour of day
- **Download Center** — CSV reports and annotated video export
- **Embedding Analysis** — 2D UMAP cluster visualization of identity embeddings

---

## 🚀 Installation

### Prerequisites
- Python 3.10
- NVIDIA GPU (optional but recommended)
- Anaconda / Miniconda

### Setup

```bash
# Clone the repository
git clone https://github.com/genocide-dj/Fisherman-Tracker.git
cd Fisherman-Tracker

# Create conda environment
conda create -n fisherman python=3.10
conda activate fisherman

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Download Models

The YOLOv8 face model is required:

```bash
# Create models directory
mkdir models

# Download yolov8n-face.pt from:
# https://github.com/akanametov/yolo-face/releases
# Place it at: models/yolov8n-face.pt
```

ArcFace weights download automatically on first run via DeepFace.

---

## 📖 Usage

### Run the Pipeline

```bash
# Process a video file
python run.py --video data/videos/your_video.mp4

# With options
python run.py --video data/videos/your_video.mp4 --no-save-video
```

### Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Pipeline Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video |
| `--no-save-video` | False | Skip saving annotated output |
| `--load-gallery` | False | Resume from previous session |
| `--no-preview` | False | Disable live preview window |

---

## ⚙️ Configuration

All parameters are in `config.py`:

```python
# Detection
YOLO_CONF_THRESH     = 0.45   # Detection confidence threshold
YOLO_MIN_FACE_SIZE   = 15     # Minimum face size in pixels

# Re-ID
REID_SIMILARITY_THRESH    = 0.72   # Cosine similarity threshold
REID_GALLERY_MAX_PER_ID   = 50    # Max embeddings stored per person
EMBEDDING_MODEL           = "Facenet"  # ArcFace or Facenet

# Tracking
DEEPSORT_MAX_AGE     = 30    # Frames before track is dropped
FRAME_SKIP           = 6     # Process every Nth frame
```

---

## 📁 Project Structure

```
Fisherman-Tracker/
├── src/
│   ├── pipeline.py      # Main orchestrator
│   ├── detector.py      # YOLOv8 face detection
│   ├── tracker.py       # DeepSORT tracking
│   ├── embedder.py      # ArcFace/Facenet embedding
│   ├── reid.py          # Re-identification engine
│   ├── timer.py         # Working hour estimation
│   └── visualizer.py    # Plot generation
├── app.py               # Streamlit dashboard
├── run.py               # CLI entry point
├── config.py            # Central configuration
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Face Detection | YOLOv8n-face (Ultralytics) |
| Multi-Object Tracking | DeepSORT |
| Face Embedding | ArcFace / Facenet (DeepFace) |
| Deep Learning | PyTorch + TensorFlow |
| Dashboard | Streamlit + Plotly |
| Data Processing | Pandas + NumPy |
| Computer Vision | OpenCV |

---

## 🔮 Future Work

- [ ] Real-time webcam support
- [ ] Multi-camera identity merging
- [ ] FastAPI REST endpoint for video upload
- [ ] PDF attendance report export
- [ ] Anomaly detection (absent workers, overtime flags)
- [ ] Docker containerization for deployment
- [ ] Lightweight MobileNet embedding for edge devices

---

## 👤 Author

**Aman Kumar** — [github.com/genocide-dj](https://github.com/genocide-dj)

---

## 📄 License

MIT License — free to use, modify, and distribute.
