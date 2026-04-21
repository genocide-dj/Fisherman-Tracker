"""
config.py
Central configuration for the Fisherman Facial Recognition & Working Hour Estimation pipeline.
Edit values here to tune the entire system.
"""

# ─── Video Input ──────────────────────────────────────────────────────────────
VIDEO_FPS          = 25            # Expected FPS of surveillance footage
FRAME_SKIP         = 5             # Process every Nth frame (reduce CPU load)
RESIZE_WIDTH       = 1280          # Resize input frames to this width
RESIZE_HEIGHT      = 720

# ─── YOLO Face Detector ───────────────────────────────────────────────────────
YOLO_MODEL_PATH    = "models/yolov8n-face.pt"   # Downloaded automatically on first run
YOLO_CONF_THRESH   = 0.45          # Minimum detection confidence
YOLO_IOU_THRESH    = 0.45          # NMS IoU threshold
YOLO_MIN_FACE_SIZE = 15            # Ignore detections smaller than this (px)

# ─── DeepSORT Tracker ─────────────────────────────────────────────────────────
DEEPSORT_MAX_AGE        = 60       # Frames before a lost track is deleted
DEEPSORT_N_INIT         = 3        # Frames needed to confirm a new track
DEEPSORT_NMS_MAX_OVERLAP= 1.0
DEEPSORT_MAX_COSINE_DIST= 0.4      # Re-ID appearance threshold inside DeepSORT

# ─── DeepFace Embeddings ──────────────────────────────────────────────────────
EMBEDDING_MODEL    = "ArcFace"     # Options: ArcFace, Facenet512, VGG-Face
EMBEDDING_DIM      = 512
EMBEDDING_INTERVAL = 10            # Extract embedding every N frames per track

# ─── Re-ID Re-scoring ─────────────────────────────────────────────────────────
REID_SIMILARITY_THRESH = 0.72      # Cosine similarity to accept re-ID match
REID_GALLERY_MAX_PER_ID= 50        # Max embeddings stored per known identity
REID_MIN_GALLERY_SIZE  = 3         # Need at least N embeddings before matching

# ─── Working Hour Estimator ───────────────────────────────────────────────────
MIN_PRESENCE_SECONDS   = 2.0       # Ignore presences shorter than this
MAX_GAP_SECONDS        = 30.0      # If gap > this, treat as a new work session

# ─── 2D Visualization ─────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.1
TSNE_PERPLEXITY    = 30

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR         = "outputs"
LOG_DIR            = "logs"
REPORT_CSV         = "outputs/working_hours_report.csv"
EMBED_PLOT_PATH    = "outputs/embedding_clusters.png"
ANNOTATED_VIDEO    = "outputs/annotated_output.mp4"
IDENTITY_DB_PATH   = "outputs/identity_gallery.pkl"

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_PORT     = 8501
MAX_WORKERS        = 4             # Threading for frame processing
