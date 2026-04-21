"""
embedder.py
DeepFace ArcFace embedding extractor.
Converts face crops to 512-d identity vectors for re-ID matching.
"""

import numpy as np
import cv2
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class FaceEmbedder:
    """
    Extracts 512-dimensional ArcFace embeddings from face crops.
    Handles low-quality / small crops gracefully.
    """

    def __init__(self):
        # Lazy-import to avoid slow TF init until needed
        from deepface import DeepFace
        self.DeepFace    = DeepFace
        self.model_name  = config.EMBEDDING_MODEL
        self.dim         = config.EMBEDDING_DIM
        self._warmup()
        print(f"[Embedder] DeepFace {self.model_name} embedder ready (dim={self.dim}).")

    def _warmup(self):
        """Pre-load model weights so first real call isn't slow."""
        try:
            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            self.get_embedding(dummy)
        except Exception:
            pass  # warmup failure is non-fatal

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        """
        Parameters
        ----------
        face_crop : np.ndarray  BGR face crop (any size ≥ 20×20)

        Returns
        -------
        embedding : np.ndarray shape (512,) or None if extraction failed
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # Resize to minimum acceptable size for ArcFace
        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            return None
        if h < 112 or w < 112:
            face_crop = cv2.resize(face_crop, (112, 112))

        try:
            result = self.DeepFace.represent(
                img_path=face_crop,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip",   # bbox already known
                align=True
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            # L2-normalise for cosine similarity via dot product
            norm = np.linalg.norm(emb)
            if norm < 1e-9:
                return None
            return emb / norm

        except Exception as e:
            return None

    def batch_embed(self, crops: list) -> list:
        """Embed a list of crops; returns list of (embedding | None)."""
        return [self.get_embedding(c) for c in crops]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        return float(np.dot(a, b))

    @staticmethod
    def mean_embedding(embeddings: list) -> np.ndarray:
        """Compute L2-normalised mean of a list of embeddings."""
        valid = [e for e in embeddings if e is not None]
        if not valid:
            return None
        mean = np.mean(valid, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-9 else mean
