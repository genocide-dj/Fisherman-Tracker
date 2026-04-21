"""
pipeline.py
Main end-to-end pipeline orchestrator.

Flow per frame:
  1. Read frame → resize
  2. YOLO face detection
  3. DeepSORT tracking → track IDs
  4. Extract face crops per track
  5. DeepFace embedding → Re-ID re-scoring → person labels
  6. Update working hour estimator
  7. Annotate frame → write to output video
  8. Periodic gallery save + visualization update

Usage:
  from src.pipeline import Pipeline
  p = Pipeline()
  p.run("data/videos/surveillance.mp4")
"""

import cv2
import numpy as np
import os, sys, time, json
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from src.detector  import FaceDetector
from src.tracker   import FaceTracker
from src.embedder  import FaceEmbedder
from src.reid      import ReIDEngine
from src.timer     import WorkingHourEstimator
from src import visualizer as viz


class Pipeline:
    """
    Full video analytics pipeline for multi-person facial re-identification
    and activity-duration tracking on surveillance footage.
    """

    def __init__(self, load_gallery: bool = False):
        print("\n" + "="*60)
        print("  Fisherman Tracking Pipeline  —  Initialising")
        print("="*60)

        self.detector  = FaceDetector()
        self.tracker   = FaceTracker()
        self.embedder  = FaceEmbedder()
        self.reid      = ReIDEngine(self.embedder)
        self.timer     = WorkingHourEstimator()

        if load_gallery and os.path.exists(config.IDENTITY_DB_PATH):
            self.reid.load(config.IDENTITY_DB_PATH)

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR,    exist_ok=True)

        self._frame_log   = []   # frame-level metadata for post-analysis
        self._start_wall  = None
        print("="*60 + "\n")

    # ─── Main entry point ─────────────────────────────────────────────────────

    def run(self, video_path: str,
            save_video: bool  = True,
            save_report: bool = True,
            show_preview: bool = False) -> dict:
        """
        Process an entire video file end-to-end.

        Parameters
        ----------
        video_path   : path to input video
        save_video   : write annotated video to outputs/
        save_report  : write CSV report and plots
        show_preview : show live OpenCV window (disable on servers)

        Returns
        -------
        results dict with keys: report_df, sessions_df, hourly_df, stats
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps      = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Pipeline] Video: {os.path.basename(video_path)}")
        print(f"[Pipeline] Resolution: {width}×{height} @ {src_fps:.1f} FPS  |  "
              f"Total frames: {total_frames}")

        writer = None
        if save_video:
            writer = self._make_writer(src_fps, width, height)

        self._start_wall = time.time()
        frame_idx = 0
        processed = 0

        pbar = tqdm(total=total_frames, desc="Processing", unit="fr")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            # Skip frames to reduce load
            if frame_idx % config.FRAME_SKIP != 0:
                if writer is not None:
                    writer.write(frame)
                continue

            processed += 1
            frame_resized = self._preprocess(frame)
            annotated, frame_meta = self._process_frame(frame_resized, frame_idx)

            self._frame_log.append(frame_meta)

            if writer is not None:
                out = cv2.resize(annotated, (width, height))
                writer.write(out)

            if show_preview:
                cv2.imshow("Fisherman Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Periodic checkpoint every 1000 processed frames
            if processed % 1000 == 0:
                self._checkpoint()

        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"\n[Pipeline] Processed {processed} frames in "
              f"{time.time()-self._start_wall:.1f}s")

        self.timer.finalize()
        return self._finalise(save_report)

    # ─── Frame processing ─────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, frame_idx: int):
        """Process a single frame. Returns (annotated_frame, metadata_dict)."""
        meta = {"frame": frame_idx, "detections": 0, "tracks": [], "persons": []}

        # 1. Detect
        boxes, scores, crops = self.detector.detect(frame)
        meta["detections"] = len(boxes)

        # 2. Track
        tracks = self.tracker.update(frame, boxes, scores)

        # 3. Re-ID + timer
        person_id_map = {}
        visible_persons = set()

        # Build a lookup: match track bboxes to nearest detector crop
        def best_crop_for_track(bbox, boxes, crops, frame):
            if not boxes:
                x1,y1,x2,y2 = [int(v) for v in bbox]
                crop = frame[max(0,y1):max(y1+1,y2), max(0,x1):max(x1+1,x2)]
                return crop
            tx1,ty1,tx2,ty2 = [int(v) for v in bbox]
            tcx, tcy = (tx1+tx2)/2, (ty1+ty2)/2
            best_idx, best_dist = 0, float('inf')
            for i,(bx1,by1,bx2,by2) in enumerate(boxes):
                cx,cy = (bx1+bx2)/2,(by1+by2)/2
                d = (cx-tcx)**2+(cy-tcy)**2
                if d < best_dist:
                    best_dist,best_idx = d,i
            return crops[best_idx]

        for t in tracks:
            tid  = t["track_id"]
            bbox = t["bbox"]

            # Use nearest detector crop instead of re-cropping from tracker bbox
            face_crop = best_crop_for_track(bbox, boxes, crops, frame)

            # Re-ID
            person_label = self.reid.process_track(tid, face_crop)
            person_id_map[tid] = person_label
            visible_persons.add(person_label)

            meta["tracks"].append(tid)
            meta["persons"].append(person_label)

        # 4. Update working hour estimator
        self.timer.update(visible_persons)

        # 5. Annotate frame
        annotated = self._annotate(frame, tracks, person_id_map)

        return annotated, meta

    # ─── Annotation ───────────────────────────────────────────────────────────

    def _annotate(self, frame, tracks, person_id_map) -> np.ndarray:
        """Draw bounding boxes, person labels, and HUD on frame."""
        vis = self.tracker.draw_tracks(frame, tracks, person_id_map)

        # HUD overlay — top-left stats
        people_on_screen = set(person_id_map.values())
        hud_lines = [
            f"People on screen: {len(people_on_screen)}",
            f"Known identities: {len(self.reid.gallery)}",
            f"Re-IDs:  {self.reid.stats['reidentifications']}",
            f"Switches prevented: {self.reid.stats['identity_switches_prevented']}"
        ]
        for i, line in enumerate(hud_lines):
            y = 22 + i * 22
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Working hours per visible person — top-right
        report = self.timer.get_report()
        if not report.empty:
            for j, row in report.iterrows():
                if row["Person ID"] in people_on_screen:
                    text = f"{row['Person ID']}: {row['Total (HH:MM:SS)']}"
                    x = vis.shape[1] - 10
                    y = 22 + j * 22
                    cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 255, 200), 1, cv2.LINE_AA)

        return vis

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))

    def _make_writer(self, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(config.ANNOTATED_VIDEO, fourcc, fps, (width, height))

    def _checkpoint(self):
        """Save gallery and intermediate report to disk."""
        self.reid.save()
        report = self.timer.get_report()
        if not report.empty:
            report.to_csv(config.REPORT_CSV, index=False)

    def _finalise(self, save_report: bool) -> dict:
        """Generate all outputs after processing completes."""
        report_df   = self.timer.get_report()
        sessions_df = self.timer.get_sessions_df()
        hourly_df   = self.timer.get_hourly_matrix()

        if save_report:
            # Save CSVs
            report_df.to_csv(config.REPORT_CSV, index=False)
            sessions_df.to_csv("outputs/sessions.csv", index=False)
            hourly_df.to_csv("outputs/hourly_activity.csv")

            # Save gallery
            self.reid.save()

            # Generate plots (gallery needs dict[str, list])
            gallery_dict = {k: list(v) for k, v in self.reid.gallery.items()}
            viz.plot_embeddings_umap(gallery_dict)
            viz.plot_working_hours(report_df)
            viz.plot_presence_timeline(sessions_df)
            viz.plot_hourly_heatmap(hourly_df)

            # Save run metadata
            meta = {
                "total_frames_processed": len(self._frame_log),
                "identities_found":       len(self.reid.gallery),
                "reid_stats":             self.reid.stats,
                "wall_time_sec":          round(time.time() - self._start_wall, 2)
            }
            with open("outputs/run_metadata.json", "w") as f:
                import json; json.dump(meta, f, indent=2)

            print("\n[Pipeline] Outputs saved:")
            print(f"  Report CSV  → {config.REPORT_CSV}")
            print(f"  Embed plot  → {config.EMBED_PLOT_PATH}")
            print(f"  Video out   → {config.ANNOTATED_VIDEO}")

        print("\n[Pipeline] ── Working Hour Report ──")
        print(report_df.to_string(index=False))
        print(f"\n[Pipeline] Re-ID stats: {self.reid.stats}")

        return {
            "report_df":   report_df,
            "sessions_df": sessions_df,
            "hourly_df":   hourly_df,
            "stats":       self.reid.stats,
            "gallery":     {k: list(v) for k, v in self.reid.gallery.items()}
        }
