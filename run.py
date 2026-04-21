"""
run.py
Command-line interface for the Fisherman Tracking pipeline.

Usage examples:
  python run.py --video data/videos/surveillance.mp4
  python run.py --video clip.mp4 --no-save-video --load-gallery
  python run.py --video clip.mp4 --conf 0.5 --reid-thresh 0.75
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import config


def parse_args():
    p = argparse.ArgumentParser(
        description="Fisherman Facial Recognition & Working Hour Estimation Pipeline"
    )
    p.add_argument("--video",        required=True, help="Path to input video")
    p.add_argument("--save-video",   action="store_true",  default=True,  help="Write annotated video")
    p.add_argument("--no-save-video",action="store_false", dest="save_video")
    p.add_argument("--load-gallery", action="store_true",  default=False, help="Resume from saved gallery")
    p.add_argument("--preview",      action="store_true",  default=False, help="Show OpenCV window")
    p.add_argument("--conf",         type=float, default=None, help=f"YOLO confidence (default {config.YOLO_CONF_THRESH})")
    p.add_argument("--reid-thresh",  type=float, default=None, help=f"Re-ID threshold (default {config.REID_SIMILARITY_THRESH})")
    p.add_argument("--frame-skip",   type=int,   default=None, help=f"Frame skip (default {config.FRAME_SKIP})")
    return p.parse_args()


def main():
    args = parse_args()

    # Override config from CLI args
    if args.conf        is not None: config.YOLO_CONF_THRESH       = args.conf
    if args.reid_thresh is not None: config.REID_SIMILARITY_THRESH  = args.reid_thresh
    if args.frame_skip  is not None: config.FRAME_SKIP              = args.frame_skip

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    from src.pipeline import Pipeline
    pipeline = Pipeline(load_gallery=args.load_gallery)
    results  = pipeline.run(
        args.video,
        save_video=args.save_video,
        save_report=True,
        show_preview=args.preview
    )

    print("\n✅ Done.")
    print(f"   Identities found: {len(results['gallery'])}")
    print(f"   Re-IDs:           {results['stats']['reidentifications']}")
    print(f"   Switches saved:   {results['stats']['identity_switches_prevented']}")
    print(f"\n   Results in: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
