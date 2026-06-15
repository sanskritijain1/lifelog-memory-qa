"""
build_frame_timestamps.py

Computes the correct frame→timestamp mapping for each session
by reading the actual video FPS and duration via OpenCV.

Run this ONCE before re-running segment_events.py.
Output: data/session_timestamps.json

Schema:
{
  "P01_09": {
    "fps"           : 59.94,
    "frame_interval": 59,        # int(fps) used during extraction
    "total_frames"  : 3627,      # saved frames in data/frames/P01_09/
    "duration_s"    : 3571.2,    # video duration in seconds
    "spf"           : 0.9844     # seconds per saved frame
  },
  ...
}
"""

import os, json
import cv2
from pathlib import Path

VIDEO_ROOT  = "epic_data/EPIC-KITCHENS"
FRAMES_ROOT = "data/frames"
OUT_PATH    = "data/session_timestamps.json"

result = {}

video_files = list(Path(VIDEO_ROOT).rglob("*.MP4"))
print(f"Found {len(video_files)} videos\n")

for vpath in sorted(video_files):
    session = vpath.stem   # e.g. P01_09

    cap        = cv2.VideoCapture(str(vpath))
    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_vid  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = total_vid / fps if fps > 0 else 0
    cap.release()

    if fps == 0:
        print(f"  ✗ {session}: could not read FPS")
        continue

    frame_interval = int(fps)   # matches extraction script

    # Count actual saved frames
    frames_dir   = Path(FRAMES_ROOT) / session
    saved_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0

    # seconds per saved frame = video duration / number of saved frames
    spf = duration_s / saved_frames if saved_frames > 0 else frame_interval

    result[session] = {
        "fps"           : round(fps, 4),
        "frame_interval": frame_interval,
        "total_frames"  : saved_frames,
        "duration_s"    : round(duration_s, 2),
        "spf"           : round(spf, 6),
    }

    print(f"  {session}:")
    print(f"    FPS           : {fps:.2f}")
    print(f"    Frame interval: {frame_interval}")
    print(f"    Saved frames  : {saved_frames}")
    print(f"    Duration      : {duration_s:.1f}s  ({duration_s/60:.1f} min)")
    print(f"    Sec/frame     : {spf:.4f}s")
    print()

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"✓ Saved → {OUT_PATH}")