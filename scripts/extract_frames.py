import cv2
import os
from pathlib import Path

VIDEO_ROOT = Path("epic_data/EPIC-KITCHENS")
OUTPUT_ROOT = Path("data/frames")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

video_files = list(VIDEO_ROOT.rglob("*.MP4"))

print(f"Found {len(video_files)} videos.\n")

for video_path in video_files:

    print(f"Processing: {video_path.name}")

    save_folder = OUTPUT_ROOT / video_path.stem
    save_folder.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Could not read {video_path}")
        continue

    frame_interval = int(fps)

    frame_count = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:

            filename = save_folder / f"frame_{saved_count:05d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Saved {saved_count} frames.\n")

print("Done!")