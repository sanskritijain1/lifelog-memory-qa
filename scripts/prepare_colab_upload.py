"""
prepare_colab_upload.py  —  Run on your Mac before uploading to Colab

Creates colab_upload/ with:
  - events.json          (your events file)
  - center_frames.zip    (all center frame images, named by event_id)

Upload both files to the Colab session Files panel.
"""

import os, json, shutil, zipfile
from pathlib import Path
from tqdm import tqdm

EVENTS_PATH = "data/events.json"
OUTPUT_DIR  = "colab_upload"

# ── Load events ───────────────────────────────────────────────────────────────
with open(EVENTS_PATH) as f:
    events = json.load(f)
print(f"Loaded {len(events)} events")

# ── Create output directory ───────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Copy events.json ──────────────────────────────────────────────────────────
shutil.copy(EVENTS_PATH, f"{OUTPUT_DIR}/events.json")
print(f"✓ Copied events.json")

# ── Collect center frames ─────────────────────────────────────────────────────
# Name each frame as {event_id}.jpg so Colab can look them up by event_id
print("Collecting center frames...")
missing = 0
zip_path = f"{OUTPUT_DIR}/center_frames.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for ev in tqdm(events):
        center = ev.get("center_path", "")
        if not center or not Path(center).exists():
            missing += 1
            continue
        # Store as event_id.jpg
        zf.write(center, f"{ev['event_id']}.jpg")

zip_size = Path(zip_path).stat().st_size / 1e6
print(f"\n✓ Created center_frames.zip  ({zip_size:.1f} MB)")
print(f"  Frames included : {len(events) - missing}")
print(f"  Missing         : {missing}")
print(f"\nUpload these two files to Colab:")
print(f"  {OUTPUT_DIR}/events.json")
print(f"  {OUTPUT_DIR}/center_frames.zip")