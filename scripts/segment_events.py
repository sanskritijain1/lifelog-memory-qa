"""
segment_events.py  — with correct per-session timestamps

Requires data/session_timestamps.json from build_frame_timestamps.py.
Frame numbers are converted to real timestamps using actual spf per session.
"""

import os, json
import numpy as np
from pathlib import Path
from collections import defaultdict

EMBEDDINGS_PATH     = "data/frame_embeddings.npy"
PATHS_FILE          = "data/frame_paths.txt"
TIMESTAMPS_PATH     = "data/session_timestamps.json"
EVENTS_OUT          = "data/events.json"
BOUNDARY_THRESHOLD  = 0.40
MIN_EVENT_SECONDS   = 3

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading embeddings and paths...")
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
with open(PATHS_FILE) as f:
    paths = [l.strip() for l in f if l.strip()]
assert len(paths) == embeddings.shape[0]
print(f"  {len(paths)} frames  |  dim={embeddings.shape[1]}")

print("Loading session timestamps...")
with open(TIMESTAMPS_PATH) as f:
    session_ts = json.load(f)
for s, info in session_ts.items():
    print(f"  {s}: spf={info['spf']:.4f}s  duration={info['duration_s']:.1f}s")

# ── GROUP BY SESSION ──────────────────────────────────────────────────────────
session_frames = defaultdict(list)
for global_idx, path in enumerate(paths):
    p         = Path(path)
    session   = p.parent.name
    frame_num = int(p.stem.split("_")[1])
    session_frames[session].append((global_idx, frame_num))

for s in session_frames:
    session_frames[s].sort(key=lambda x: x[1])

sessions = sorted(session_frames.keys())
print(f"\nFound {len(sessions)} sessions: {sessions}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def frame_to_seconds(session: str, frame_num: int) -> float:
    """Convert saved frame number to real timestamp in seconds."""
    if session in session_ts:
        return frame_num * session_ts[session]["spf"]
    return float(frame_num)   # fallback: assume 1fps

def seconds_to_hms(s: float) -> str:
    s   = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ── SEGMENT ───────────────────────────────────────────────────────────────────
all_events = []
event_id   = 0
skipped    = 0
boundaries = 0

print(f"\nSegmenting (threshold={BOUNDARY_THRESHOLD}, min={MIN_EVENT_SECONDS}s)...")

for session in sessions:
    frames         = session_frames[session]
    global_indices = [f[0] for f in frames]
    frame_numbers  = [f[1] for f in frames]
    embs           = embeddings[global_indices]

    sims = np.einsum('ij,ij->i', embs[:-1], embs[1:])

    boundary_pos = [0]
    for i, sim in enumerate(sims):
        if sim < BOUNDARY_THRESHOLD:
            boundary_pos.append(i + 1)
            boundaries += 1
    boundary_pos.append(len(frames))

    for b in range(len(boundary_pos) - 1):
        start_pos = boundary_pos[b]
        end_pos   = boundary_pos[b + 1] - 1

        start_frame = frame_numbers[start_pos]
        end_frame   = frame_numbers[end_pos]

        # ✅ Convert frame numbers to real timestamps
        start_s = frame_to_seconds(session, start_frame)
        end_s   = frame_to_seconds(session, end_frame)
        duration = end_s - start_s

        if duration < MIN_EVENT_SECONDS:
            skipped += 1
            continue

        event_global_indices = global_indices[start_pos : end_pos + 1]
        event_paths          = [paths[i] for i in event_global_indices]
        center_pos           = (start_pos + end_pos) // 2
        center_path          = paths[global_indices[center_pos]]
        center_idx           = global_indices[center_pos]

        all_events.append({
            "event_id"   : event_id,
            "session"    : session,
            "start_frame": start_frame,
            "end_frame"  : end_frame,
            "start_s"    : round(start_s, 1),
            "end_s"      : round(end_s, 1),
            "start_time" : seconds_to_hms(start_s),
            "end_time"   : seconds_to_hms(end_s),
            "duration_s" : round(duration, 1),
            "frame_count": len(event_paths),
            "frame_paths": event_paths,
            "center_path": center_path,
            "center_idx" : center_idx,
        })
        event_id += 1

# ── SAVE ──────────────────────────────────────────────────────────────────────
with open(EVENTS_OUT, "w") as f:
    json.dump(all_events, f, indent=2)

durations = [e["duration_s"] for e in all_events]
print(f"\n✓ Segmentation complete")
print(f"  Boundaries detected : {boundaries}")
print(f"  Events kept         : {len(all_events)}")
print(f"  Short events skipped: {skipped}")
print(f"  Duration  min/median/max: "
      f"{min(durations):.0f}s / {np.median(durations):.0f}s / {max(durations):.0f}s")
print(f"\n  Sample events (with correct timestamps):")
for e in all_events[:5]:
    print(f"    [{e['event_id']:>4}] {e['session']}  "
          f"{e['start_time']} → {e['end_time']}  "
          f"({e['duration_s']:.0f}s)")
print(f"\n  Saved → {EVENTS_OUT}")