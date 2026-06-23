import json
from pathlib import Path

import numpy as np
import faiss

FRAME_EMBEDDINGS_PATH = "data/frame_embeddings.npy"
FRAME_PATHS_PATH = "data/frame_paths.txt"
EVENTS_PATH = "data/events.json"

EVENT_EMBEDDINGS_PATH = "data/event_embeddings.npy"
EVENT_METADATA_PATH = "data/event_metadata.json"
EVENT_INDEX_PATH = "data/event_index.faiss"

TOP_FRAMES_PER_EVENT = 20

print("Loading frame embeddings...")
frame_embeddings = np.load(FRAME_EMBEDDINGS_PATH).astype("float32")

with open(FRAME_PATHS_PATH, "r") as f:
    frame_paths = [line.strip() for line in f if line.strip()]

with open(EVENTS_PATH, "r") as f:
    events = json.load(f)

path_to_idx = {p: i for i, p in enumerate(frame_paths)}

event_embeddings = []
event_metadata = []

print(f"Building embeddings for {len(events)} events...")

for event in events:
    event_frame_paths = event.get("frame_paths", [])

    valid_indices = [
        path_to_idx[p]
        for p in event_frame_paths
        if p in path_to_idx
    ]

    if not valid_indices:
        continue

    # sample frames if event is very long
    if len(valid_indices) > TOP_FRAMES_PER_EVENT:
        sample_positions = np.linspace(
            0,
            len(valid_indices) - 1,
            TOP_FRAMES_PER_EVENT
        ).astype(int)

        valid_indices = [valid_indices[i] for i in sample_positions]

    vectors = frame_embeddings[valid_indices]

    # mean event representation
    event_vec = vectors.mean(axis=0)

    # normalize event vector
    event_vec = event_vec.astype("float32")
    event_vec = event_vec / (np.linalg.norm(event_vec) + 1e-8)

    event_embeddings.append(event_vec)

    caption = (
        event.get("blip2_caption", "").strip()
        or event.get("activity_label", "")
        or ""
    )

    event_metadata.append({
        "event_id": event["event_id"],
        "session": event["session"],
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "start_s": event.get("start_s", 0),
        "duration_s": event.get("duration_s", 0),
        "frame_count": event.get("frame_count", len(valid_indices)),
        "center_frame": event.get("center_path", event_frame_paths[len(event_frame_paths)//2]),
        "caption": caption,
        "num_frames_used": len(valid_indices),
    })

event_embeddings = np.vstack(event_embeddings).astype("float32")

print("Normalizing event embeddings...")
faiss.normalize_L2(event_embeddings)

print("Building FAISS event index...")
index = faiss.IndexFlatIP(event_embeddings.shape[1])
index.add(event_embeddings)

np.save(EVENT_EMBEDDINGS_PATH, event_embeddings)

with open(EVENT_METADATA_PATH, "w") as f:
    json.dump(event_metadata, f, indent=2)

faiss.write_index(index, EVENT_INDEX_PATH)

print("Done!")
print("Event embeddings:", event_embeddings.shape)
print("Saved:", EVENT_EMBEDDINGS_PATH)
print("Saved:", EVENT_METADATA_PATH)
print("Saved:", EVENT_INDEX_PATH)