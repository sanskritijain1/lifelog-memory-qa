"""
search_events.py

Event-level semantic search over the segmented memory.
Returns timestamped Memory Events, not individual frames.
"""

import os, sys, json

os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

LAVILA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LaViLa"
)
if LAVILA_ROOT not in sys.path:
    sys.path.insert(0, LAVILA_ROOT)

import numpy as np
import faiss
import torch
from transformers import CLIPTokenizer

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = "data/frame_embeddings.npy"
PATHS_FILE      = "data/frame_paths.txt"
MEAN_PATH       = "data/frame_mean.npy"
EVENTS_PATH     = "data/events.json"
CHECKPOINT_PATH = "pretrained/lavila_tsf_base_ep5.pth"
TOP_K           = 5
DEVICE          = "cpu"
NUM_FRAMES      = 4

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading LaViLa dual encoder...")
from lavila.models import models as lavila_models

ckpt       = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
model      = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
print("  ✓ Model ready")

# ── LOAD EMBEDDINGS ───────────────────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
with open(PATHS_FILE) as f:
    paths = [l.strip() for l in f if l.strip()]

# ✅ FIX: O(1) path → index lookup (was O(N) with list.index())
path_to_idx = {p: i for i, p in enumerate(paths)}

mean_vec   = np.load(MEAN_PATH).astype("float32")
embeddings = embeddings - mean_vec
faiss.normalize_L2(embeddings)
print(f"  ✓ {len(paths)} frame embeddings loaded")

# ── LOAD EVENTS + BUILD EVENT EMBEDDINGS ─────────────────────────────────────
print("Loading events and building event embeddings...")
with open(EVENTS_PATH) as f:
    events = json.load(f)

event_embeddings = np.zeros((len(events), embeddings.shape[1]), dtype="float32")

for i, event in enumerate(events):
    # ✅ FIX: dict lookup instead of list.index() — O(1) not O(N)
    indices = [path_to_idx[p] for p in event["frame_paths"] if p in path_to_idx]
    if not indices:
        indices = [event["center_idx"]]
    event_embeddings[i] = embeddings[indices].mean(axis=0)

faiss.normalize_L2(event_embeddings)
print(f"  ✓ {len(events)} event embeddings built")

# ── BUILD FAISS INDEX ─────────────────────────────────────────────────────────
print("Building FAISS index...")
index = faiss.IndexFlatIP(event_embeddings.shape[1])
index.add(event_embeddings)
print(f"  ✓ {index.ntotal} events indexed")

# ── TEXT ENCODER ──────────────────────────────────────────────────────────────
def encode_query(text: str) -> np.ndarray:
    tokens    = tokenizer(
        text, return_tensors="pt",
        padding="max_length", truncation=True, max_length=77,
    )
    input_ids = tokens["input_ids"].to(DEVICE)
    with torch.no_grad():
        feat = model.encode_text(input_ids)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec = feat.cpu().numpy().astype("float32")
    vec = vec - mean_vec
    faiss.normalize_L2(vec)
    return vec

# ── SEARCH (also exported for use by memory_qa.py) ───────────────────────────
def search_events(query_text: str, top_k: int = TOP_K) -> list[dict]:
    query_vec          = encode_query(query_text)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        e = events[idx]
        results.append({
            "rank"        : len(results) + 1,
            "score"       : float(score),
            "event_id"    : e["event_id"],
            "session"     : e["session"],
            "start_time"  : e["start_time"],
            "end_time"    : e["end_time"],
            "duration_s"  : e["duration_s"],
            "frame_count" : e["frame_count"],
            "center_frame": e["center_path"],
        })
    return results

def format_results(results: list[dict]) -> str:
    """Returns a formatted string — used both for printing and LLM context."""
    if not results:
        return "  (no results)"
    lines = []
    for r in results:
        lines.append(
            f"  ── Memory Event #{r['event_id']}  "
            f"(rank {r['rank']}, score {r['score']:.4f})\n"
            f"     Session   : {r['session']}\n"
            f"     Time      : {r['start_time']} → {r['end_time']}  "
            f"({r['duration_s']}s, {r['frame_count']} frames)\n"
            f"     Key frame : {r['center_frame']}"
        )
    return "\n\n".join(lines)

def print_results(results: list[dict]) -> None:
    print("\n" + format_results(results))

# ── INTERACTIVE LOOP ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  LIFELOG EVENT SEARCH  —  LaViLa egocentric model")
    print("=" * 62)
    print("  Returns memory events (time windows), not individual frames.")
    print("  Query style: short action narrations work best.")
    print("    ✓  'cut the onion'    ✓  'open the fridge'")
    print("    ✓  'wash hands'       ✓  'pour water into pot'")
    print("  Type 'exit' to quit.")
    print("=" * 62)

    while True:
        try:
            raw = input("\n  Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        try:
            results = search_events(raw)
            print_results(results)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()