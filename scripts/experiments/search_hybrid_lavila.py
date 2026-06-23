import os, sys, json, re
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

LAVILA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "LaViLa"
)
if LAVILA_ROOT not in sys.path:
    sys.path.insert(0, LAVILA_ROOT)

import numpy as np
import faiss
import torch
from transformers import CLIPTokenizer
from lavila.models import models as lavila_models

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# -----------------------------
# CONFIG
# -----------------------------
CHECKPOINT_PATH = "pretrained/lavila_tsf_base_ep5.pth"

FRAME_EMBEDDINGS_PATH = "data/frame_embeddings.npy"
FRAME_PATHS_PATH = "data/frame_paths.txt"
MEAN_PATH = "data/frame_mean.npy"

EVENT_INDEX_PATH = "data/event_index.faiss"
EVENT_METADATA_PATH = "data/event_metadata.json"
EVENTS_PATH = "data/events.json"

DEVICE = "cpu"
NUM_FRAMES = 4

FRAME_POOL = 100
EVENT_POOL = 30
TOP_K = 10

FRAME_WEIGHT = 0.65
EVENT_WEIGHT = 0.35

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading LaViLa text encoder...")

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}

model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
model.load_state_dict(state_dict, strict=False)
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
mean_vec = np.load(MEAN_PATH).astype("float32")

print("  ✓ Model ready")

# -----------------------------
# LOAD FRAME INDEX
# -----------------------------
print("Loading frame embeddings...")

frame_embeddings = np.load(FRAME_EMBEDDINGS_PATH).astype("float32")

with open(FRAME_PATHS_PATH, "r") as f:
    frame_paths = [line.strip() for line in f if line.strip()]

frame_embeddings = frame_embeddings - mean_vec
faiss.normalize_L2(frame_embeddings)

frame_index = faiss.IndexFlatIP(frame_embeddings.shape[1])
frame_index.add(frame_embeddings)

print(f"  ✓ Frame index ready: {frame_index.ntotal} frames")

# -----------------------------
# LOAD EVENT INDEX
# -----------------------------
print("Loading event index...")

event_index = faiss.read_index(EVENT_INDEX_PATH)

with open(EVENT_METADATA_PATH, "r") as f:
    event_metadata = json.load(f)

print(f"  ✓ Event index ready: {len(event_metadata)} events")

# -----------------------------
# LOAD EVENTS FOR FRAME→EVENT MAPPING
# -----------------------------
with open(EVENTS_PATH, "r") as f:
    events = json.load(f)

frame_to_event = {}

for ev in events:
    for fp in ev.get("frame_paths", []):
        fn = int(Path(fp).stem.split("_")[1])
        frame_to_event[(ev["session"], fn)] = ev

# -----------------------------
# QUERY CLEANING
# -----------------------------
def clean_query(q: str) -> str:
    q = q.lower().replace("?", "").strip()

    remove_patterns = [
        r"\bwhen did\b",
        r"\bwhen was\b",
        r"\bwhere did\b",
        r"\bwhat time did\b",
        r"\bwhat happened\b",
        r"\bsomeone\b",
        r"\bthe person\b",
        r"\bhe\b",
        r"\bshe\b",
        r"\bthey\b",
        r"\bdid\b",
        r"\bwas\b",
        r"\bwere\b",
        r"\bis\b",
        r"\bare\b",
        r"\ba\b",
        r"\ban\b",
        r"\bthe\b",
    ]

    for pattern in remove_patterns:
        q = re.sub(pattern, "", q)

    return " ".join(q.split())


def build_query_variants(query: str):
    raw = query.strip()
    cleaned = clean_query(query)

    variants = []

    if raw:
        variants.append(raw)

    if cleaned and cleaned not in variants:
        variants.append(cleaned)

    return variants

# -----------------------------
# ENCODE QUERY
# -----------------------------
def encode_query(text: str):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )

    with torch.no_grad():
        feat = model.encode_text(tokens["input_ids"])
        feat = feat / feat.norm(dim=-1, keepdim=True)

    vec = feat.cpu().numpy().astype("float32")
    vec = vec - mean_vec
    faiss.normalize_L2(vec)

    return vec

# -----------------------------
# SCORE NORMALIZATION
# -----------------------------
def normalize_scores(score_dict):
    if not score_dict:
        return {}

    vals = np.array(list(score_dict.values()), dtype="float32")

    mn = float(vals.min())
    mx = float(vals.max())

    if mx - mn < 1e-8:
        return {k: 1.0 for k in score_dict}

    return {
        k: (v - mn) / (mx - mn)
        for k, v in score_dict.items()
    }

# -----------------------------
# FRAME SEARCH → EVENTS
# -----------------------------
def search_frames(query_variants):
    event_frame_scores = {}
    event_best_frame = {}

    for q in query_variants:
        vec = encode_query(q)
        D, I = frame_index.search(vec, FRAME_POOL)

        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue

            fp = frame_paths[idx]
            p = Path(fp)
            session = p.parent.name
            frame_num = int(p.stem.split("_")[1])

            ev = frame_to_event.get((session, frame_num))
            if not ev:
                continue

            eid = ev["event_id"]

            if eid not in event_frame_scores or score > event_frame_scores[eid]:
                event_frame_scores[eid] = float(score)
                event_best_frame[eid] = {
                    "best_frame": fp,
                    "frame_query_used": q,
                }

    return event_frame_scores, event_best_frame

# -----------------------------
# EVENT SEARCH
# -----------------------------
def search_events(query_variants):
    event_scores = {}

    for q in query_variants:
        vec = encode_query(q)
        D, I = event_index.search(vec, EVENT_POOL)

        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue

            ev = event_metadata[idx]
            eid = ev["event_id"]

            if eid not in event_scores or score > event_scores[eid]["score"]:
                event_scores[eid] = {
                    "score": float(score),
                    "event_query_used": q,
                }

    return event_scores

# -----------------------------
# HYBRID SEARCH
# -----------------------------
def hybrid_search(query):
    variants = build_query_variants(query)

    frame_scores, frame_info = search_frames(variants)
    event_scores = search_events(variants)

    frame_norm = normalize_scores(frame_scores)
    event_norm = normalize_scores({
        eid: info["score"] for eid, info in event_scores.items()
    })

    all_event_ids = set(frame_norm.keys()) | set(event_norm.keys())

    event_lookup = {ev["event_id"]: ev for ev in event_metadata}

    results = []

    for eid in all_event_ids:
        ev = event_lookup.get(eid)
        if not ev:
            continue

        fs = frame_norm.get(eid, 0.0)
        es = event_norm.get(eid, 0.0)

        final_score = FRAME_WEIGHT * fs + EVENT_WEIGHT * es

        caption = ev.get("caption", "")

        results.append({
            **ev,
            "hybrid_score": final_score,
            "frame_score_norm": fs,
            "event_score_norm": es,
            "best_frame": frame_info.get(eid, {}).get("best_frame", ev.get("center_frame")),
            "frame_query_used": frame_info.get(eid, {}).get("frame_query_used", ""),
            "event_query_used": event_scores.get(eid, {}).get("event_query_used", ""),
            "caption": caption,
        })

    results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

    print(f"\nQuery variants: {variants}")
    print("\nRank  Hybrid  Frame   Event   Session   Time                      Caption")
    print("-" * 110)

    for i, r in enumerate(results[:TOP_K], start=1):
        print(
            f"{i:<5} "
            f"{r['hybrid_score']:.4f}  "
            f"{r['frame_score_norm']:.3f}   "
            f"{r['event_score_norm']:.3f}   "
            f"{r['session']:<8} "
            f"{r['start_time']} → {r['end_time']}   "
            f"{r.get('caption', '')}"
        )

        print(f"      best frame: {r.get('best_frame')}")
        print(f"      frame query: {r.get('frame_query_used')} | event query: {r.get('event_query_used')}")

# -----------------------------
# CLI
# -----------------------------
print("\n" + "=" * 62)
print("  HYBRID LIFelog MEMORY SEARCH")
print("=" * 62)
print("  Combines:")
print("    65% frame-level evidence")
print("    35% event-level evidence")
print()
print("  Try:")
print("    cut onions")
print("    white plate")
print("    washing hands")
print("    open fridge")
print("    cooking potatoes")
print("  Type 'exit' to quit.")
print("=" * 62)

while True:
    q = input("\nQuery > ").strip()

    if q.lower() in {"exit", "quit", "q"}:
        print("Exiting.")
        break

    if not q:
        continue

    try:
        hybrid_search(q)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()