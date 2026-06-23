import os, sys, json, re

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
EVENT_INDEX_PATH = "data/event_index.faiss"
EVENT_METADATA_PATH = "data/event_metadata.json"
MEAN_PATH = "data/frame_mean.npy"

DEVICE = "cpu"
NUM_FRAMES = 4
TOP_K = 10

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading LaViLa text encoder...")

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = {
    k.replace("module.", ""): v
    for k, v in ckpt["state_dict"].items()
}

model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(
    num_frames=NUM_FRAMES
)

model.load_state_dict(state_dict, strict=False)
model.eval()

tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-base-patch32"
)

mean_vec = np.load(MEAN_PATH).astype("float32")

print("  ✓ Model ready")
print("  ✓ Tokenizer ready")

# -----------------------------
# LOAD EVENT INDEX
# -----------------------------
print("Loading event index...")

index = faiss.read_index(EVENT_INDEX_PATH)

with open(EVENT_METADATA_PATH, "r") as f:
    metadata = json.load(f)

print(f"  ✓ Loaded {len(metadata)} events")

# -----------------------------
# QUERY CLEANING
# -----------------------------
def clean_query(q: str) -> str:
    """
    Convert question-like text into LaViLa-style action text
    without corrupting words.

    Examples:
    'when did someone cut the onions?' -> 'cut onions'
    'when was a white plate used?'     -> 'white plate used'
    'washing hands'                    -> 'washing hands'
    """
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
    ]

    for pattern in remove_patterns:
        q = re.sub(pattern, "", q)

    # remove standalone articles only
    q = re.sub(r"\bthe\b", "", q)
    q = re.sub(r"\ba\b", "", q)
    q = re.sub(r"\ban\b", "", q)

    return " ".join(q.split())


def build_query_variants(query: str) -> list[str]:
    raw = query.strip()
    cleaned = clean_query(query)

    variants = []

    if raw:
        variants.append(raw)

    if cleaned and cleaned not in variants:
        variants.append(cleaned)

    return variants

# -----------------------------
# ENCODING
# -----------------------------
def encode_query(text: str) -> np.ndarray:
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

    # Important: same mean-centering used during frame/event embedding creation
    vec = vec - mean_vec
    faiss.normalize_L2(vec)

    return vec

# -----------------------------
# SEARCH
# -----------------------------
def search(query: str):
    variants = build_query_variants(query)

    merged = {}

    for q in variants:
        vec = encode_query(q)
        D, I = index.search(vec, TOP_K)

        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue

            event = metadata[idx]
            eid = event["event_id"]

            if eid not in merged or score > merged[eid]["score"]:
                merged[eid] = {
                    **event,
                    "score": float(score),
                    "query_used": q,
                }

    results = sorted(
        merged.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    print(f"\nQuery variants: {variants}")
    print("\nRank   Score    Session   Time                      Query Used              Caption")
    print("-" * 115)

    for i, r in enumerate(results[:TOP_K], start=1):
        caption = r.get("caption", "")
        query_used = r.get("query_used", "")

        print(
            f"{i:<6} "
            f"{r['score']:.4f}   "
            f"{r['session']:<8} "
            f"{r['start_time']} → {r['end_time']}   "
            f"{query_used:<22} "
            f"{caption}"
        )

# -----------------------------
# CLI
# -----------------------------
print("\n" + "=" * 62)
print("  EVENT-LEVEL LIFelog MEMORY SEARCH — LaViLa")
print("=" * 62)
print("  Test queries:")
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
        search(q)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()