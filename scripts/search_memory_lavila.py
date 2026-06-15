import os, sys

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
CHECKPOINT_PATH = "pretrained/lavila_tsf_base_ep5.pth"
TOP_K           = 5
DEVICE          = "cpu"
NUM_FRAMES      = 4   # must match build script

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading LaViLa dual encoder...")

from lavila.models import models as lavila_models

ckpt       = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = {k.replace("module.", ""): v
              for k, v in ckpt["state_dict"].items()}

model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()
print("  ✓ Model ready")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
print("  ✓ Tokenizer ready")

# ── LOAD EMBEDDINGS + PATHS ───────────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
with open(PATHS_FILE) as f:
    paths = [l.strip() for l in f if l.strip()]

assert len(paths) == embeddings.shape[0], \
    f"Mismatch: {len(paths)} paths vs {embeddings.shape[0]} embeddings"
print(f"  ✓ {len(paths)} frames  |  dim={embeddings.shape[1]}")

# ── MEAN-CENTERING ────────────────────────────────────────────────────────────
# Removes the "DC offset" that pushes all egocentric kitchen frames
# into the same region of the embedding space.
if os.path.exists(MEAN_PATH):
    mean_vec = np.load(MEAN_PATH).astype("float32")
    print(f"  ✓ Loaded mean vector from {MEAN_PATH}")
else:
    mean_vec = embeddings.mean(axis=0).astype("float32")
    np.save(MEAN_PATH, mean_vec)
    print(f"  ✓ Computed and saved mean vector → {MEAN_PATH}")

embeddings = embeddings - mean_vec
faiss.normalize_L2(embeddings)

sample   = embeddings[:200]
mask     = ~np.eye(200, dtype=bool)
mean_sim = (sample @ sample.T)[mask].mean()
print(f"  Mean pairwise sim after centering: {mean_sim:.4f}")

# ── BUILD FAISS INDEX ─────────────────────────────────────────────────────────
print("Building FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"  ✓ {index.ntotal} vectors indexed")

# ── TEXT ENCODER ──────────────────────────────────────────────────────────────
def encode_query(text: str) -> np.ndarray:
    """
    Encode a text query using LaViLa's text encoder.

    LaViLa's text encoder is an OpenAI CLIP transformer trained on
    egocentric narrations — queries phrased as short actions work best:
      'cut the onion'  'open the fridge'  'wash hands'

    The output is mean-centred and normalised to match stored embeddings.
    """
    # Tokenize and pad/truncate to exactly 77 tokens (CLIP context length)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",   # pad to max_length
        truncation=True,
        max_length=77,
    )
    input_ids = tokens["input_ids"].to(DEVICE)   # shape (1, 77) ✓

    with torch.no_grad():
        text_feat = model.encode_text(input_ids)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    vec = text_feat.cpu().numpy().astype("float32")
    vec = vec - mean_vec
    faiss.normalize_L2(vec)
    return vec

# ── SEARCH ────────────────────────────────────────────────────────────────────
def search(query_text: str, top_k: int = TOP_K) -> list[dict]:
    query_vec          = encode_query(query_text)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "rank":  len(results) + 1,
            "path":  paths[idx],
            "score": float(score),
        })
    return results

def print_results(results: list[dict]) -> None:
    if not results:
        print("  (no results)")
        return
    print(f"\n  {'Rank':<5} {'Score':>7}   Path")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['rank']:<5} {r['score']:>7.4f}   {r['path']}")

# ── INTERACTIVE LOOP ──────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  LIFELOG MEMORY SEARCH  —  LaViLa egocentric model")
print("=" * 62)
print("  Query tips (LaViLa was trained on Ego4D narrations):")
print("    ✓  'cut the onion'")
print("    ✓  'open the fridge'")
print("    ✓  'wash hands in the sink'")
print("    ✓  'pour water into the pot'")
print("    ✗  'a person holding a knife'  ← CLIP-style, less effective")
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
        results = search(raw)
        print_results(results)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()