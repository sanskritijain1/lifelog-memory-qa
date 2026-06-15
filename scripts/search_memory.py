import os

# ============================================================
# SAFETY ENV VARS — must be set BEFORE any other imports
# ============================================================
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

# ============================================================
# CONFIG
# ============================================================
EMBEDDINGS_PATH = "data/frame_embeddings.npy"
PATHS_FILE      = "data/frame_paths.txt"
MEAN_PATH       = "data/frame_mean.npy"   # saved on first run, reused after
MODEL_NAME      = "openai/clip-vit-base-patch32"
TOP_K           = 5
DEVICE          = "cpu"

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# ============================================================
# LOAD CLIP MODEL
# ============================================================
print("Loading CLIP model...")
model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("  ✓ Model loaded")

# ============================================================
# LOAD EMBEDDINGS + PATHS
# ============================================================
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

with open(PATHS_FILE) as f:
    paths = [l.strip() for l in f if l.strip()]

assert len(paths) == embeddings.shape[0], (
    f"Mismatch: {len(paths)} paths vs {embeddings.shape[0]} embeddings"
)
print(f"  ✓ {len(paths)} frames  |  dim: {embeddings.shape[1]}")

# ============================================================
# MEAN-CENTERING
#
# All EPIC-Kitchens frames cluster tightly in one region of the
# 512-d CLIP space (mean pairwise sim ~0.80) because the model
# was trained on web images, not egocentric kitchen video.
#
# Subtracting the dataset mean vector removes this "DC offset",
# spreading frames across the space so text queries can actually
# discriminate between them.
#
# The mean vector is computed once and saved so search_memory.py
# and any future scripts use the exact same centring.
# ============================================================
if os.path.exists(MEAN_PATH):
    mean_vec = np.load(MEAN_PATH).astype("float32")
    print(f"  ✓ Loaded mean vector from {MEAN_PATH}")
else:
    mean_vec = embeddings.mean(axis=0).astype("float32")
    np.save(MEAN_PATH, mean_vec)
    print(f"  ✓ Computed and saved mean vector → {MEAN_PATH}")

# Apply mean-centering then re-normalise
embeddings = embeddings - mean_vec
faiss.normalize_L2(embeddings)

# Quick diversity check after centering
sample = embeddings[:200]
mask   = ~np.eye(200, dtype=bool)
mean_sim_after = (sample @ sample.T)[mask].mean()
print(f"  Mean pairwise sim after centering: {mean_sim_after:.4f}  (target < 0.30)")

# ============================================================
# BUILD FAISS INDEX
# IndexFlatIP = inner product on L2-normalised vectors = cosine similarity
# Higher score = better match, range roughly -1 to 1
# ============================================================
print("Building FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"  ✓ Index ready  |  {index.ntotal} vectors")

# ============================================================
# QUERY ENCODER
# ============================================================
# Prompt templates: CLIP retrieves better with caption-style text
# than bare keywords, especially on egocentric video.
PROMPT_TEMPLATES = [
    "{}",
    "a photo of {}",
    "a first-person view of {}",
    "a person {}",
]

def encode_query(text: str, use_templates: bool = True) -> np.ndarray:
    """
    Encode a text query into a mean-centred, normalised CLIP embedding.

    If use_templates=True, encodes the query with multiple prompt templates
    and averages the results — this is the same trick OpenAI uses in their
    zero-shot ImageNet classifier and improves retrieval quality noticeably.
    """
    if use_templates:
        prompts = [t.format(text) for t in PROMPT_TEMPLATES]
    else:
        prompts = [text]

    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(DEVICE)

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    # Average across templates, then re-normalise
    vec = features.mean(dim=0, keepdim=True).cpu().numpy().astype("float32")

    # Apply the same mean-centering as the stored embeddings
    vec = vec - mean_vec
    faiss.normalize_L2(vec)
    return vec

# ============================================================
# SEARCH
# ============================================================
def search(query_text: str, top_k: int = TOP_K) -> list[dict]:
    query_vec = encode_query(query_text)
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

# ============================================================
# INTERACTIVE LOOP
# ============================================================
print("\n" + "=" * 60)
print("  LIFELOG MEMORY SEARCH  (mean-centred CLIP)")
print("  Commands: 'exit' to quit  |  prefix with '!' to skip templates")
print("  Example:  refrigerator")
print("  Example:  !a photo of a refrigerator  (raw query, no templates)")
print("=" * 60)

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

    use_templates = True
    query_text    = raw
    if raw.startswith("!"):
        use_templates = False
        query_text    = raw[1:].strip()

    try:
        results = search(query_text)
        print_results(results)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()