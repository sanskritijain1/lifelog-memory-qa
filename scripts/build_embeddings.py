import os
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np

torch.set_num_threads(1)

# ---------------------------
# CONFIG
# ---------------------------
FRAME_ROOT   = "data/frames"
OUTPUT_FILE  = "data/frame_embeddings.npy"
OUTPUT_PATHS = "data/frame_paths.txt"
BATCH_SIZE   = 32
device       = "cpu"   # Force CPU — MPS + CLIP causes silent bugs on macOS

# ---------------------------
# LOAD CLIP
# ---------------------------
print("Loading CLIP model...")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ---------------------------
# COLLECT FRAME PATHS
# ---------------------------
frame_paths = []
for video_folder in sorted(os.listdir(FRAME_ROOT)):
    video_path = os.path.join(FRAME_ROOT, video_folder)
    if not os.path.isdir(video_path):
        continue
    for frame in sorted(os.listdir(video_path)):
        if frame.endswith(".jpg"):
            frame_paths.append(os.path.join(video_path, frame))

print(f"Found {len(frame_paths)} frames")

# ---------------------------
# EMBEDDING STORAGE
# valid_paths lives here (outer scope) so it accumulates across
# all batches and stays in sync with all_embeddings.
# BUG IN ORIGINAL: valid_paths was declared inside the loop,
# resetting every batch, then frame_paths (not valid_paths) was
# written to disk — causing path/embedding index mismatch when
# any image failed to load.
# ---------------------------
all_embeddings = []
valid_paths    = []

# ---------------------------
# PROCESS IN BATCHES
# ---------------------------
for i in tqdm(range(0, len(frame_paths), BATCH_SIZE)):

    batch_paths = frame_paths[i:i + BATCH_SIZE]

    images            = []
    batch_valid_paths = []

    # Load images safely
    for path in batch_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            batch_valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if len(images) == 0:
        continue

    # Preprocess — images= only, no text= argument
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ---------------------------
    # CLIP FORWARD PASS  ← THE FIX
    #
    # ORIGINAL (broken):
    #   vision_outputs = model.vision_model(inputs["pixel_values"])
    #   pooled_output  = vision_outputs.pooler_output
    #   image_features = model.visual_projection(pooled_output)
    #
    # Without return_dict=True, model.vision_model() returns a plain
    # tuple. Accessing .pooler_output on a tuple doesn't error —
    # it silently returns the wrong tensor (index-based fallback).
    # This produces embeddings that are valid (norm=1.0) but encode
    # almost no visual content, so all frames cluster together
    # (mean pairwise sim ~0.77 instead of ~0.22).
    #
    # get_image_features() passes return_dict=True internally and
    # is the correct API for this task.
    # ---------------------------
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    all_embeddings.append(image_features.cpu().numpy())
    valid_paths.extend(batch_valid_paths)  # extend AFTER successful encode

# ---------------------------
# SAVE OUTPUT
# ---------------------------
embeddings = np.concatenate(all_embeddings, axis=0)

assert len(valid_paths) == embeddings.shape[0], \
    f"Mismatch: {len(valid_paths)} paths vs {embeddings.shape[0]} embeddings"

np.save(OUTPUT_FILE, embeddings)
with open(OUTPUT_PATHS, "w") as f:
    for p in valid_paths:
        f.write(p + "\n")

# Quick sanity check on output diversity
sample = embeddings[:min(200, len(embeddings))]
mask   = ~np.eye(len(sample), dtype=bool)
sims   = (sample @ sample.T)[mask]
print(f"\nDone!")
print(f"  Embeddings shape      : {embeddings.shape}")
print(f"  Mean pairwise sim     : {sims.mean():.4f}  (healthy = 0.20–0.30)")
if sims.mean() > 0.5:
    print("  ⚠️  WARNING: embeddings still too similar — check CLIP is loading correctly")
else:
    print("  ✓  Embedding diversity looks healthy — search should work")