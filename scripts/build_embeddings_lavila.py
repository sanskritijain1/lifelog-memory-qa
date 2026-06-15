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
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

torch.set_num_threads(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
FRAMES_DIR      = "data/frames"
EMBEDDINGS_OUT  = "data/frame_embeddings.npy"
PATHS_OUT       = "data/frame_paths.txt"
CHECKPOINT_PATH = "pretrained/lavila_tsf_base_ep5.pth"
BATCH_SIZE      = 16    # lower to 4 if you get memory errors
DEVICE          = "cpu"

# The checkpoint was trained with num_frames=4.
# We repeat each frame 4 times to match the expected temporal_embed shape.
# This is standard practice for single-frame retrieval with video models.
NUM_FRAMES = 4

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading LaViLa dual encoder...")

from lavila.models import models as lavila_models

ckpt       = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = {k.replace("module.", ""): v
              for k, v in ckpt["state_dict"].items()}

# Build with num_frames=4 to match checkpoint's temporal_embed shape (1,4,768)
model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  Checkpoint loaded  |  missing={len(missing)}  unexpected={len(unexpected)}")

model = model.to(DEVICE)
model.eval()
print("  ✓ Model ready")

# ── IMAGE PREPROCESSING ───────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    ),
])

def make_clip_tensor(pil_img):
    """
    PIL image → (1, 3, NUM_FRAMES, 224, 224) tensor.
    TimeSformer patch_embed expects (B, C, T, H, W).
    The same frame is repeated NUM_FRAMES times along the time dim.
    """
    t = preprocess(pil_img)               # (3, 224, 224)
    t = t.unsqueeze(1)                    # (3, 1, 224, 224)
    t = t.expand(-1, NUM_FRAMES, -1, -1)  # (3, 4, 224, 224)
    t = t.unsqueeze(0)                    # (1, 3, 4, 224, 224)
    return t.contiguous()

# ── SANITY CHECK ──────────────────────────────────────────────────────────────
print("\nRunning encoder sanity check...")
img_w = Image.new("RGB", (224, 224), (255, 255, 255))
img_b = Image.new("RGB", (224, 224), (0,   0,   0))
with torch.no_grad():
    fw = model.encode_image(make_clip_tensor(img_w))
    fb = model.encode_image(make_clip_tensor(img_b))
    fw = fw / fw.norm(dim=-1, keepdim=True)
    fb = fb / fb.norm(dim=-1, keepdim=True)
sim = float((fw @ fb.T)[0, 0])
print(f"  White vs black similarity: {sim:.4f}")
if sim > 0.95:
    print("  ⚠ WARNING: encoder may be collapsed")
else:
    print(f"  ✓ Encoder healthy  |  embedding dim = {fw.shape[1]}")

# ── DISCOVER FRAMES ───────────────────────────────────────────────────────────
print(f"\nDiscovering frames in {FRAMES_DIR}...")
frame_paths = sorted(
    [str(p) for p in Path(FRAMES_DIR).rglob("*.jpg")] +
    [str(p) for p in Path(FRAMES_DIR).rglob("*.png")]
)
print(f"  Found {len(frame_paths)} frames")
if not frame_paths:
    raise RuntimeError(f"No frames found under {FRAMES_DIR}")

# ── ENCODE ────────────────────────────────────────────────────────────────────
all_embeddings = []
valid_paths    = []
skipped        = 0

print(f"\nEncoding frames (batch={BATCH_SIZE}, device={DEVICE})...")
print("  Each frame is repeated x4 to match TimeSformer's temporal dimension.")
print("  Expect ~20-40 min for 13K frames on CPU.")

for i in tqdm(range(0, len(frame_paths), BATCH_SIZE)):
    batch_paths = frame_paths[i : i + BATCH_SIZE]
    tensors     = []
    batch_valid = []

    for p in batch_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(make_clip_tensor(img))   # (1, 4, 3, 224, 224)
            batch_valid.append(p)
        except Exception as e:
            skipped += 1
            continue

    if not tensors:
        continue

    # Stack along batch dim → (B, 4, 3, 224, 224)
    batch_tensor = torch.cat(tensors, dim=0).to(DEVICE)

    with torch.no_grad():
        feats = model.encode_image(batch_tensor)           # (B, 256)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # normalise

    all_embeddings.append(feats.cpu().numpy().astype("float32"))
    valid_paths.extend(batch_valid)

# ── SAVE ──────────────────────────────────────────────────────────────────────
embeddings = np.vstack(all_embeddings)

assert len(valid_paths) == embeddings.shape[0], \
    f"Mismatch: {len(valid_paths)} paths vs {embeddings.shape[0]} embeddings"

# Diversity check
n        = min(200, len(embeddings))
sample   = embeddings[:n]
mask     = ~np.eye(n, dtype=bool)
mean_sim = (sample @ sample.T)[mask].mean()

np.save(EMBEDDINGS_OUT, embeddings)
with open(PATHS_OUT, "w") as f:
    f.write("\n".join(valid_paths) + "\n")

print(f"\n✓ Done!")
print(f"  Shape             : {embeddings.shape}")
print(f"  Embedding dim     : {embeddings.shape[1]}  (256 for LaViLa BASE)")
print(f"  Skipped frames    : {skipped}")
print(f"  Mean pairwise sim : {mean_sim:.4f}  (target < 0.40)")
if mean_sim > 0.6:
    print("  ⚠ Similarity still high — dataset is very visually homogeneous")
else:
    print("  ✓ Diversity looks healthy — search should work well")
print(f"\n  Saved → {EMBEDDINGS_OUT}")
print(f"  Saved → {PATHS_OUT}")