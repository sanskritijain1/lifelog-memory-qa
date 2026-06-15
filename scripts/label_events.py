"""
label_events.py  —  CLIP zero-shot event labelling

Runs once offline. For each event's center frame, scores it against
a vocabulary of kitchen activity labels using CLIP softmax.
Stores the top label + confidence in events.json.

After running this, memory_qa.py timelines will show:
  Event 1: 00:09:45 → 00:09:55  [cutting vegetables on board]  (conf: 0.42)
  Event 2: 00:10:18 → 00:10:56  [stirring food in pan]         (conf: 0.38)

instead of just timestamps.

Runtime: ~2 minutes for 502 events on CPU.
"""

import os, json
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

torch.set_num_threads(1)

EVENTS_PATH     = "data/events.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ── ACTIVITY VOCABULARY ───────────────────────────────────────────────────────
# These labels are scored against each event's center frame.
# More specific = better discrimination. Phrased as captions (CLIP was
# trained on captions, not keyword labels).
ACTIVITY_LABELS = [
    "a person opening or closing a refrigerator",
    "a person washing hands under running water",
    "a person cutting or chopping vegetables with a knife",
    "a person stirring or cooking food in a pan on the stove",
    "a person peeling a vegetable",
    "a person placing food on a plate or bowl",
    "a person pouring liquid into a pot or pan",
    "a person cracking or whisking eggs",
    "a person opening or closing a kitchen cabinet",
    "a person picking up or putting down a bottle",
    "a person throwing something in the bin",
    "a person eating food",
    "a person drying hands with a towel",
    "a person grating or grinding food",
    "a person mixing ingredients in a bowl",
    "a person boiling water in a pot",
    "a person taking food out of the fridge",
    "a person using a microwave",
    "a close-up of a cutting board with vegetables",
    "a close-up of a kitchen counter with ingredients",
    "a person standing at the kitchen sink",
    "a person walking in the kitchen",
]

# ── LOAD CLIP ─────────────────────────────────────────────────────────────────
print("Loading CLIP model...")
model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model.eval()

# Pre-compute all label embeddings once
print(f"Pre-computing {len(ACTIVITY_LABELS)} label embeddings...")
with torch.no_grad():
    label_inputs = processor(
        text=ACTIVITY_LABELS, return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    )
    label_feats = model.get_text_features(**label_inputs)
    label_feats = label_feats / label_feats.norm(dim=-1, keepdim=True)
label_feats_np = label_feats.cpu().numpy().astype("float32")
print("  ✓ Label embeddings ready")

# ── LOAD EVENTS ───────────────────────────────────────────────────────────────
with open(EVENTS_PATH) as f:
    events = json.load(f)
print(f"  Labelling {len(events)} events...")

# ── LABEL EACH EVENT ──────────────────────────────────────────────────────────
def label_frame(frame_path: str) -> tuple[str, float]:
    """
    Score a single frame against all activity labels.
    Returns (best_label, confidence) where confidence is the
    softmax probability of the top label.
    """
    try:
        img = Image.open(frame_path).convert("RGB")
    except Exception:
        return "unknown activity", 0.0

    with torch.no_grad():
        img_inputs = processor(images=img, return_tensors="pt")
        img_feat   = model.get_image_features(**img_inputs)
        img_feat   = img_feat / img_feat.norm(dim=-1, keepdim=True)

    img_np   = img_feat.cpu().numpy().astype("float32")
    sims     = (img_np @ label_feats_np.T)[0]        # cosine sims (V,)
    exp_sims = np.exp(sims - sims.max())
    probs    = exp_sims / exp_sims.sum()              # softmax

    best_idx   = int(probs.argmax())
    best_label = ACTIVITY_LABELS[best_idx]
    confidence = float(probs[best_idx])

    # Strip the "a person " prefix for cleaner display
    display = best_label
    for prefix in ["a person ", "a close-up of "]:
        if display.startswith(prefix):
            display = display[len(prefix):]
            break

    return display, confidence

skipped = 0
for event in tqdm(events):
    # Score the center frame (representative of the event)
    center = event.get("center_path", "")
    if not center:
        event["activity_label"]      = "unknown activity"
        event["activity_confidence"] = 0.0
        skipped += 1
        continue

    label, conf = label_frame(center)
    event["activity_label"]      = label
    event["activity_confidence"] = round(conf, 4)

# ── SAVE ──────────────────────────────────────────────────────────────────────
with open(EVENTS_PATH, "w") as f:
    json.dump(events, f, indent=2)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
from collections import Counter
label_counts = Counter(e["activity_label"] for e in events)

print(f"\n✓ Done! Labelled {len(events) - skipped} events")
print(f"\nTop activity labels found:")
for label, count in label_counts.most_common(10):
    bar = "█" * count
    print(f"  {count:>4}x  {label:<50}  {bar[:40]}")
print(f"\nSaved → {EVENTS_PATH}")
print("\nSample labelled events:")
for ev in events[:5]:
    print(f"  [{ev['event_id']:>4}] {ev['session']}  "
          f"{ev['start_time']} → {ev['end_time']}  "
          f"[{ev['activity_label']}]  "
          f"(conf: {ev['activity_confidence']:.3f})")