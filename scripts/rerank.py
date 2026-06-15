"""
rerank.py  —  Step 3.5: CLIP-based action re-ranker

Fixes applied:
  1. RERANK_ALPHA raised to 0.7 — gives CLIP enough power to reorder
  2. Max-pool over multiple frames per event (not just center frame)
  3. More specific action labels for fridge and other ambiguous actions
"""

import os, sys, json

os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

torch.set_num_threads(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
EVENTS_PATH     = "data/events.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# FIX 1: raised from 0.4 → 0.7 so CLIP can actually reorder candidates
# when LaViLa scores are clustered closely together
RERANK_ALPHA    = 0.7

CANDIDATE_POOL  = 15

# Max frames to sample per event when scoring
# More = more accurate but slower. 5 is a good balance.
MAX_FRAMES_PER_EVENT = 5

# ── ACTION-LEVEL VOCABULARY ───────────────────────────────────────────────────
# FIX 3: more specific labels for ambiguous actions (fridge, bin, sink)
# "a person pulling open a refrigerator with their hand" scores higher
# on the actual opening moment than "a person opening a refrigerator door"
KITCHEN_ACTIONS = [
    # fridge — specific to the pulling/opening motion
    "a person pulling open a refrigerator with their hand",
    "a person closing a refrigerator door",
    "the inside of an open refrigerator with food",
    # bin — specific to the throwing/lid motion
    "a person throwing rubbish into a bin",
    "a person lifting a bin lid with their hand",
    # sink
    "a person washing hands under running water at a sink",
    "a person turning on a kitchen tap",
    # cutting / chopping
    "a person cutting vegetables with a knife on a board",
    "a person chopping food with a knife",
    "a person holding a knife over a cutting board",
    "a close up of a cutting board with chopped vegetables",
    # cooking
    "a person stirring food in a pan on a stove",
    "a person placing a pan on the hob",
    "a person cooking on a gas stove",
    "a person pouring liquid from a bottle into a pot",
    "a person boiling water in a pot",
    # food prep
    "a person peeling a vegetable with a peeler",
    "a person grating cheese",
    "a person mixing ingredients in a bowl",
    "a person cracking an egg into a bowl",
    # serving / eating
    "a person placing food onto a plate",
    "a person eating food with a fork",
    # storage
    "a person opening a kitchen cabinet door",
    "a person taking food out of a cupboard",
    "a person putting food into a microwave",
    # misc
    "a person picking up a bottle from a counter",
    "a person drying hands with a kitchen towel",
    "a close up of kitchen counter with food ingredients",
    "a person standing at a kitchen counter",
]

# ── QUERY → ACTION LABEL MAPPING ─────────────────────────────────────────────
QUERY_TO_ACTION = {
    # fridge — FIX 3: more specific label
    "fridge"       : "a person pulling open a refrigerator with their hand",
    "refrigerator" : "a person pulling open a refrigerator with their hand",
    # bin
    "bin"          : "a person throwing rubbish into a bin",
    "trash"        : "a person throwing rubbish into a bin",
    "rubbish"      : "a person lifting a bin lid with their hand",
    "throw"        : "a person throwing rubbish into a bin",
    # sink
    "sink"         : "a person washing hands under running water at a sink",
    "wash"         : "a person washing hands under running water at a sink",
    "hands"        : "a person washing hands under running water at a sink",
    "tap"          : "a person turning on a kitchen tap",
    # cutting
    "cut"          : "a person cutting vegetables with a knife on a board",
    "chop"         : "a person chopping food with a knife",
    "knife"        : "a person holding a knife over a cutting board",
    "board"        : "a close up of a cutting board with chopped vegetables",
    "slice"        : "a person cutting vegetables with a knife on a board",
    # cooking
    "cook"         : "a person stirring food in a pan on a stove",
    "stove"        : "a person placing a pan on the hob",
    "hob"          : "a person placing a pan on the hob",
    "pan"          : "a person stirring food in a pan on a stove",
    "pot"          : "a person boiling water in a pot",
    "stir"         : "a person stirring food in a pan on a stove",
    "boil"         : "a person boiling water in a pot",
    "pour"         : "a person pouring liquid from a bottle into a pot",
    # food prep
    "peel"         : "a person peeling a vegetable with a peeler",
    "grate"        : "a person grating cheese",
    "mix"          : "a person mixing ingredients in a bowl",
    "crack"        : "a person cracking an egg into a bowl",
    "egg"          : "a person cracking an egg into a bowl",
    # serving / eating
    "plate"        : "a person placing food onto a plate",
    "bowl"         : "a person mixing ingredients in a bowl",
    "eat"          : "a person eating food with a fork",
    # storage
    "cabinet"      : "a person opening a kitchen cabinet door",
    "cupboard"     : "a person taking food out of a cupboard",
    "microwave"    : "a person putting food into a microwave",
    # misc
    "bottle"       : "a person picking up a bottle from a counter",
    "water"        : "a person pouring liquid from a bottle into a pot",
    "dry"          : "a person drying hands with a kitchen towel",
    "towel"        : "a person drying hands with a kitchen towel",
    "onion"        : "a person cutting vegetables with a knife on a board",
    "tomato"       : "a person cutting vegetables with a knife on a board",
    "vegetable"    : "a person cutting vegetables with a knife on a board",
    "vegetables"   : "a person cutting vegetables with a knife on a board",
    "cheese"       : "a person grating cheese",
}

# ── LOAD CLIP ─────────────────────────────────────────────────────────────────
print("Loading CLIP re-ranker...")
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

print("  Pre-computing action vocabulary embeddings...")
with torch.no_grad():
    vocab_inputs = clip_processor(
        text=KITCHEN_ACTIONS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    vocab_feats = clip_model.get_text_features(**vocab_inputs)
    vocab_feats = vocab_feats / vocab_feats.norm(dim=-1, keepdim=True)
vocab_feats_np = vocab_feats.cpu().numpy().astype("float32")
print(f"  ✓ CLIP re-ranker ready  |  {len(KITCHEN_ACTIONS)} action labels")

with open(EVENTS_PATH) as f:
    events    = json.load(f)
event_map = {e["event_id"]: e for e in events}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def extract_action(query: str) -> str | None:
    q_lower = query.lower()
    for keyword, action_label in QUERY_TO_ACTION.items():
        if keyword in q_lower:
            return action_label
    return None

def clip_score_frame(frame_path: str, target_action: str) -> float:
    """
    Score a single frame against target_action using softmax over
    the full action vocabulary for robust relative scoring.
    """
    try:
        img = Image.open(frame_path).convert("RGB")
    except Exception:
        return 0.0

    with torch.no_grad():
        img_inputs = clip_processor(images=img, return_tensors="pt")
        img_feat   = clip_model.get_image_features(**img_inputs)
        img_feat   = img_feat / img_feat.norm(dim=-1, keepdim=True)

    img_np   = img_feat.cpu().numpy().astype("float32")
    sims     = (img_np @ vocab_feats_np.T)[0]
    exp_sims = np.exp(sims - sims.max())
    probs    = exp_sims / exp_sims.sum()

    try:
        target_idx = KITCHEN_ACTIONS.index(target_action)
        return float(probs[target_idx])
    except ValueError:
        with torch.no_grad():
            t_inputs = clip_processor(
                text=[target_action], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            )
            t_feat = clip_model.get_text_features(**t_inputs)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        return float((img_feat @ t_feat.T)[0, 0].item())

def clip_score_event(event: dict, target_action: str) -> float:
    """
    FIX 2: Score an event by its BEST frame across up to MAX_FRAMES_PER_EVENT
    sampled frames — not just the center frame.

    The key moment (e.g. hand touching fridge handle) may last only 1 second
    and could easily fall outside the center frame. Max-pooling ensures we
    catch it wherever it appears in the event.
    """
    frame_paths = event["frame_paths"]

    if len(frame_paths) <= MAX_FRAMES_PER_EVENT:
        sampled = frame_paths
    else:
        # Evenly spaced sample across the event
        step    = len(frame_paths) // MAX_FRAMES_PER_EVENT
        sampled = frame_paths[::step][:MAX_FRAMES_PER_EVENT]

    scores = [clip_score_frame(p, target_action) for p in sampled]
    return max(scores)   # max not mean — best moment wins

# ── RE-RANKER ─────────────────────────────────────────────────────────────────
def rerank(
    candidates : list[dict],
    query      : str,
    alpha      : float = RERANK_ALPHA,
    top_k      : int   = 5,
) -> list[dict]:
    """
    Re-rank LaViLa retrieval candidates using CLIP action scoring.
    Uses max-pooled frame scores and action-specific labels.
    """
    target_action = extract_action(query)

    if target_action is None:
        for r in candidates:
            r["lavila_score"]    = r["score"]
            r["clip_score"]      = None
            r["final_score"]     = r["score"]
            r["detected_action"] = None
        return candidates[:top_k]

    print(f"  Re-ranking on action: '{target_action}'")

    scored = []
    for candidate in candidates:
        event = event_map.get(candidate["event_id"])
        if event is None:
            continue

        # FIX 2: max-pool over sampled frames instead of center only
        clip_s   = clip_score_event(event, target_action)
        lavila_s = (candidate["score"] + 1) / 2   # normalise [-1,1] → [0,1]
        final_s  = (1 - alpha) * lavila_s + alpha * clip_s

        scored.append({
            **candidate,
            "lavila_score"   : round(candidate["score"], 4),
            "clip_score"     : round(clip_s, 4),
            "final_score"    : round(final_s, 4),
            "detected_action": target_action,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    for i, r in enumerate(scored):
        r["rank"] = i + 1

    return scored[:top_k]