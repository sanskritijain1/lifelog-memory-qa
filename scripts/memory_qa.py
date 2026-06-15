"""
memory_qa.py  —  Unified Memory QA

Query types (regex-classified, reliable on any hardware):
  factual   — "when did someone open the fridge?"
  anchor    — "what happened before/after opening the fridge?"
  timerange — "what happened between minute 5 and 10 in P01_09?"
  counting  — "how many times did he wash his hands?"
  comparison— "compare P01_09 and P30_107"
  summary   — "summarise session P01_09"

Uses BLIP-2 captions (from events.json) for rich timeline context.
Query expansion uses hardcoded rules — fast and reliable on CPU.
"""

import os, sys, json, gc, re
from pathlib import Path
from collections import defaultdict

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
import requests
from transformers import CLIPTokenizer

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = "data/frame_embeddings.npy"
PATHS_FILE      = "data/frame_paths.txt"
MEAN_PATH       = "data/frame_mean.npy"
EVENTS_PATH     = "data/events.json"
TIMESTAMPS_PATH = "data/session_timestamps.json"
CHECKPOINT_PATH = "pretrained/lavila_tsf_base_ep5.pth"

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3"

FRAME_POOL      = 50
SCORE_THRESHOLD = 0.45
TOP_K           = 5
BEFORE_WINDOW   = 120
AFTER_WINDOW    = 120
DEVICE          = "cpu"
NUM_FRAMES      = 4

# ── LOAD SYSTEM ───────────────────────────────────────────────────────────────
print("Loading LaViLa retrieval system...")
from lavila.models import models as lavila_models

ckpt       = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
model      = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()

tokenizer  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

with open(PATHS_FILE) as f:
    paths = [l.strip() for l in f if l.strip()]

mean_vec   = np.load(MEAN_PATH).astype("float32")
embeddings = embeddings - mean_vec
faiss.normalize_L2(embeddings)

frame_index = faiss.IndexFlatIP(embeddings.shape[1])
frame_index.add(embeddings)

with open(EVENTS_PATH) as f:
    events = json.load(f)

frame_to_event = {}
session_events = defaultdict(list)
for ev in events:
    session_events[ev["session"]].append(ev)
    for fp in ev["frame_paths"]:
        fn = int(Path(fp).stem.split("_")[1])
        frame_to_event[(ev["session"], fn)] = ev

for s in session_events:
    session_events[s].sort(key=lambda e: e.get("start_s", e["start_frame"]))

with open(TIMESTAMPS_PATH) as f:
    session_ts = json.load(f)

print(f"  ✓ {len(events)} events  |  {len(session_events)} sessions")

del model; gc.collect()
print("  ✓ Vision backbone unloaded")

print("  Loading text encoder...")
_ckpt  = torch.load(CHECKPOINT_PATH, map_location="cpu")
_sd    = {k.replace("module.", ""): v for k, v in _ckpt["state_dict"].items()}
_model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
_model.load_state_dict(_sd, strict=False)
_model.eval()
text_encoder = _model
del _ckpt, _sd; gc.collect()
print("  ✓ Text encoder ready")

# ── QUERY EXPANSION ───────────────────────────────────────────────────────────
EXPANSION_RULES = [
    (["fridge","refrigerator","cold","milk","dairy"],
     ["open the fridge","take food out of the fridge","close the fridge door"]),
    (["bin","trash","rubbish","throw","garbage"],
     ["throw rubbish in the bin","open the bin lid","put trash in the bin"]),
    (["wash","sink","hands","tap","rinse"],
     ["wash hands in the sink","turn on the tap","rinse hands under water"]),
    (["cut","chop","knife","board","slice","dice"],
     ["cut vegetables on the board","chop food with a knife",
      "slice ingredients on the cutting board"]),
    (["cook","stove","hob","pan","pot","stir","boil","fry"],
     ["stir food in the pan","cook on the stove","boil water in the pot"]),
    (["plate","dish","serve"],
     ["put food on a plate","use a white plate","place food on the dish"]),
    (["peel","vegetable","onion","tomato","potato"],
     ["peel a vegetable","prepare vegetables","peel potatoes"]),
    (["knead","dough","bread","mix","batter"],
     ["knead dough on the counter","mix ingredients in a bowl",
      "prepare dough for baking"]),
    (["eat","meal","food"],
     ["eat food","pick up food to eat","have a meal"]),
    (["pour","water","liquid","bottle"],
     ["pour water into the pot","pour liquid from a bottle",
      "fill the pot with water"]),
    (["egg","crack","whisk"],
     ["crack an egg into a bowl","break an egg","whisk eggs in a bowl"]),
    (["cabinet","cupboard","shelf","drawer"],
     ["open kitchen cabinet","take item from shelf",
      "open drawer in kitchen"]),
    (["towel","dry","wipe"],
     ["dry hands with a towel","wipe hands on kitchen towel",
      "pat hands dry"]),
]

def expand_query(q: str) -> list[str]:
    ql = q.lower()
    for keywords, expansions in EXPANSION_RULES:
        if any(kw in ql for kw in keywords):
            return expansions
    return [q]

# ── INTENT CLASSIFIER (regex) ─────────────────────────────────────────────────
def classify_intent(query: str) -> dict:
    q = query.lower()
    session_m = re.search(r'\bP\d+_\d+\b', query)
    session   = session_m.group(0) if session_m else None

    # counting
    if re.search(r'\bhow many times\b|\bcount\b|\bnumber of times\b|\bhow often\b', q):
        return {"type": "counting", "session": session,
                "anchor_phrase": query, "direction": None,
                "start_s": None, "end_s": None}

    # comparison
    if re.search(r'\bcompare\b|\bvs\b|\bversus\b|\bdifference between\b', q):
        return {"type": "comparison", "session": session,
                "anchor_phrase": None, "direction": None,
                "start_s": None, "end_s": None}

    # summary — also catch "summarise/summary session X" and bare session queries
    if re.search(r'\bsummar\b|\beverything\b|\boverview\b|\ball events\b|\bwhole session\b', q):
        return {"type": "summary", "session": session,
                "anchor_phrase": None, "direction": None,
                "start_s": None, "end_s": None}
    # catch "session P01_09" without a more specific intent
    if session and re.search(r'\bsession\b', q) and not any(
        kw in q for kw in ["before","after","around","between","minute","when","how many","compare"]
    ):
        return {"type": "summary", "session": session,
                "anchor_phrase": None, "direction": None,
                "start_s": None, "end_s": None}

    # timerange — explicit range
    range_m = re.search(
        r'between\s+(?:minute\s+)?(\d+)\s+and\s+(?:minute\s+)?(\d+)', q)
    if range_m:
        return {"type": "timerange", "session": session,
                "anchor_phrase": None, "direction": None,
                "start_s": int(range_m.group(1)) * 60,
                "end_s"  : int(range_m.group(2)) * 60}

    # timerange — around minute X
    around_m = re.search(
        r'(?:around|at)\s+(?:minute\s+)?(\d+)|(\d+)\s*minutes?\s+(?:in|into)', q)
    if around_m:
        val = int(around_m.group(1) or around_m.group(2))
        mid = val * 60
        return {"type": "timerange", "session": session,
                "anchor_phrase": None, "direction": None,
                "start_s": max(0, mid - 60), "end_s": mid + 60}

    # anchor — before/after/around an event
    before_m = re.search(r'\b(before|prior to)\b\s+(.+?)(?:\?|$)', q)
    after_m  = re.search(r'\b(after|following)\b\s+(.+?)(?:\?|$)', q)
    around_m2= re.search(r'\b(around|during|while)\b\s+(.+?)(?:\?|$)', q)

    if before_m:
        return {"type": "anchor", "direction": "before",
                "anchor_phrase": before_m.group(2).strip(),
                "session": session, "start_s": None, "end_s": None}
    if after_m:
        return {"type": "anchor", "direction": "after",
                "anchor_phrase": after_m.group(2).strip(),
                "session": session, "start_s": None, "end_s": None}
    if around_m2:
        return {"type": "anchor", "direction": "around",
                "anchor_phrase": around_m2.group(2).strip(),
                "session": session, "start_s": None, "end_s": None}

    # factual — default
    return {"type": "factual", "session": session,
            "anchor_phrase": None, "direction": None,
            "start_s": None, "end_s": None}

# ── CORE RETRIEVAL ────────────────────────────────────────────────────────────
def encode_and_search(phrase: str, session_filter: str = None,
                      top_k: int = FRAME_POOL) -> list[dict]:
    narrations = expand_query(phrase)
    tokens = tokenizer(
        narrations, return_tensors="pt",
        padding="max_length", truncation=True, max_length=77,
    )
    with torch.no_grad():
        feat = text_encoder.encode_text(tokens["input_ids"])
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec = feat.mean(dim=0, keepdim=True).cpu().numpy().astype("float32")
    vec = vec - mean_vec
    faiss.normalize_L2(vec)

    D, I = frame_index.search(vec, top_k)
    seen = {}
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        fp  = paths[idx]
        p   = Path(fp)
        ses = p.parent.name
        fn  = int(p.stem.split("_")[1])
        if session_filter and ses != session_filter: continue
        ev  = frame_to_event.get((ses, fn))
        if not ev: continue
        eid = ev["event_id"]
        if eid not in seen or score > seen[eid]["score"]:
            seen[eid] = {
                "score"      : float(score),
                "event_id"   : eid,
                "session"    : ev["session"],
                "start_time" : ev["start_time"],
                "end_time"   : ev["end_time"],
                "start_s"    : ev.get("start_s", 0),
                "duration_s" : ev["duration_s"],
                "frame_count": ev["frame_count"],
                "best_frame" : fp,
                "caption"    : ev.get("blip2_caption","").strip() or
                               ev.get("activity_label",""),
            }
    results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results): r["rank"] = i + 1
    return results

def get_events_in_window(session: str, start_s: float,
                         end_s: float) -> list[dict]:
    out = []
    for ev in session_events.get(session, []):
        es = ev.get("start_s", ev["start_frame"])
        ee = ev.get("end_s",   ev["end_frame"])
        if es <= end_s and ee >= start_s:
            out.append(ev)
    return sorted(out, key=lambda e: e.get("start_s", e["start_frame"]))

def format_event(ev: dict, index: int, anchor_id: int = None) -> str:
    marker  = " ← [ANCHOR]" if anchor_id and ev["event_id"] == anchor_id else ""
    caption = ev.get("blip2_caption","").strip() or ev.get("activity_label","")
    cap_str = f"  [{caption}]" if caption else ""
    return (f"  Event {index}: {ev['start_time']} → {ev['end_time']}  "
            f"({ev.get('duration_s',0):.0f}s){cap_str}  [{ev['session']}]{marker}")

# ── OLLAMA ────────────────────────────────────────────────────────────────────
def check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return any(OLLAMA_MODEL in m["name"]
                   for m in r.json().get("models", []))
    except:
        return False

def call_llm(prompt: str, system: str, max_tokens: int = 400) -> str:
    payload = {
        "model"  : OLLAMA_MODEL,
        "prompt" : prompt,
        "system" : system,
        "stream" : False,
        "options": {"temperature": 0.2, "num_predict": max_tokens,
                    "num_ctx": 2048, "num_thread": 4}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except:
        return "(LLM unavailable)"

# ── HANDLERS ─────────────────────────────────────────────────────────────────
def handle_factual(question, intent, ollama_ready):
    results = encode_and_search(question, intent.get("session"))
    results = [r for r in results if r["score"] >= SCORE_THRESHOLD][:TOP_K]

    if not results:
        print("  No confident matches found. Try rephrasing.")
        return

    print(f"\n  Top {len(results)} matching events:")
    for r in results:
        cap = f"  [{r['caption']}]" if r.get("caption") else ""
        print(f"    [{r['rank']}] {r['session']}  "
              f"{r['start_time']} → {r['end_time']}  "
              f"({r['duration_s']:.0f}s)  score={r['score']:.4f}{cap}")

    if not ollama_ready: return

    context = (
        "Retrieved memory entries ranked by confidence "
        "(Memory 1 = highest confidence, trust it most):\n\n" +
        "\n\n".join([
            f"  Memory {r['rank']} (confidence {r['score']:.3f}):\n"
            f"    Session  : {r['session']}\n"
            f"    Time     : {r['start_time']} → {r['end_time']} ({r['duration_s']:.0f}s)\n"
            f"    Activity : {r.get('caption') or 'unknown'}"
            for r in results
        ])
    )
    system = (
        "You are a memory assistant for egocentric kitchen video. "
        "Answer using ONLY what the activity captions explicitly say. "
        "Do NOT infer reasons, intentions, or objects not mentioned in the captions. "
        "If the caption says 'opening refrigerator door' do not say 'to get milk' — "
        "only report what is stated. "
        "Memory 1 has the highest confidence — report it as the primary answer. "
        "Also mention other instances if captions clearly match the query. "
        "Report timestamps exactly. Never invent or infer details."
    )
    print("\n  Generating answer...")
    answer = call_llm(f"{context}\n\nQuestion: {question}\n\nAnswer:", system)
    print(f"\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print(f"  ────────────────────────────────────────────────")


def handle_anchor(question, intent, ollama_ready):
    phrase = intent.get("anchor_phrase") or question
    print(f"  Finding anchor event: '{phrase}'...")
    results = encode_and_search(phrase, intent.get("session"))
    results = [r for r in results if r["score"] >= SCORE_THRESHOLD]

    if not results:
        print("  ✗ Could not find anchor event. Try rephrasing.")
        return

    anchor    = results[0]
    anchor_s  = anchor["start_s"]
    anchor_ses= anchor["session"]
    print(f"  ✓ Anchor: {anchor_ses}  "
          f"{anchor['start_time']} → {anchor['end_time']}  "
          f"score={anchor['score']:.4f}")
    if anchor.get("caption"):
        print(f"    Caption: {anchor['caption']}")

    direction = intent.get("direction") or "around"
    if direction == "before":
        win_start, win_end = anchor_s - BEFORE_WINDOW, anchor_s
    elif direction == "after":
        win_start = anchor_s + anchor.get("duration_s", 0)
        win_end   = win_start + AFTER_WINDOW
    else:
        win_start = anchor_s - BEFORE_WINDOW // 2
        win_end   = anchor_s + AFTER_WINDOW // 2

    window_events = get_events_in_window(
        anchor_ses, max(0, win_start), win_end
    )

    if not window_events:
        print("  ✗ No events found in the temporal window.")
        return

    timeline = "\n".join([
        format_event(ev, i+1, anchor["event_id"])
        for i, ev in enumerate(window_events)
    ])

    print(f"\n  ── Timeline ({len(window_events)} events, "
          f"{direction} anchor) ────────────")
    print(timeline)
    print("  " + "─" * 52)

    if not ollama_ready: return

    direction_instruction = {
        "before": (
            "Summarise ALL events that occurred BEFORE the [ANCHOR] event "
            "in chronological order. List every event with its timestamp — "
            "do not skip any."
        ),
        "after": (
            "Summarise ALL events that occurred AFTER the [ANCHOR] event "
            "in chronological order. List every event with its timestamp — "
            "do not skip any."
        ),
        "around": (
            "Describe the full sequence of events around the [ANCHOR], "
            "including what happened before and after it."
        ),
    }.get(direction, "Describe all events chronologically.")

    context = (
        f"Chronological timeline ({direction} the anchor event):\n\n"
        f"{timeline}\n\n"
        f"Anchor event: {anchor['start_time']} → {anchor['end_time']}"
        + (f"  [{anchor['caption']}]" if anchor.get("caption") else "") +
        f"\n\n{direction_instruction}"
    )
    system = (
        "You are a memory assistant for egocentric kitchen video. "
        "You have a chronological timeline with AI-generated activity captions. "
        "Describe every event listed — do not skip any. "
        "Reference specific timestamps. "
        "IMPORTANT: captions for very short events (under 8 seconds) may be "
        "inaccurate — treat them with caution and note the uncertainty. "
        "Never invent events not in the timeline. "
        "Never mention specific foods or objects unless they appear verbatim "
        "in a caption."
    )
    print("\n  Reasoning over timeline...")
    answer = call_llm(
        f"{context}\n\nQuestion: {question}\n\nAnswer:", system
    )
    print(f"\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print(f"  ────────────────────────────────────────────────")


def handle_timerange(question, intent, ollama_ready):
    start_s = intent.get("start_s") or 0
    end_s   = intent.get("end_s")   or 600
    session = intent.get("session")

    print(f"  Time range: {start_s//60}:{start_s%60:02d} → "
          f"{end_s//60}:{end_s%60:02d}"
          + (f"  session: {session}" if session else "  all sessions"))

    if session:
        window_events = get_events_in_window(session, start_s, end_s)
    else:
        window_events = []
        for ses in session_events:
            window_events.extend(get_events_in_window(ses, start_s, end_s))
        window_events.sort(key=lambda e: e.get("start_s", e["start_frame"]))

    if not window_events:
        print("  ✗ No events found in this time range.")
        return

    timeline = "\n".join([
        format_event(ev, i+1) for i, ev in enumerate(window_events)
    ])
    print(f"\n  ── Timeline ({len(window_events)} events) ──────────────────")
    print(timeline)
    print("  " + "─" * 52)

    if not ollama_ready: return

    system = (
        "You are a memory assistant. Summarise what happened during the "
        "requested time period using ALL events in the timeline. "
        "List each activity with its timestamp. "
        "Never invent events not in the timeline."
    )
    answer = call_llm(
        f"Timeline:\n{timeline}\n\nQuestion: {question}\n\nAnswer:", system
    )
    print(f"\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print(f"  ────────────────────────────────────────────────")


def handle_counting(question, intent, ollama_ready):
    """
    Count occurrences by scanning ALL events in a session directly,
    not by embedding search (which only returns top-K and misses many).

    If no session specified, asks the user which session to count in,
    or counts across all sessions.
    """
    session = intent.get("session")

    # Extract what activity to count from the question
    # e.g. "how many times did he wash his hands" → "wash hands"
    activity_keywords = []
    for keywords, _ in EXPANSION_RULES:
        if any(kw in question.lower() for kw in keywords):
            activity_keywords = keywords
            break

    # If no session specified, count across all sessions
    target_sessions = [session] if session else list(session_events.keys())

    print(f"  Counting in sessions: {target_sessions}")
    print(f"  Activity keywords: {activity_keywords or ['(from question)']}")

    # Scan ALL events in target sessions, filter by caption
    matching_events = []
    for ses in target_sessions:
        for ev in session_events.get(ses, []):
            caption = (ev.get("blip2_caption","") or
                      ev.get("activity_label","")).lower()
            if not caption:
                continue
            # Match if any keyword appears in caption OR question words appear
            q_words = [w for w in question.lower().split()
                      if len(w) > 3 and w not in
                      {"many","times","often","count","does","when","what","have","been","did"}]
            if (any(kw in caption for kw in activity_keywords) or
                any(w in caption for w in q_words)):
                matching_events.append({
                    "session"   : ev["session"],
                    "start_time": ev["start_time"],
                    "end_time"  : ev["end_time"],
                    "duration_s": ev.get("duration_s", 0),
                    "caption"   : caption,
                })

    if not matching_events:
        print("  No matching events found in session event captions.")
        print("  Falling back to embedding search...")
        results = encode_and_search(question, session)
        results = [r for r in results if r["score"] >= SCORE_THRESHOLD]
        matching_events = [{
            "session"   : r["session"],
            "start_time": r["start_time"],
            "end_time"  : r["end_time"],
            "duration_s": r["duration_s"],
            "caption"   : r.get("caption","unknown"),
        } for r in results]

    print(f"\n  Found {len(matching_events)} matching events:")
    for ev in matching_events:
        print(f"    {ev['session']}  {ev['start_time']} → {ev['end_time']}  "
              f"[{ev['caption']}]")

    if not ollama_ready: return

    context = (
        f"All events matching the query (scanned from complete session data):\n\n" +
        "\n".join([
            f"  {ev['session']}  {ev['start_time']} → {ev['end_time']}  "
            f"({ev['duration_s']:.0f}s)  [{ev['caption']}]"
            for ev in matching_events
        ])
    )
    system = (
        "You are a memory assistant. Count the distinct occurrences listed above. "
        "Each line is one event. Count only lines where the caption clearly matches "
        "the queried activity. Give a total count and list the timestamps."
    )
    answer = call_llm(f"{context}\n\nQuestion: {question}\n\nAnswer:", system)
    print(f"\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print(f"  ────────────────────────────────────────────────")


def handle_comparison(question, intent, ollama_ready):
    sessions_mentioned = re.findall(r'\bP\d+_\d+\b', question)

    if len(sessions_mentioned) >= 2:
        timelines = {}
        for ses in sessions_mentioned[:2]:
            evs = session_events.get(ses, [])
            timelines[ses] = "\n".join([
                format_event(ev, i+1) for i, ev in enumerate(evs[:15])
            ])
    else:
        results = encode_and_search(question)
        results = [r for r in results if r["score"] >= SCORE_THRESHOLD]
        timelines = defaultdict(list)
        for r in results:
            timelines[r["session"]].append(
                f"  {r['start_time']} → {r['end_time']}  "
                f"[{r.get('caption','unknown')}]"
            )
        timelines = {k: "\n".join(v) for k, v in timelines.items()}

    if not timelines:
        print("  No events found for comparison.")
        return

    context = "\n\n".join([
        f"Session {ses}:\n{tl}" for ses, tl in timelines.items()
    ])
    print(f"\n  Comparing {list(timelines.keys())}...")

    if not ollama_ready:
        print(context)
        return

    system = (
        "You are a memory assistant. Compare the activities across the "
        "provided sessions using ONLY what the captions explicitly state. "
        "Do not mention specific foods, objects, or activities unless they "
        "appear verbatim in a caption. "
        "Note similarities, differences, and patterns. "
        "Reference specific timestamps. "
        "Never invent details not present in the captions."
    )
    answer = call_llm(f"{context}\n\nQuestion: {question}\n\nAnswer:", system)
    print(f"\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print(f"  ────────────────────────────────────────────────")


def handle_summary(question, intent, ollama_ready):
    session = intent.get("session") or \
              (re.search(r'\bP\d+_\d+\b', question) or
               type('', (), {'group': lambda s, n: None})()).group(0)

    if not session:
        session = list(session_events.keys())[0]

    evs = session_events.get(session, [])
    if not evs:
        print(f"  No events found for session {session}.")
        return

    timeline = "\n".join([
        format_event(ev, i+1) for i, ev in enumerate(evs)
    ])
    duration = evs[-1].get("end_time", "?")
    print(f"\n  Summarising {session} ({len(evs)} events, {duration} total)...")

    if not ollama_ready:
        print(timeline)
        return

    system = (
        "You are a memory assistant. Provide a structured summary of "
        "everything that happened in this kitchen session. "
        "Group related activities. Note the overall cooking flow with timestamps."
    )
    answer = call_llm(
        f"Full timeline for {session}:\n{timeline}\n\n"
        f"Question: {question}\n\nAnswer:", system, max_tokens=600
    )
    print(f"\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print(f"  ────────────────────────────────────────────────")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("\nChecking Ollama...")
ollama_ready = check_ollama()
print(f"  {'✓ Ollama ready  |  model: ' + OLLAMA_MODEL if ollama_ready else '⚠ Ollama not running — showing timelines only'}")

HANDLERS = {
    "factual"   : handle_factual,
    "anchor"    : handle_anchor,
    "timerange" : handle_timerange,
    "counting"  : handle_counting,
    "comparison": handle_comparison,
    "summary"   : handle_summary,
}

print("\n" + "=" * 62)
print("  LIFELOG MEMORY QA  —  LaViLa + Llama + BLIP-2 captions")
print("=" * 62)
print("  Factual:    'When did someone open the fridge?'")
print("              'When did he get milk?'")
print("  Anchor:     'What happened before opening the fridge?'")
print("              'What was the person doing after washing hands?'")
print("  Time range: 'What happened between minute 5 and 10 in P01_09?'")
print("  Counting:   'How many times did he wash his hands?'")
print("  Comparison: 'Compare P01_09 and P30_107'")
print("  Summary:    'Summarise session P01_09'")
print("  Type 'exit' to quit.")
print("=" * 62)

while True:
    try:
        question = input("\n  Question > ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if not question:
        continue
    if question.lower() in {"exit", "quit", "q"}:
        print("Exiting.")
        break

    try:
        intent = classify_intent(question)
        print(f"  Query type: {intent['type'].upper()}"
              + (f"  direction: {intent['direction']}"
                 if intent.get("direction") else "")
              + (f"  session: {intent['session']}"
                 if intent.get("session") else ""))

        HANDLERS[intent["type"]](question, intent, ollama_ready)

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()