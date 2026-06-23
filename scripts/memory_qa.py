"""
memory_qa.py — Unified Memory QA

Fix:
- Factual/anchor retrieval now preserves raw LaViLa search behavior.
- No aggressive rule-based query expansion during retrieval.
- Uses raw query + cleaned query variants separately, then merges results.
"""

import os, sys, json, gc, re
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

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
PATHS_FILE = "data/frame_paths.txt"
MEAN_PATH = "data/frame_mean.npy"
EVENTS_PATH = "data/events.json"
TIMESTAMPS_PATH = "data/session_timestamps.json"
CHECKPOINT_PATH = "pretrained/lavila_tsf_base_ep5.pth"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

FRAME_POOL = 50
SCORE_THRESHOLD = 0.45
TOP_K = 5
BEFORE_WINDOW = 120
AFTER_WINDOW = 120
DEVICE = "cpu"
NUM_FRAMES = 4

# ── LOAD SYSTEM ───────────────────────────────────────────────────────────────
print("Loading LaViLa retrieval system...")
from lavila.models import models as lavila_models

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}

model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

with open(PATHS_FILE) as f:
    paths = [l.strip() for l in f if l.strip()]

mean_vec = np.load(MEAN_PATH).astype("float32")
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

del model
gc.collect()

print("  Loading text encoder...")
_ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
_sd = {k.replace("module.", ""): v for k, v in _ckpt["state_dict"].items()}

_model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
_model.load_state_dict(_sd, strict=False)
_model.eval()

text_encoder = _model

del _ckpt, _sd
gc.collect()

print("  ✓ Text encoder ready")

# ── LIGHTWEIGHT QUERY CLEANING ────────────────────────────────────────────────
def clean_query_for_lavila(q: str) -> str:
    """
    Converts question-style text into LaViLa-style action text,
    without replacing specific objects.

    Example:
    "when did someone cut onions?" → "cut onions"
    "when was a white plate used?" → "white plate used"
    """
    q = q.lower().strip().replace("?", "")

    remove_patterns = [
        r"\bwhen did\b",
        r"\bwhen was\b",
        r"\bwhere did\b",
        r"\bwhat time did\b",
        r"\bsomeone\b",
        r"\bthe person\b",
        r"\bhe\b",
        r"\bshe\b",
        r"\bthey\b",
        r"\bwas\b",
        r"\bdid\b",
        r"\bdoing\b",
    ]

    for pattern in remove_patterns:
        q = re.sub(pattern, "", q)

    return " ".join(q.split())


def build_query_variants(q: str) -> list[str]:
    """
    Use raw query + cleaned query.
    Do NOT use hardcoded semantic expansion for retrieval.
    """
    raw = q.strip()
    cleaned = clean_query_for_lavila(q)

    variants = []

    if raw:
        variants.append(raw)

    if cleaned and cleaned not in variants:
        variants.append(cleaned)

    return variants

# ── INTENT CLASSIFIER ─────────────────────────────────────────────────────────
def classify_intent(query: str) -> dict:
    q = query.lower()
    session_m = re.search(r"\bP\d+_\d+\b", query)
    session = session_m.group(0) if session_m else None

    if re.search(r"\bhow many times\b|\bcount\b|\bnumber of times\b|\bhow often\b", q):
        return {
            "type": "counting",
            "session": session,
            "anchor_phrase": query,
            "direction": None,
            "start_s": None,
            "end_s": None,
        }

    if re.search(r"\bcompare\b|\bvs\b|\bversus\b|\bdifference between\b", q):
        return {
            "type": "comparison",
            "session": session,
            "anchor_phrase": None,
            "direction": None,
            "start_s": None,
            "end_s": None,
        }

    if re.search(r"\bsummar\b|\beverything\b|\boverview\b|\ball events\b|\bwhole session\b", q):
        return {
            "type": "summary",
            "session": session,
            "anchor_phrase": None,
            "direction": None,
            "start_s": None,
            "end_s": None,
        }

    if session and re.search(r"\bsession\b", q) and not any(
        kw in q for kw in ["before", "after", "around", "between", "minute", "when", "how many", "compare"]
    ):
        return {
            "type": "summary",
            "session": session,
            "anchor_phrase": None,
            "direction": None,
            "start_s": None,
            "end_s": None,
        }

    range_m = re.search(
        r"between\s+(?:minute\s+)?(\d+)\s+and\s+(?:minute\s+)?(\d+)", q
    )
    if range_m:
        return {
            "type": "timerange",
            "session": session,
            "anchor_phrase": None,
            "direction": None,
            "start_s": int(range_m.group(1)) * 60,
            "end_s": int(range_m.group(2)) * 60,
        }

    around_m = re.search(
        r"(?:around|at)\s+(?:minute\s+)?(\d+)|(\d+)\s*minutes?\s+(?:in|into)", q
    )
    if around_m:
        val = int(around_m.group(1) or around_m.group(2))
        mid = val * 60

        return {
            "type": "timerange",
            "session": session,
            "anchor_phrase": None,
            "direction": None,
            "start_s": max(0, mid - 60),
            "end_s": mid + 60,
        }

    before_m = re.search(r"\b(before|prior to)\b\s+(.+?)(?:\?|$)", q)
    after_m = re.search(r"\b(after|following)\b\s+(.+?)(?:\?|$)", q)
    around_m2 = re.search(r"\b(around|during|while)\b\s+(.+?)(?:\?|$)", q)

    if before_m:
        return {
            "type": "anchor",
            "direction": "before",
            "anchor_phrase": before_m.group(2).strip(),
            "session": session,
            "start_s": None,
            "end_s": None,
        }

    if after_m:
        return {
            "type": "anchor",
            "direction": "after",
            "anchor_phrase": after_m.group(2).strip(),
            "session": session,
            "start_s": None,
            "end_s": None,
        }

    if around_m2:
        return {
            "type": "anchor",
            "direction": "around",
            "anchor_phrase": around_m2.group(2).strip(),
            "session": session,
            "start_s": None,
            "end_s": None,
        }

    return {
        "type": "factual",
        "session": session,
        "anchor_phrase": None,
        "direction": None,
        "start_s": None,
        "end_s": None,
    }

# ── CORE RETRIEVAL ────────────────────────────────────────────────────────────
def encode_single_query(text: str) -> np.ndarray:
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )

    with torch.no_grad():
        feat = text_encoder.encode_text(tokens["input_ids"])
        feat = feat / feat.norm(dim=-1, keepdim=True)

    vec = feat.cpu().numpy().astype("float32")
    vec = vec - mean_vec
    faiss.normalize_L2(vec)

    return vec


def encode_and_search(
    phrase: str,
    session_filter: str = None,
    top_k: int = FRAME_POOL,
) -> list[dict]:
    """
    Searches raw query and cleaned query separately.
    Merges frame results before mapping to events.
    This preserves LaViLa's direct retrieval quality.
    """
    query_variants = build_query_variants(phrase)
    merged_frames = {}

    for q in query_variants:
        vec = encode_single_query(q)
        D, I = frame_index.search(vec, top_k)

        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue

            fp = paths[idx]

            if fp not in merged_frames or score > merged_frames[fp]["score"]:
                merged_frames[fp] = {
                    "score": float(score),
                    "path": fp,
                    "query_used": q,
                }

    frame_results = sorted(
        merged_frames.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    seen = {}

    for fr in frame_results:
        fp = fr["path"]
        p = Path(fp)
        ses = p.parent.name
        fn = int(p.stem.split("_")[1])

        if session_filter and ses != session_filter:
            continue

        ev = frame_to_event.get((ses, fn))

        if not ev:
            continue

        eid = ev["event_id"]

        if eid not in seen or fr["score"] > seen[eid]["score"]:
            seen[eid] = {
                "score": float(fr["score"]),
                "event_id": eid,
                "session": ev["session"],
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "start_s": ev.get("start_s", 0),
                "duration_s": ev["duration_s"],
                "frame_count": ev["frame_count"],
                "best_frame": fp,
                "query_used": fr["query_used"],
                "caption": ev.get("blip2_caption", "").strip()
                or ev.get("activity_label", ""),
            }

    results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1

    print(f"  Query variants used: {query_variants}")

    return results


def get_events_in_window(session: str, start_s: float, end_s: float) -> list[dict]:
    out = []

    for ev in session_events.get(session, []):
        es = ev.get("start_s", ev["start_frame"])
        ee = ev.get("end_s", ev["end_frame"])

        if es <= end_s and ee >= start_s:
            out.append(ev)

    return sorted(out, key=lambda e: e.get("start_s", e["start_frame"]))


def format_event(ev: dict, index: int, anchor_id: int = None) -> str:
    marker = " ← [ANCHOR]" if anchor_id and ev["event_id"] == anchor_id else ""
    caption = ev.get("blip2_caption", "").strip() or ev.get("activity_label", "")
    cap_str = f"  [{caption}]" if caption else ""

    return (
        f"  Event {index}: {ev['start_time']} → {ev['end_time']}  "
        f"({ev.get('duration_s', 0):.0f}s){cap_str}  [{ev['session']}]{marker}"
    )

# ── OLLAMA ────────────────────────────────────────────────────────────────────
def check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return any(OLLAMA_MODEL in m["name"] for m in r.json().get("models", []))
    except Exception:
        return False


def call_llm(prompt: str, system: str, max_tokens: int = 400) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
            "num_ctx": 2048,
            "num_thread": 4,
        },
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
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

        print(
            f"    [{r['rank']}] {r['session']}  "
            f"{r['start_time']} → {r['end_time']}  "
            f"({r['duration_s']:.0f}s)  score={r['score']:.4f}  "
            f"query='{r.get('query_used', '')}'{cap}"
        )

    if not ollama_ready:
        return

    context = (
        "Retrieved memory entries ranked by visual-language confidence "
        "(Memory 1 = highest confidence):\n\n"
        + "\n\n".join(
            [
                f"  Memory {r['rank']} (confidence {r['score']:.3f}):\n"
                f"    Session  : {r['session']}\n"
                f"    Time     : {r['start_time']} → {r['end_time']} "
                f"({r['duration_s']:.0f}s)\n"
                f"    Query    : {r.get('query_used', '')}\n"
                f"    Caption  : {r.get('caption') or 'unknown'}"
                for r in results
            ]
        )
    )

    system = (
        "You are a memory assistant for egocentric kitchen video. "
        "The retrieved entries are visual-language matches ranked by confidence. "
        "Use Memory 1 as the primary visual match. "
        "Captions are supporting descriptions and may be imperfect. "
        "If the exact queried object is not explicitly confirmed by captions, "
        "say that the visual retrieval suggests a likely match but the caption "
        "does not explicitly confirm the exact object. "
        "Do not invent intentions. Report timestamps exactly."
    )

    print("\n  Generating answer...")

    answer = call_llm(
        f"{context}\n\nQuestion: {question}\n\nAnswer:",
        system,
    )

    print("\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print("  ────────────────────────────────────────────────")


def handle_anchor(question, intent, ollama_ready):
    phrase = intent.get("anchor_phrase") or question

    print(f"  Finding anchor event: '{phrase}'...")

    results = encode_and_search(phrase, intent.get("session"))
    results = [r for r in results if r["score"] >= SCORE_THRESHOLD]

    if not results:
        print("  ✗ Could not find anchor event. Try rephrasing.")
        return

    anchor = results[0]
    anchor_s = anchor["start_s"]
    anchor_ses = anchor["session"]

    print(
        f"  ✓ Anchor: {anchor_ses}  "
        f"{anchor['start_time']} → {anchor['end_time']}  "
        f"score={anchor['score']:.4f}"
    )

    if anchor.get("caption"):
        print(f"    Caption: {anchor['caption']}")

    direction = intent.get("direction") or "around"

    if direction == "before":
        win_start, win_end = anchor_s - BEFORE_WINDOW, anchor_s
    elif direction == "after":
        win_start = anchor_s + anchor.get("duration_s", 0)
        win_end = win_start + AFTER_WINDOW
    else:
        win_start = anchor_s - BEFORE_WINDOW // 2
        win_end = anchor_s + AFTER_WINDOW // 2

    window_events = get_events_in_window(anchor_ses, max(0, win_start), win_end)

    if not window_events:
        print("  ✗ No events found in the temporal window.")
        return

    timeline = "\n".join(
        [format_event(ev, i + 1, anchor["event_id"]) for i, ev in enumerate(window_events)]
    )

    print(
        f"\n  ── Timeline ({len(window_events)} events, "
        f"{direction} anchor) ────────────"
    )
    print(timeline)
    print("  " + "─" * 52)

    if not ollama_ready:
        return

    system = (
        "You are a memory assistant for egocentric kitchen video. "
        "Answer only from the chronological timeline. "
        "List relevant events in order with timestamps. "
        "Do not invent events not in the timeline."
    )

    print("\n  Reasoning over timeline...")

    answer = call_llm(
        f"Timeline:\n{timeline}\n\nQuestion: {question}\n\nAnswer:",
        system,
    )

    print("\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print("  ────────────────────────────────────────────────")


def handle_timerange(question, intent, ollama_ready):
    start_s = intent.get("start_s") or 0
    end_s = intent.get("end_s") or 600
    session = intent.get("session")

    print(
        f"  Time range: {start_s//60}:{start_s%60:02d} → "
        f"{end_s//60}:{end_s%60:02d}"
        + (f"  session: {session}" if session else "  all sessions")
    )

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

    timeline = "\n".join([format_event(ev, i + 1) for i, ev in enumerate(window_events)])

    print(f"\n  ── Timeline ({len(window_events)} events) ──────────────────")
    print(timeline)
    print("  " + "─" * 52)

    if not ollama_ready:
        return

    system = (
        "You are a memory assistant. Summarise what happened during the "
        "requested time period using all events in the timeline. "
        "List activities with timestamps. Never invent events."
    )

    answer = call_llm(
        f"Timeline:\n{timeline}\n\nQuestion: {question}\n\nAnswer:",
        system,
    )

    print("\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print("  ────────────────────────────────────────────────")


def handle_counting(question, intent, ollama_ready):
    session = intent.get("session")
    target_sessions = [session] if session else list(session_events.keys())

    q = question.lower()
    matching_events = []

    for ses in target_sessions:
        for ev in session_events.get(ses, []):
            caption = (
                ev.get("blip2_caption", "")
                or ev.get("activity_label", "")
            ).lower()

            if not caption:
                continue

            if "wash" in q and "hand" in q:
                positive = [
                    "washing his hands",
                    "washing their hands",
                    "wash hands",
                    "washing hands",
                    "washing her hands",
                ]
                negative = [
                    "washing dishes",
                    "dishwasher",
                    "cleaning the sink",
                    "sink faucet",
                    "washing a dish",
                ]

                if any(p in caption for p in positive) and not any(n in caption for n in negative):
                    matching_events.append(ev)

            else:
                cleaned = clean_query_for_lavila(question)
                words = [w for w in cleaned.split() if len(w) > 3]

                if all(w in caption for w in words[:2]):
                    matching_events.append(ev)

    print(f"\n  Found {len(matching_events)} matching events:")

    for ev in matching_events:
        caption = ev.get("blip2_caption", "") or ev.get("activity_label", "")

        print(
            f"    {ev['session']}  {ev['start_time']} → {ev['end_time']}  "
            f"[{caption}]"
        )

    if ollama_ready:
        print(f"\n  Answer: {len(matching_events)} occurrences.")


def handle_comparison(question, intent, ollama_ready):
    sessions_mentioned = re.findall(r"\bP\d+_\d+\b", question)

    if len(sessions_mentioned) < 2:
        print("  Please provide two sessions, e.g. Compare P01_09 and P30_107.")
        return

    timelines = {}

    for ses in sessions_mentioned[:2]:
        evs = session_events.get(ses, [])
        timelines[ses] = "\n".join(
            [format_event(ev, i + 1) for i, ev in enumerate(evs[:30])]
        )

    context = "\n\n".join([f"Session {ses}:\n{tl}" for ses, tl in timelines.items()])

    print(f"\n  Comparing {list(timelines.keys())}...")

    if not ollama_ready:
        print(context)
        return

    system = (
        "You are a memory assistant. Compare the activities across the "
        "provided sessions using only the captions and timestamps. "
        "Do not invent details."
    )

    answer = call_llm(
        f"{context}\n\nQuestion: {question}\n\nAnswer:",
        system,
        max_tokens=600,
    )

    print("\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print("  ────────────────────────────────────────────────")


def handle_summary(question, intent, ollama_ready):
    session = intent.get("session")

    if not session:
        m = re.search(r"\bP\d+_\d+\b", question)
        session = m.group(0) if m else None

    if not session:
        print("  Please specify a session, e.g. Summarise session P01_09.")
        return

    evs = session_events.get(session, [])

    if not evs:
        print(f"  No events found for session {session}.")
        return

    timeline = "\n".join([format_event(ev, i + 1) for i, ev in enumerate(evs)])

    duration = evs[-1].get("end_time", "?")

    print(f"\n  Summarising {session} ({len(evs)} events, {duration} total)...")

    if not ollama_ready:
        print(timeline)
        return

    system = (
        "You are a memory assistant. Provide a structured summary of the session "
        "using only the timeline. Group related activities. Include timestamps "
        "where useful. Do not invent details."
    )

    answer = call_llm(
        f"Full timeline for {session}:\n{timeline}\n\nQuestion: {question}\n\nAnswer:",
        system,
        max_tokens=600,
    )

    print("\n  ── Answer ──────────────────────────────────────")
    print(f"  {answer}")
    print("  ────────────────────────────────────────────────")


HANDLERS = {
    "factual": handle_factual,
    "anchor": handle_anchor,
    "timerange": handle_timerange,
    "counting": handle_counting,
    "comparison": handle_comparison,
    "summary": handle_summary,
}

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    print("\nChecking Ollama...")
    ollama_ready = check_ollama()

    print(
        f"  {'✓ Ollama ready  |  model: ' + OLLAMA_MODEL if ollama_ready else '⚠ Ollama not running — showing timelines only'}"
    )

    print("\n" + "=" * 62)
    print("  LIFELOG MEMORY QA  —  LaViLa raw retrieval + Event QA")
    print("=" * 62)
    print("  Factual:    'When did someone open the fridge?'")
    print("              'When did someone cut onions?'")
    print("              'When was a white plate used?'")
    print("  Anchor:     'What happened before opening the fridge?'")
    print("  Time range: 'What happened between minute 5 and 10 in P01_09?'")
    print("  Counting:   'How many times did he wash his hands in session P01_09?'")
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

            print(
                f"  Query type: {intent['type'].upper()}"
                + (f"  direction: {intent['direction']}" if intent.get("direction") else "")
                + (f"  session: {intent['session']}" if intent.get("session") else "")
            )

            HANDLERS[intent["type"]](question, intent, ollama_ready)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()