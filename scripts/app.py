import os, sys, json, gc, re, threading, http.server, socketserver
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"

LAVILA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LaViLa"
)
if LAVILA_ROOT not in sys.path:
    sys.path.insert(0, LAVILA_ROOT)

import streamlit as st
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPTokenizer

torch.set_num_threads(1)
faiss.omp_set_num_threads(1)

st.set_page_config(page_title="Lifelog Memory Search", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0a0a0f;--surface:#12121a;--surface2:#1c1c28;--border:#2a2a3d;
      --accent:#7c6af7;--accent2:#4ecdc4;--text:#e8e8f0;--text-dim:#6b6b8a;}
html,body,.stApp{background-color:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 3rem;max-width:1400px;}
.hero{text-align:center;padding:3rem 0 2rem;border-bottom:1px solid var(--border);margin-bottom:2.5rem;}
.hero-title{font-family:'Space Mono',monospace;font-size:2.8rem;font-weight:700;
            background:linear-gradient(135deg,var(--accent),var(--accent2));
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;}
.hero-sub{color:var(--text-dim);font-family:'Space Mono',monospace;font-size:0.75rem;
          text-transform:uppercase;letter-spacing:0.05em;margin-top:0.5rem;}
.stTextInput input{background:var(--surface)!important;border:1px solid var(--border)!important;
                   border-radius:12px!important;color:var(--text)!important;
                   font-size:1.1rem!important;padding:1rem 1.5rem!important;}
.stTextInput input:focus{border-color:var(--accent)!important;
                         box-shadow:0 0 0 3px rgba(124,106,247,0.15)!important;}
.stButton button{background:linear-gradient(135deg,var(--accent),#5b4fe0)!important;
                 color:white!important;border:none!important;border-radius:10px!important;
                 font-weight:600!important;font-size:1rem!important;
                 padding:0.7rem 2rem!important;width:100%!important;}
.result-card{background:var(--surface);border:1px solid var(--border);
             border-radius:16px;padding:1.5rem;margin-bottom:0.5rem;}
.result-card:hover{border-color:var(--accent);}
.rank-badge{font-family:'Space Mono',monospace;font-size:0.7rem;color:var(--accent);
            border:1px solid var(--accent);border-radius:6px;padding:0.1rem 0.5rem;
            margin-bottom:0.75rem;display:inline-block;}
.session-label{font-family:'Space Mono',monospace;font-size:1rem;font-weight:700;margin-bottom:0.3rem;}
.timestamp-label{font-family:'Space Mono',monospace;font-size:0.9rem;color:var(--accent2);margin-bottom:0.2rem;}
.duration-label{font-size:0.85rem;color:var(--text-dim);margin-bottom:0.75rem;}
.score-bar-bg{background:var(--surface2);border-radius:4px;height:4px;width:100%;margin-top:0.5rem;}
.score-bar-fill{height:4px;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2));}
.score-text{font-family:'Space Mono',monospace;font-size:0.7rem;color:var(--text-dim);margin-top:0.3rem;}
.expansion-pill{display:inline-block;background:var(--surface2);border:1px solid var(--border);
                border-radius:20px;padding:0.2rem 0.75rem;font-size:0.8rem;
                color:var(--text-dim);margin:0.2rem;font-family:'Space Mono',monospace;}
.no-results{text-align:center;padding:3rem;color:var(--text-dim);font-family:'Space Mono',monospace;}
</style>
""", unsafe_allow_html=True)

# ── VIDEO FILE SERVER ─────────────────────────────────────────────────────────
VIDEO_SERVER_PORT = 8502

def start_video_server():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.path.abspath("."), **kwargs)
        def log_message(self, *args): pass  # suppress logs
    try:
        with socketserver.TCPServer(("", VIDEO_SERVER_PORT), Handler) as httpd:
            httpd.serve_forever()
    except OSError:
        pass  # already running

if "video_server_started" not in st.session_state:
    threading.Thread(target=start_video_server, daemon=True).start()
    st.session_state.video_server_started = True

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "results"     not in st.session_state: st.session_state.results     = []
if "narrations"  not in st.session_state: st.session_state.narrations  = []
if "last_query"  not in st.session_state: st.session_state.last_query  = ""
if "show_videos" not in st.session_state: st.session_state.show_videos = set()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_video_url(session: str):
    participant = session.split("_")[0]
    rel = f"epic_data/EPIC-KITCHENS/{participant}/videos/{session}.MP4"
    return f"http://localhost:{VIDEO_SERVER_PORT}/{rel}" if Path(rel).exists() else None

# ── LOAD SYSTEM ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading memory system...")
def load_system():
    from lavila.models import models as lavila_models
    DEVICE = "cpu"; NUM_FRAMES = 4

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    ckpt = torch.load("pretrained/lavila_tsf_base_ep5.pth", map_location="cpu")
    sd   = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model = lavila_models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=NUM_FRAMES)
    model.load_state_dict(sd, strict=False)
    model.eval()
    del ckpt; gc.collect()

    emb = np.load("data/frame_embeddings.npy").astype("float32")
    with open("data/frame_paths.txt") as f:
        paths = [l.strip() for l in f if l.strip()]
    mean_vec = np.load("data/frame_mean.npy").astype("float32")
    emb = emb - mean_vec
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    with open("data/events.json") as f: events = json.load(f)
    f2e = {}
    for ev in events:
        for fp in ev["frame_paths"]:
            f2e[(ev["session"], int(Path(fp).stem.split("_")[1]))] = ev

    with open("data/session_timestamps.json") as f: ts = json.load(f)
    return dict(model=model, tokenizer=tokenizer, index=idx, paths=paths,
                mean_vec=mean_vec, events=events, f2e=f2e, ts=ts, device=DEVICE)

# ── QUERY EXPANSION ───────────────────────────────────────────────────────────
RULES = [
    (["fridge","refrigerator"],
     ["open the fridge","take food out of the fridge","close the fridge door"]),
    (["bin","trash","rubbish","throw"],
     ["throw rubbish in the bin","open the bin lid","put trash in the bin"]),
    (["wash","sink","hands","tap"],
     ["wash hands in the sink","turn on the tap","rinse hands under water"]),
    (["cut","chop","knife","board","slice"],
     ["cut vegetables on the board","chop food with a knife",
      "slice ingredients on the cutting board"]),
    (["cook","stove","hob","pan","pot","stir","boil"],
     ["stir food in the pan","cook on the stove","boil water in the pot"]),
    (["plate","dish","serve"],
     ["put food on a plate","use a white plate","place food on the dish"]),
    (["peel","vegetable","onion","tomato"],
     ["peel a vegetable","prepare vegetables","chop an onion on the board"]),
    (["eat","meal"], ["eat food","pick up food to eat","have a meal"]),
    (["pour","water","liquid","bottle"],
     ["pour water into the pot","pour liquid from a bottle","fill the pot with water"]),
    (["egg","crack"], ["crack an egg into a bowl","break an egg","whisk eggs in a bowl"]),
]

def expand(q):
    ql = q.lower()
    for kws, exps in RULES:
        if any(k in ql for k in kws): return exps
    return [q]

def parse_filter(q):
    s = None; t = None
    m = re.search(r'\bP\d+_\d+\b', q)
    if m: s = m.group(0)
    m = re.search(
        r'(?:around\s+|at\s+)?(\d+)\s*(?:minutes?|mins?)\b|(?:minute|min)\s+(\d+)\b',
        q, re.IGNORECASE)
    if m: t = int(m.group(1) or m.group(2)) * 60
    return s, t

# ── SEARCH ────────────────────────────────────────────────────────────────────
def search(query, pool, threshold, top_k):
    sys = load_system()
    nars = expand(query)
    toks = sys["tokenizer"](nars, return_tensors="pt",
                            padding="max_length", truncation=True, max_length=77)
    with torch.no_grad():
        f = sys["model"].encode_text(toks["input_ids"])
        f = f / f.norm(dim=-1, keepdim=True)
    vec = f.mean(dim=0, keepdim=True).cpu().numpy().astype("float32")
    vec = vec - sys["mean_vec"]
    faiss.normalize_L2(vec)

    D, I = sys["index"].search(vec, pool)
    fs, ft = parse_filter(query)
    seen = {}
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        fp  = sys["paths"][idx]
        p   = Path(fp)
        ses = p.parent.name
        fn  = int(p.stem.split("_")[1])
        if fs and ses != fs: continue
        if ft is not None:
            spf = sys["ts"].get(ses, {}).get("spf", 1.0)
            if abs(fn * spf - ft) > 180: continue
        ev = sys["f2e"].get((ses, fn))
        if not ev: continue
        eid = ev["event_id"]
        if eid not in seen or score > seen[eid]["score"]:
            seen[eid] = dict(score=float(score), event_id=eid,
                             session=ev["session"],
                             start_time=ev["start_time"], end_time=ev["end_time"],
                             start_s=ev.get("start_s", 0),
                             duration_s=ev["duration_s"], frame_count=ev["frame_count"],
                             best_frame=fp, center_frame=ev["center_path"])
    results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] >= threshold]
    for i, r in enumerate(results): r["rank"] = i + 1
    return results[:top_k], nars

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 class="hero-title">🧠 LIFELOG MEMORY</h1>
    <p class="hero-sub">Egocentric Video · Semantic Search · LaViLa + FAISS</p>
</div>""", unsafe_allow_html=True)

system = load_system()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Score threshold", 0.30, 0.70, 0.45, 0.05)
    pool_size = st.slider("Frame pool", 20, 100, 50, 10)
    top_k     = st.slider("Max results", 1, 10, 5)
    st.markdown("---")
    st.markdown(f"**Frames:** {len(system['paths']):,}")
    st.markdown(f"**Events:** {len(system['events']):,}")
    st.markdown("---")
    st.markdown("**Tips:** `open the fridge` · `wash hands` · `cut vegetables`")

col_q, col_b = st.columns([5, 1])
with col_q:
    query = st.text_input("", placeholder="e.g. 'open the fridge'  ·  'wash hands'",
                          label_visibility="collapsed", key="q",
                          value=st.session_state.last_query)
with col_b:
    go = st.button("Search", use_container_width=True)

if go and query.strip():
    with st.spinner("Searching memory..."):
        res, nars = search(query.strip(), pool_size, threshold, top_k)
    st.session_state.results    = res
    st.session_state.narrations = nars
    st.session_state.last_query = query.strip()
    st.session_state.show_videos = set()

if st.session_state.results:
    results = st.session_state.results
    nars    = st.session_state.narrations

    pills = " ".join([f'<span class="expansion-pill">"{n}"</span>' for n in nars])
    st.markdown(f"<div style='margin-bottom:1rem;color:#6b6b8a;font-size:0.85rem'>"
                f"Searched as: {pills}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#6b6b8a;font-family:Space Mono,monospace;"
                f"font-size:0.8rem;margin-bottom:1.5rem'>"
                f"Found {len(results)} events</div>", unsafe_allow_html=True)

    for r in results:
        pct       = min(100, int(r["score"] * 120))
        vid_url   = get_video_url(r["session"])
        start_sec = int(r.get("start_s") or 0)
        eid       = r["event_id"]

        st.markdown(f"""
<div class="result-card">
    <span class="rank-badge">MATCH #{r['rank']}</span>
    <div class="session-label">{r['session']}</div>
    <div class="timestamp-label">⏱ {r['start_time']} → {r['end_time']}</div>
    <div class="duration-label">{r['duration_s']:.0f}s · {r['frame_count']} frames</div>
    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>
    <div class="score-text">relevance {r['score']:.3f}</div>
</div>""", unsafe_allow_html=True)

        img_col, vid_col = st.columns([1, 2])
        with img_col:
            try:
                st.image(Image.open(r["best_frame"]),
                         caption="Best matching frame", use_container_width=True)
            except Exception:
                st.warning("Frame not found")

        with vid_col:
            if vid_url:
                if st.button(f"▶ Play from {r['start_time']}", key=f"p_{eid}"):
                    if eid in st.session_state.show_videos:
                        st.session_state.show_videos.discard(eid)
                    else:
                        st.session_state.show_videos.add(eid)

                if eid in st.session_state.show_videos:
                    st.markdown(
                        f"<p style='color:#6b6b8a;font-family:Space Mono,monospace;"
                        f"font-size:0.75rem;margin-bottom:0.3rem'>"
                        f"📹 {r['session']} · {r['start_time']}</p>",
                        unsafe_allow_html=True
                    )
                    # autoplay + muted satisfies Chrome autoplay policy
                    # #t=start_sec in URL hints the browser to seek immediately
                    # onloadedmetadata sets currentTime precisely after metadata loads
                    st.markdown(
                        f'<video width="100%" controls autoplay muted preload="auto" '
                        f'style="border-radius:8px;background:#000;max-height:380px" '
                        f'onloadedmetadata="this.currentTime={start_sec}">'
                        f'<source src="{vid_url}#t={start_sec}" type="video/mp4">'
                        f'</video>',
                        unsafe_allow_html=True
                    )
            else:
                st.info(f"Video not found for {r['session']}")

        st.markdown("<hr style='border-color:#1c1c28;margin:1rem 0'>",
                    unsafe_allow_html=True)

elif go and not query.strip():
    st.warning("Please enter a search query.")