import io
import re
import random
import contextlib
from pathlib import Path

import streamlit as st
from PIL import Image

import memory_qa as mq

st.set_page_config(
    page_title="Lifelog Memory QA",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef4f2 45%, #f7f3ee 100%);
    font-family: 'Inter', sans-serif;
    color: #1f2937;
}

.block-container {
    padding-top: 2.2rem;
    max-width: 1220px;
}

.hero {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(226,232,240,0.9);
    border-radius: 28px;
    padding: 2.2rem;
    box-shadow: 0 20px 55px rgba(31,41,55,0.08);
    backdrop-filter: blur(18px);
    margin-bottom: 1.6rem;
}

.hero-title {
    font-size: 3rem;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -0.055em;
    color: #172033;
    margin-bottom: 0.7rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #667085;
    max-width: 740px;
    line-height: 1.65;
}

.tech-pill {
    display: inline-block;
    padding: 0.38rem 0.72rem;
    background: #eef6f4;
    color: #245b53;
    border: 1px solid #d7ebe7;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-top: 0.8rem;
}

.stat-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 1.15rem 1rem;
    box-shadow: 0 12px 30px rgba(31,41,55,0.055);
    transition: all 0.22s ease;
}

.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 38px rgba(31,41,55,0.095);
}

.stat-number {
    font-size: 1.65rem;
    font-weight: 800;
    color: #1f2937;
}

.stat-label {
    color: #7a8494;
    font-size: 0.82rem;
    font-weight: 500;
}

.search-panel {
    background: rgba(255,255,255,0.86);
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    padding: 1.45rem;
    margin-top: 1.4rem;
    box-shadow: 0 14px 40px rgba(31,41,55,0.065);
}

.example-chip {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #dfe7ef;
    color: #465366;
    border-radius: 999px;
    padding: 0.5rem 0.82rem;
    margin: 0.25rem 0.2rem;
    font-size: 0.86rem;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(31,41,55,0.035);
    transition: all 0.2s ease;
}

.example-chip:hover {
    transform: translateY(-2px) scale(1.01);
    border-color: #a7c8c1;
    color: #245b53;
    box-shadow: 0 10px 22px rgba(31,41,55,0.08);
}

.answer-card {
    background: rgba(255,255,255,0.9);
    border: 1px solid #e5e7eb;
    border-left: 5px solid #7aa69e;
    border-radius: 24px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 16px 42px rgba(31,41,55,0.075);
    margin-top: 1.4rem;
}

.answer-title {
    font-size: 1.25rem;
    font-weight: 760;
    color: #1f2937;
    margin-bottom: 0.75rem;
}

.answer-text {
    white-space: pre-wrap;
    line-height: 1.72;
    color: #3f4754;
    font-size: 0.98rem;
}

.intent-badge {
    display: inline-block;
    padding: 0.34rem 0.65rem;
    border-radius: 999px;
    background: #f0f7f5;
    color: #2f615b;
    border: 1px solid #d5ebe6;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.7rem;
}

.event-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 14px 36px rgba(31,41,55,0.075);
    margin-bottom: 1.3rem;
    transition: all 0.24s ease;
}

.event-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 22px 48px rgba(31,41,55,0.12);
    border-color: #bdd8d2;
}

.event-inner {
    padding: 1.05rem 1.15rem 1.2rem;
}

.event-rank {
    color: #2f615b;
    background: #eff8f6;
    border: 1px solid #d6ebe6;
    border-radius: 999px;
    padding: 0.26rem 0.55rem;
    font-size: 0.74rem;
    font-weight: 750;
}

.event-title {
    font-size: 1.02rem;
    font-weight: 760;
    color: #1f2937;
    margin-top: 0.75rem;
}

.meta {
    color: #6b7280;
    font-size: 0.86rem;
    margin-top: 0.2rem;
}

.auto-caption {
    color: #485466;
    font-size: 0.9rem;
    line-height: 1.55;
    margin-top: 0.7rem;
}

.score {
    margin-top: 0.65rem;
    font-size: 0.8rem;
    color: #7a8494;
}

div[data-testid="stTextInput"] input {
    border-radius: 18px;
    border: 1px solid #d6dee8;
    padding: 0.95rem 1rem;
    font-size: 1rem;
    background: #ffffff;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #7aa69e;
    box-shadow: 0 0 0 4px rgba(122,166,158,0.14);
}

.stButton > button {
    border-radius: 16px;
    padding: 0.72rem 1.25rem;
    font-weight: 700;
    border: none;
    background: linear-gradient(135deg, #52796f, #6fa89d);
    color: white;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(82,121,111,0.22);
    background: linear-gradient(135deg, #456b63, #619a90);
}

img {
    border-radius: 18px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

img:hover {
    transform: scale(1.015);
    box-shadow: 0 18px 38px rgba(31,41,55,0.16);
}

.section-title {
    font-size: 1.35rem;
    font-weight: 780;
    color: #1f2937;
    margin: 1.8rem 0 0.9rem;
}

.footer-note {
    color: #8a94a3;
    font-size: 0.82rem;
    margin-top: 2rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def clean_answer(raw_output: str) -> str:
    if "── Answer" in raw_output:
        raw_output = raw_output.split("── Answer")[-1]

    raw_output = re.sub(r"─+", "", raw_output).strip()

    lines = []
    keep = False

    for line in raw_output.splitlines():
        stripped = line.strip()

        if not stripped:
            if keep:
                lines.append("")
            continue

        if (
            stripped.startswith("Based on")
            or stripped.startswith("According to")
            or stripped.startswith("Here")
            or stripped.startswith("The")
            or stripped.startswith("*")
            or re.match(r"^\d+\.", stripped)
        ):
            keep = True

        if keep:
            lines.append(line)

    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else raw_output


def run_query(question):
    intent = mq.classify_intent(question)
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        mq.HANDLERS[intent["type"]](question, intent, mq.check_ollama())

    return intent, buffer.getvalue()


def get_events_for_display(question, intent):
    if intent["type"] in ["factual", "anchor"]:
        query = intent.get("anchor_phrase", question) if intent["type"] == "anchor" else question
        results, _ = mq.retrieve_events_frame_first(
            query,
            session_filter=intent.get("session")
        )
        return results[:6]

    if intent["type"] == "timerange":
        session = intent.get("session")
        if not session:
            return []
        return mq.get_events_in_window(
            session,
            intent.get("start_s", 0),
            intent.get("end_s", 600)
        )

    if intent["type"] == "summary":
        session = intent.get("session")
        if session:
            return mq.session_events.get(session, [])[:12]

    return []


def get_caption(event):
    return (
        event.get("caption")
        or event.get("blip2_caption", "")
        or event.get("activity_label", "")
        or "No auto-caption available"
    )


def get_frame(event):
    return (
        event.get("best_frame")
        or event.get("center_frame")
        or event.get("center_path")
    )


def random_hero_frames(n=3):
    candidates = []
    for p in mq.paths:
        if Path(p).exists():
            candidates.append(p)

    random.seed(7)
    return random.sample(candidates, min(n, len(candidates)))


# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">Lifelog Memory QA</div>
    <div class="hero-subtitle">
        A visual memory assistant for egocentric videos. Ask natural-language questions,
        retrieve relevant moments, inspect visual evidence, and reason over events and timestamps.
    </div>
    <span class="tech-pill">LaViLa</span>
    <span class="tech-pill">FAISS</span>
    <span class="tech-pill">Frame-first Retrieval</span>
    <span class="tech-pill">Event Grounding</span>
    <span class="tech-pill">Llama Reasoning</span>
</div>
""", unsafe_allow_html=True)

# Hero images
hero_imgs = random_hero_frames(3)
if hero_imgs:
    hcols = st.columns(3)
    for col, img_path in zip(hcols, hero_imgs):
        with col:
            st.image(Image.open(img_path), use_container_width=True)

# Stats
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{len(mq.paths):,}</div><div class="stat-label">Indexed video frames</div></div>',
        unsafe_allow_html=True
    )
with s2:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{len(mq.events):,}</div><div class="stat-label">Temporal memory events</div></div>',
        unsafe_allow_html=True
    )
with s3:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{len(mq.session_events)}</div><div class="stat-label">Egocentric sessions</div></div>',
        unsafe_allow_html=True
    )

# -----------------------------
# Search
# -----------------------------
st.markdown('<div class="search-panel">', unsafe_allow_html=True)

st.markdown("### Ask a memory question")
query = st.text_input(
    label="",
    placeholder="Try: When did someone cut the onion?",
    label_visibility="collapsed"
)

st.markdown("""
<span class="example-chip">When did someone open the fridge?</span>
<span class="example-chip">When did someone cut the onion?</span>
<span class="example-chip">When did someone hold a white plate?</span>
<span class="example-chip">What happened between minute 5 and 10 in P01_09?</span>
<span class="example-chip">Summarise session P01_09</span>
""", unsafe_allow_html=True)

search = st.button("Search Memory", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Results
# -----------------------------
if search:
    if not query.strip():
        st.warning("Please enter a memory question.")
        st.stop()

    question = query.strip()

    with st.spinner("Retrieving visual memories and reasoning over events..."):
        intent, raw_output = run_query(question)
        events = get_events_for_display(question, intent)

    answer = clean_answer(raw_output)

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="intent-badge">{intent["type"].upper()}</div>
            <div class="answer-title">Answer</div>
            <div class="answer-text">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if events:
        st.markdown('<div class="section-title">Retrieved visual evidence</div>', unsafe_allow_html=True)

        cols = st.columns(3)

        for i, ev in enumerate(events, start=1):
            with cols[(i - 1) % 3]:
                session = ev.get("session", "unknown")
                start = ev.get("start_time", "?")
                end = ev.get("end_time", "?")
                duration = ev.get("duration_s", 0)
                score = ev.get("score", None)
                caption = get_caption(ev)
                frame_path = get_frame(ev)

                st.markdown('<div class="event-card">', unsafe_allow_html=True)

                if frame_path and Path(frame_path).exists():
                    st.image(Image.open(frame_path), use_container_width=True)

                st.markdown('<div class="event-inner">', unsafe_allow_html=True)
                st.markdown(f'<span class="event-rank">Evidence {i}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="event-title">{session}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="meta">{start} → {end} · {duration:.0f}s</div>',
                    unsafe_allow_html=True
                )

                if score is not None:
                    st.markdown(
                        f'<div class="score">Visual retrieval score: {score:.4f}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f'<div class="auto-caption"><b>Auto-caption:</b> {caption}</div>',
                    unsafe_allow_html=True
                )

                st.markdown('</div></div>', unsafe_allow_html=True)

    else:
        st.info("No related events found.")

st.markdown(
    '<div class="footer-note">Frame-level LaViLa retrieval is used as primary evidence. Auto-captions are auxiliary and may be noisy.</div>',
    unsafe_allow_html=True
)