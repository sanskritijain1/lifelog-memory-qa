# Lifelog Memory QA

A visual memory retrieval system for egocentric videos that allows users to search past activities using natural language.

Built using **LaViLa**, **FAISS**, **Llama 3**, and **Streamlit**, the system retrieves relevant moments from first-person video recordings and provides timestamped answers with visual evidence.

---

## Overview

Humans naturally remember experiences through events rather than individual images. This project explores how a machine can build and query a memory of daily activities from egocentric video.

Given a natural language query such as:

* *"When did someone cut the onion?"*
* *"What happened before opening the fridge?"*
* *"What happened between minute 5 and 10?"*
* *"Summarise session P01_09."*

the system retrieves relevant video moments and generates evidence-based answers grounded in retrieved memories.

---

## Features

### Visual Memory Retrieval

* Frame-level retrieval using LaViLa embeddings
* FAISS similarity search over 13,825 video frames
* Natural language search for actions, objects, and activities

### Event-Based Memory

* Automatic event segmentation
* Temporal grouping of related frames
* Timestamped memory representation

### Temporal Reasoning

Supports queries such as:

* Before an event
* After an event
* Around an event
* Time-range exploration
* Session summarisation

### Evidence-Based Answers

* Returns retrieved visual memories
* Displays timestamps
* Shows supporting frames
* LLM reasoning grounded on retrieved events

### Interactive Interface

* Streamlit web application
* Visual evidence cards
* Session exploration
* Natural language interaction

---

## Dataset

This project uses videos from the EPIC-KITCHENS dataset, a large-scale egocentric vision benchmark containing first-person recordings of everyday kitchen activities.

Dataset used:

* EPIC-KITCHENS
* Multiple participant sessions
* More than 13,000 extracted frames indexed for retrieval

---

## System Architecture

```text
Video Sessions
      │
      ▼
Frame Extraction
      │
      ▼
LaViLa Embeddings
      │
      ▼
FAISS Vector Index
      │
      ▼
Frame Retrieval
      │
      ▼
Event Aggregation
      │
      ▼
Temporal Reasoning
      │
      ▼
Llama 3
      │
      ▼
Answer + Visual Evidence
```

---

## Technology Stack

| Component           | Technology          |
| ------------------- | ------------------- |
| Video Understanding | LaViLa              |
| Embeddings          | LaViLa Dual Encoder |
| Vector Search       | FAISS               |
| Language Model      | Llama 3 (Ollama)    |
| Interface           | Streamlit           |
| Dataset             | EPIC-KITCHENS       |
| Language            | Python              |

---

## Example Queries

### Factual Retrieval

```text
When did someone cut the onion?
When did someone hold a white plate?
When did someone open the fridge?
```

### Temporal Queries

```text
What happened before opening the fridge?
What was the person doing after washing hands?
```

### Timeline Exploration

```text
What happened between minute 5 and 10 in P01_09?
What was happening around minute 3?
```

### Session Understanding

```text
Summarise session P01_09.
Compare P01_09 and P30_107.
```

---

## Example Output

Question:

```text
When did someone cut the onion?
```

Retrieved Memories:

```text
P01_09   00:17:01 → 00:18:28
P01_09   00:19:43 → 00:21:19
P01_109  00:18:36 → 00:18:40
```

Answer:

```text
The strongest visual matches indicate onion cutting activities
between 00:17:01–00:18:28 and 00:19:43–00:21:19 in session P01_09.
```

---

## Project Structure

```text
scripts/
│
├── extract_frames.py
├── build_embeddings_lavila.py
├── build_event_embeddings.py
├── segment_events.py
├── label_events.py
├── search_memory_lavila.py
├── search_events_lavila.py
├── memory_qa.py
├── app.py
│
data/
│
├── frame_embeddings.npy
├── frame_paths.txt
├── events.json
├── session_timestamps.json
│
pretrained/
│
└── lavila_tsf_base_ep5.pth
```

---

## Future Improvements

* Stronger multimodal captioning models
* Hybrid frame + event retrieval
* Cross-session memory linking
* Multi-hop temporal reasoning
* Memory graph representation
* Larger egocentric datasets

---

## Motivation

This project was built to explore how AI systems can construct and query memories from continuous visual experiences. It combines computer vision, retrieval systems, temporal reasoning, and language models into a unified memory assistant for egocentric video understanding.
