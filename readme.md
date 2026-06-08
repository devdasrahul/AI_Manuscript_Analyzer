<div align="center">

# 📚 AI Manuscript Analyzer

### A production-grade RAG platform for publishing intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.25-1C3C3C?style=for-the-badge)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-FF6B35?style=for-the-badge)](https://trychroma.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Gradio](https://img.shields.io/badge/Gradio-5.29-F9A825?style=for-the-badge)](https://gradio.app)
[![Colab](https://img.shields.io/badge/Google_Colab-FREE_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

**Upload a manuscript → Get summaries, genre labels, character insights, themes, marketing copy, and more — powered by LangChain RAG + ChromaDB vector search.**

[▶ Open in Colab](#quick-start) · [Architecture](#architecture) · [Features](#features) · [Results](#sample-output)

</div>

---

## 📋 Table of Contents

- [What This Project Does](#what-this-project-does)
- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [RAG Pipeline Deep Dive](#rag-pipeline-deep-dive)
- [Analysis Modules](#analysis-modules)
- [Evaluation Metrics](#evaluation-metrics)
- [Dependency Conflict Fix](#dependency-conflict-fix)
- [Configuration Reference](#configuration-reference)
- [Sample Output](#sample-output)
- [Tech Stack](#tech-stack)
- [Resume Bullets](#resume-bullets)

---

## What This Project Does

Traditional manuscript review requires 6–12 weeks of editorial work. This platform automates the analytical layer in minutes:

| Manual Process | This Platform |
|----------------|---------------|
| Genre classification (hours) | Instant RAG-based classification with evidence |
| Theme extraction (days) | 5 themes with textual citations |
| Character mapping (hours) | Full cast with arcs, traits, relationships |
| Back-cover blurb (days) | Marketing package with comps and pitch |
| Structural analysis (hours) | Plot architecture with issue flagging |
| Sensitivity review (days) | Content advisory with reader recommendations |

Works on manuscripts up to **1,000+ pages**. Runs **entirely free** on Google Colab's T4 GPU.

---

## Features

### 8 Specialized Analysis Modules

| Module | Output |
|--------|--------|
| 📝 Executive Summary | 3-paragraph publisher catalog entry |
| 🏷️ Genre Classification | Primary/sub-genres with confidence + shelf placement |
| 🎭 Theme Extraction | Top 5 themes with evidence and significance |
| 🧑 Character Analysis | Full cast — traits, arcs, roles, relationships |
| 📣 Marketing Package | Pitch, blurb, comps, hashtags, award eligibility |
| ✍️ Writing Style & Craft | Voice, pacing, dialogue, strengths, revision notes |
| 🏗️ Narrative Structure | Framework, inciting incident, climax, subplots |
| 🔍 Sensitivity Review | Content warnings, representation, age rating |

### Technical Features

- **RAG with MMR retrieval** — Maximal Marginal Relevance prevents redundant chunk retrieval
- **Dual LLM backend** — free Flan-T5-Large or OpenAI GPT-4o-mini (one config change)
- **PDF extraction with fallback** — pdfplumber for tables/columns, PyPDF2 as fallback
- **Quality evaluation module** — measures retrieval relevance and chunk diversity
- **Export system** — save any analysis to timestamped `.txt` reports
- **Semantic chunk explorer** — inspect exactly which chunks are retrieved per query
- **Conflict-safe install** — pins `requests` and `opentelemetry` to fix all Colab errors

---

## Quick Start

### Option A — Google Colab (recommended, free GPU)

1. Upload `AI_Manuscript_Analyzer_v2.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU** (free)
3. Run cells **1 through 7** in order
4. Open the public `gradio.live` link — the UI is live
5. Upload a PDF or paste text, then click any analysis button

### Option B — Local

```bash
git clone https://github.com/YOUR_USERNAME/ai-manuscript-analyzer
cd ai-manuscript-analyzer
pip install -r requirements.txt
jupyter notebook AI_Manuscript_Analyzer_v2.ipynb
```

### Option C — OpenAI backend (better output quality)

In **Cell 2**, paste your key:
```python
OPENAI_API_KEY: str = "sk-..."   # uses GPT-4o-mini (~$0.01/analysis)
```
No other changes needed. The rest of the pipeline is identical.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       AI Manuscript Analyzer                        │
│                     (Google Colab / Local)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
             ┌───────────────▼──────────────────┐
             │           Gradio Web UI           │
             │  Upload │ Analyze │ Q&A │ Diag.  │
             └───────────────┬──────────────────┘
                             │
             ┌───────────────▼──────────────────┐
             │       ManuscriptRAG Engine        │
             │   (LangChain RetrievalQA chain)  │
             └─────────┬───────────┬────────────┘
                       │           │
         ┌─────────────▼──┐   ┌────▼───────────────────┐
         │  Text Splitter  │   │  ChromaDB Vector Store  │
         │  chunk=800      │   │  HNSW cosine index      │
         │  overlap=120    │   │  persisted to disk      │
         │  recursive      │   │  collection metadata    │
         └─────────────────┘   └────────┬───────────────┘
                                        │
                          ┌─────────────▼──────────────────┐
                          │   Sentence-BERT Embeddings      │
                          │   all-MiniLM-L6-v2              │
                          │   384-dim, cosine similarity    │
                          │   batch size=32, normalized     │
                          └────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────┐
│                          LLM Backends                             │
│   FREE : google/flan-t5-large  (HuggingFacePipeline, ~750MB)    │
│   PAID : gpt-4o-mini           (ChatOpenAI, ~$0.01/call)        │
└───────────────────────────────────────────────────────────────────┘
```

---

## RAG Pipeline Deep Dive

Understanding this pipeline is key to explaining it in interviews.

### 1. Text Ingestion

```python
# pdfplumber first (handles tables/multi-column layouts)
with pdfplumber.open(path) as pdf:
    for page in pdf.pages:
        text += page.extract_text()

# PyPDF2 fallback for encrypted/complex PDFs
```

**Why pdfplumber over PyPDF2?** pdfplumber uses pdfminer under the hood and correctly handles multi-column layouts and tables that PyPDF2 often merges into garbled text.

### 2. Text Cleaning

```python
text = re.sub(r'[ \t]+',  ' ',   text)   # collapse whitespace
text = re.sub(r'\n{4,}',  '\n\n\n', text) # max 3 consecutive newlines
text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)  # strip control chars
```

Consistent whitespace normalization prevents chunking artifacts where a paragraph straddles two chunks at whitespace boundaries.

### 3. Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,                   # 15% overlap
    separators=['\n\n\n', '\n\n', '\n', '. ', '! ', '? ', ' ', '']
)
```

**Why recursive?** The splitter tries each separator in order, only using smaller separators if a chunk still exceeds `chunk_size`. This means it tries to split on triple-newlines (scene breaks) first, then paragraph breaks, then sentence ends — preserving the most semantic content per chunk.

**Why 120-token overlap?** Overlap ensures that a sentence split across two chunks doesn't lose context at the boundary. 15% of chunk size is a standard heuristic.

### 4. Embedding

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)
```

**Why all-MiniLM-L6-v2?**
- 22M parameters — 5× faster than `all-mpnet-base-v2`
- 384-dim output — small enough for fast cosine search
- MTEB score: 56.26 — near-BERT quality
- L2-normalized vectors → cosine similarity = dot product (faster)
- Batch size 32 optimizes GPU utilization for large manuscripts

### 5. Vector Store

```python
Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_metadata={"hnsw:space": "cosine"},  # explicit metric
)
```

**Why ChromaDB?**
- Runs locally — no API keys, no network latency
- HNSW (Hierarchical Navigable Small World) index: O(log n) search
- Built-in persistence to disk — survives Colab session restarts
- `collection_metadata={"hnsw:space": "cosine"}` explicitly sets the distance metric to cosine (vs default L2)

### 6. MMR Retrieval

```python
vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,           # return 6 chunks
        "fetch_k": 18,    # consider top 18 by cosine first
        "lambda_mult": 0.7,  # 70% relevance, 30% diversity
    }
)
```

**Why MMR over simple top-K?** With pure cosine similarity, the top-6 chunks for a query like "describe the protagonist" often return near-duplicate passages from the same scene. MMR iteratively selects chunks that are both relevant to the query AND maximally different from already-selected chunks. `lambda_mult=0.7` weights relevance at 70% and diversity at 30%.

### 7. Task-Specific Prompting

Each analysis module has its own prompt template:

```python
# Marketing module example
task_prompt = (
    "Manuscript context:\n{context}\n\n"
    "Create a full marketing package:\n"
    "ELEVATOR PITCH (25 words max): ...\n"
    "BACK COVER BLURB (100 words): ...\n"
    ...
)
```

**Why task-specific prompts?** A single generic prompt ("analyze this manuscript") produces unfocused output. Structured prompts with explicit output format requirements:
- Force the LLM to address specific aspects
- Produce consistently structured output
- Reduce hallucination by constraining the answer space
- Make outputs directly usable in publishing workflows

---

## Analysis Modules

### 📝 Executive Summary
Generates a 3-paragraph publisher catalog entry covering premise, narrative arc, and market differentiation. Designed to match the format used by acquisitions editors.

### 🏷️ Genre Classification
Returns primary genre, up to 3 sub-genres, suggested bookstore shelf placement, supporting textual evidence, and confidence level.

### 🎭 Theme Extraction
Identifies top 5 major themes with: manifestation in narrative, specific textual evidence, and significance to the work.

### 🧑 Character Analysis
Full cast breakdown with: name, role (Protagonist/Antagonist/Supporting), defining personality traits, character arc, narrative function, and key relationships.

### 📣 Marketing Package
Complete publisher sales package: elevator pitch (25 words), back-cover blurb (100 words), target audience demographics, 3 comp titles with "fans of X" framing, 5 key selling points, social media hooks and hashtags, and award eligibility.

### ✍️ Writing Style & Craft
Editorial assessment covering: narrative voice and POV, prose rhythm and complexity, dialogue quality, pacing, world-building, top 3 strengths, top 3 revision recommendations.

### 🏗️ Narrative Structure
Structural mapping: story framework identification (3-act, Hero's Journey, etc.), opening hook analysis, inciting incident, midpoint shift, climax, subplots, and structural issue flagging.

### 🔍 Sensitivity Review
Content advisory report: trigger warnings, representation audit, cultural accuracy flags, recommended sensitivity readers, and age appropriateness rating.

---

## Evaluation Metrics

After ingestion, run `evaluate_rag_pipeline()` to see:

```
══════════════════════════════════════════
  RAG PIPELINE EVALUATION
══════════════════════════════════════════
  Source        : my_manuscript.pdf
  Characters    : 487,234
  Chunks stored : 612
  Est. pages    : ~243
  Chars/chunk   : 796
  Ingestion     : 8.4s

  Avg relevance : 0.6821  (max 1.0)
  Score range   : 0.4102 – 0.8934
  MMR diversity : 0.7340  (higher = less redundant)
  Rating        : 🟢 Excellent
══════════════════════════════════════════
```

**Avg relevance > 0.6** means retrieved chunks are highly similar to the query — strong embedding alignment.

**MMR diversity > 0.7** means the retrieved chunks cover different parts of the manuscript — MMR is working correctly.

---

## Dependency Conflict Fix

Colab's pre-installed packages cause two known conflicts:

| Package | Colab requires | Our install installs | Fix |
|---------|---------------|----------------------|-----|
| `requests` | `==2.32.4` | `2.34.2` | Pin `requests==2.32.4` before other installs |
| `opentelemetry-sdk` | `<1.39.0` | `1.42.1` | Pin entire opentelemetry suite to `1.38.0` |

Cell 1 resolves both by pinning these versions **before** installing the rest of the stack:

```python
pip('requests==2.32.4')
pip(
    'opentelemetry-api==1.38.0',
    'opentelemetry-sdk==1.38.0',
    'opentelemetry-exporter-otlp-proto-http==1.38.0',
    'opentelemetry-proto==1.38.0',
    'opentelemetry-exporter-otlp-proto-common==1.38.0',
)
```

---

## Configuration Reference

All settings are in **Cell 2**:

| Variable | Default | Options |
|----------|---------|---------|
| `OPENAI_API_KEY` | `""` (free mode) | `"sk-..."` for GPT-4o-mini |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | `all-mpnet-base-v2`, `BAAI/bge-small-en-v1.5` |
| `FREE_LLM_MODEL` | `flan-t5-large` | `flan-t5-xl` (needs >8GB VRAM) |
| `CHUNK_SIZE` | `800` | 500–1200 (smaller = more precise retrieval) |
| `CHUNK_OVERLAP` | `120` | 10–20% of chunk size |
| `TOP_K_RETRIEVAL` | `6` | 4–10 (more = more context, slower) |

---

## Sample Output

**Input:** First 4 chapters of a literary fiction manuscript (~4,000 words)

**Genre Classification output:**
```
PRIMARY GENRE: Literary Fiction
SUB-GENRES: Magical Realism, Grief Narrative, Cartographic Mystery
SHELF PLACEMENT: Literary Fiction / General Fiction
EVIDENCE: The map-as-grief metaphor runs through Chapter 1;
  the 'blood-encoded maps' device in Chapter 2 signals magical realism;
  the institutional conspiracy thread places it in mystery territory.
CONFIDENCE: High — multiple genre markers present across all chapters
```

**Marketing Package — Elevator Pitch:**
```
A grieving cartographer follows her dead mother's impossible coordinates
to a village that shouldn't exist — and discovers what maps can't hold.
```

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Orchestration | LangChain | 0.3.25 | RAG chain, prompt management |
| Vector Store | ChromaDB | 0.5.23 | Embedding storage and retrieval |
| Embeddings | sentence-transformers | 3.4.1 | Text → vector conversion |
| Free LLM | HuggingFace Transformers | 4.51.3 | Flan-T5-Large inference |
| Paid LLM | OpenAI | 1.77.0 | GPT-4o-mini (optional) |
| PDF Parsing | pdfplumber + PyPDF2 | 0.11.4 / 3.0.1 | Text extraction from PDFs |
| UI | Gradio | 5.29.1 | Web interface |
| Compute | Google Colab T4 | Free | GPU inference |
| Language | Python | 3.10+ | Runtime |

---

## Resume Bullets

Copy these directly into your resume or LinkedIn:

```
AI Manuscript Analyzer | Python, LangChain, ChromaDB, HuggingFace, Gradio | 2026

• Built publishing-focused AI platform analyzing book drafts with summaries,
  genre classification, theme extraction, character insights, narrative structure
  analysis, and full marketing packages using LangChain RAG + ChromaDB

• Implemented semantic search across 1,000+ page manuscripts using
  sentence-transformers (all-MiniLM-L6-v2, 384-dim) and MMR retrieval for
  diversity-weighted chunk selection over ChromaDB HNSW vector index

• Engineered 8 task-specific prompt templates per analysis module, improving
  output structure and quality over single-prompt baseline

• Added RAG evaluation pipeline measuring retrieval relevance (avg cosine
  similarity) and MMR diversity scores for pipeline diagnostics

• Resolved Colab dependency conflicts (requests, opentelemetry) via pinned
  pre-install strategy; deployed Gradio UI with PDF ingestion, export, and
  semantic chunk inspector
```

---

## Project Structure

```
ai-manuscript-analyzer/
├── AI_Manuscript_Analyzer_v2.ipynb   # Main notebook (8 cells)
├── README.md                         # This file
└── requirements.txt                  # For local install
```

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built with 🤖 LangChain · 🗃️ ChromaDB · 🤗 HuggingFace · 🎨 Gradio

*Made to run free on Google Colab T4*

</div>
