# 🧠 AI Memory Compression Engine

> **Reduce LLM token usage by 60-80%** through multi-strategy memory compression — summarization, importance scoring, semantic deduplication, and hierarchical memory tiers.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper%20%2B%20GPT-orange)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

A very under-explored area — most AI systems naively append all conversation history until the context window fills up. This engine solves that.

---

## 🎯 Problem

LLMs have fixed context windows. Naive memory = just append everything:

```
turn 1 (200 tokens) + turn 2 (150 tokens) + ... + turn 50 (300 tokens) = 💀 context overflow
```

This engine manages memory intelligently so your LLM always has the **most relevant context** within budget.

---

## ✨ Features

### 🔄 5 Compression Strategies

| Strategy | Description | Token Savings |
|----------|-------------|--------------|
| **Hybrid** | All strategies combined (recommended) | 60-80% |
| **Summarize** | GPT-4o-mini compresses conversation blocks | 50-70% |
| **Deduplicate** | Removes semantically redundant entries (Jaccard similarity) | 10-30% |
| **Score & Prune** | Drops lowest-importance memories | Variable |
| **Hierarchical** | Promotes/demotes entries across memory tiers | Lossless |

### 🏗️ 4 Memory Tiers

```
┌─────────────────────────────────────────────────────────────┐
│  WORKING MEMORY    ⚡  Recent turns, full fidelity (~10 msgs)│
├─────────────────────────────────────────────────────────────┤
│  EPISODIC MEMORY   📚  Compressed session summaries         │
├─────────────────────────────────────────────────────────────┤
│  SEMANTIC MEMORY   💡  Distilled facts & long-term knowledge │
├─────────────────────────────────────────────────────────────┤
│  ARCHIVAL MEMORY   🗄️  Deep history, rarely retrieved       │
└─────────────────────────────────────────────────────────────┘
```

### 📊 Importance Scoring
Multi-signal importance scoring for every memory entry:

- **Recency decay** — exponential decay over time
- **Access frequency** — log-scaled boost for frequently retrieved memories
- **Keyword boosting** — detects critical terms (decision, deadline, goal, etc.)
- **Role weighting** — system > user > assistant
- **Length factor** — penalizes very short, low-value entries

### 🖥️ Live Dashboard
- 3-panel layout: Chat | Memory Visualizer | Controls
- Real-time token budget gauge with color-coded alerts
- Memory timeline with tier filtering and importance bars
- Compression history chart (before/after comparison)
- Instant memory search across all tiers

---

## 🏗️ Architecture

```
ai-memory-compression-engine/
├── engine/
│   └── memory_engine.py     # Core engine (700+ lines)
│       ├── MemoryEntry       # Dataclass for memory entries
│       ├── ImportanceScorer  # Multi-signal importance scoring
│       ├── SemanticDeduplicator  # Jaccard-based deduplication
│       ├── LLMSummarizer     # GPT-powered compression
│       └── MemoryCompressionEngine  # Main orchestrator
├── app.py                   # Flask API (15 endpoints)
├── requirements.txt
├── .env.example
├── templates/
│   └── index.html           # 3-panel dashboard UI
└── static/
    ├── css/style.css        # Dark terminal-aesthetic theme
    └── js/app.js            # Frontend with real-time viz
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/PranayMahendrakar/ai-memory-compression-engine.git
cd ai-memory-compression-engine

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

python app.py
# Open http://localhost:5000
```

---

## 📡 API Reference

### Session Management
```
POST /api/session              Create new session
GET  /api/sessions             List all sessions  
DELETE /api/session/<id>       Delete session
```

### Memory Operations
```
GET  /api/memory/<id>          Get all memories + stats
POST /api/memory/<id>/add      Add single message
POST /api/memory/<id>/batch    Add multiple messages
GET  /api/memory/<id>/search   Search memories
GET  /api/memory/<id>/context  Get context window for LLM
DELETE /api/memory/<id>/clear/<tier>  Clear tier
```

### Compression
```
POST /api/compress/<id>        Trigger compression
GET  /api/compress/<id>/history  Compression history
```

### Chat & Visualization
```
POST /api/chat/<id>            Memory-aware chat
GET  /api/visualize/<id>       Visualization data
GET  /api/health               Health check
```

---

## 🧩 Use As Python Library

```python
from engine.memory_engine import MemoryCompressionEngine, CompressionStrategy

# Initialize
engine = MemoryCompressionEngine(
    api_key="sk-...",
    token_budget=4000,
    working_window=10,
    auto_compress=True   # Auto-compresses when budget exceeded
)

# Add messages
engine.add_message("user", "My name is Alice and I'm building a chatbot")
engine.add_message("assistant", "Great! What kind of chatbot are you building?")
engine.add_message("user", "A customer support bot for an e-commerce platform")

# Get compressed context for LLM call
context = engine.get_context_window(max_tokens=3000)

# Manual compression
stats = engine.compress(strategy=CompressionStrategy.HYBRID)
print(f"Saved {stats.tokens_saved} tokens ({stats.savings_pct}%)")

# Search memory
results = engine.search_memory("customer support", top_k=5)

# Full stats
print(engine.get_stats())
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `LLM_MODEL` | gpt-4o-mini | Model for summarization |
| `TOKEN_BUDGET` | 4000 | Max tokens in context |
| `WORKING_WINDOW` | 10 | Recent messages to keep verbatim |
| `FLASK_DEBUG` | 1 | Debug mode |
| `PORT` | 5000 | Server port |

---

## 🔬 How Compression Works

```
New Message Added
      ↓
Auto-trigger check: total_tokens > budget?
      ↓ YES
1. Deduplicate: Remove Jaccard-similar entries
      ↓
2. Hierarchical: Demote old episodic → archival
      ↓  
3. Summarize: GPT compresses episodic blocks
      ↓  "[MEMORY SUMMARY] Alice is building a customer support 
           chatbot for e-commerce. Discussed architecture choices..." 
      ↓
4. Score & Prune: Drop lowest-importance remaining
      ↓
Budget satisfied ✅
```

---

## 🎯 Use Cases

- **AI Assistants** — Maintain long-running conversations without context overflow
- **Autonomous Agents** — Keep task history across many tool-call loops
- **Chatbots** — Remember user preferences across sessions
- **RAG Systems** — Compress retrieved chunks before injection
- **Multi-agent Systems** — Share compressed memory between agents

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
