"""
AI Memory Compression Engine - Flask API
=========================================
REST API for the memory compression engine with:
  - Session management (multiple independent memory sessions)
  - Chat endpoint with memory-aware context injection  
  - Manual/auto compression triggers
  - Real-time stats and visualization data
  - Memory search and retrieval
"""

import os
import sys
import json
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# Add engine to path
sys.path.insert(0, os.path.dirname(__file__))
from engine.memory_engine import (
    MemoryCompressionEngine,
    CompressionStrategy,
    MemoryTier,
    count_tokens
)

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Session Store ──────────────────────────────
# In production: use Redis/DB. Here: in-memory dict per session_id.
sessions: dict[str, MemoryCompressionEngine] = {}


def get_engine(session_id: str) -> MemoryCompressionEngine:
    """Get or create a memory engine for a session."""
    if session_id not in sessions:
        sessions[session_id] = MemoryCompressionEngine(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            token_budget=int(os.environ.get("TOKEN_BUDGET", "4000")),
            working_window=int(os.environ.get("WORKING_WINDOW", "10")),
            auto_compress=True,
        )
    return sessions[session_id]


# ── Routes ─────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    api_key = os.environ.get("OPENAI_API_KEY")
    return jsonify({
        "status": "healthy",
        "api_key_configured": bool(api_key and not api_key.startswith("your-")),
        "active_sessions": len(sessions),
        "version": "1.0.0"
    })


# ── Session Endpoints ──────────────────────────

@app.route("/api/session", methods=["POST"])
def create_session():
    """Create a new memory session with custom config."""
    data = request.json or {}
    session_id = data.get("session_id") or f"session_{int(time.time())}"

    engine = MemoryCompressionEngine(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=data.get("model", os.environ.get("LLM_MODEL", "gpt-4o-mini")),
        token_budget=data.get("token_budget", 4000),
        working_window=data.get("working_window", 10),
        importance_threshold=data.get("importance_threshold", 0.2),
        dedup_threshold=data.get("dedup_threshold", 0.85),
        decay_hours=data.get("decay_hours", 24.0),
        auto_compress=data.get("auto_compress", True),
    )
    sessions[session_id] = engine

    return jsonify({"session_id": session_id, "created": True})


@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"deleted": True})
    return jsonify({"error": "Session not found"}), 404


@app.route("/api/sessions")
def list_sessions():
    return jsonify({
        "sessions": [
            {"session_id": sid, "stats": engine.get_stats()}
            for sid, engine in sessions.items()
        ]
    })


# ── Memory Endpoints ────────────────────────────

@app.route("/api/memory/<session_id>", methods=["GET"])
def get_memory(session_id):
    """Get all memories grouped by tier."""
    engine = get_engine(session_id)
    return jsonify({
        "session_id": session_id,
        "memories": engine.get_all_memories(),
        "stats": engine.get_stats()
    })


@app.route("/api/memory/<session_id>/add", methods=["POST"])
def add_memory(session_id):
    """Add a single message to memory."""
    data = request.json or {}
    role = data.get("role", "user")
    content = data.get("content", "")
    metadata = data.get("metadata", {})

    if not content:
        return jsonify({"error": "content is required"}), 400

    engine = get_engine(session_id)
    entry = engine.add_message(role, content, metadata)

    return jsonify({
        "added": True,
        "entry_id": entry.id,
        "token_count": entry.token_count,
        "tier": entry.tier,
        "stats": engine.get_stats()
    })


@app.route("/api/memory/<session_id>/batch", methods=["POST"])
def add_batch(session_id):
    """Add a batch of messages to memory."""
    data = request.json or {}
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "messages array required"}), 400

    engine = get_engine(session_id)
    entries = engine.add_batch(messages)

    return jsonify({
        "added": len(entries),
        "entry_ids": [e.id for e in entries],
        "stats": engine.get_stats()
    })


@app.route("/api/memory/<session_id>/search", methods=["GET"])
def search_memory(session_id):
    """Search memories by query string."""
    query = request.args.get("q", "")
    top_k = int(request.args.get("k", 5))

    if not query:
        return jsonify({"error": "query parameter 'q' required"}), 400

    engine = get_engine(session_id)
    results = engine.search_memory(query, top_k)

    return jsonify({
        "query": query,
        "results": [e.to_dict() for e in results],
        "count": len(results)
    })


@app.route("/api/memory/<session_id>/context", methods=["GET"])
def get_context(session_id):
    """Get the context window for LLM inference."""
    max_tokens = request.args.get("max_tokens", type=int)
    engine = get_engine(session_id)
    context = engine.get_context_window(max_tokens)
    total_tokens = sum(count_tokens(m["content"]) for m in context)

    return jsonify({
        "context": context,
        "message_count": len(context),
        "total_tokens": total_tokens
    })


@app.route("/api/memory/<session_id>/clear/<tier>", methods=["DELETE"])
def clear_tier(session_id, tier):
    """Clear a specific memory tier."""
    valid_tiers = [t.value for t in MemoryTier]
    if tier not in valid_tiers and tier != "all":
        return jsonify({"error": f"Invalid tier. Valid: {valid_tiers + ['all']}"}), 400

    engine = get_engine(session_id)
    if tier == "all":
        engine.reset()
        return jsonify({"cleared": "all"})

    count = engine.clear_tier(tier)
    return jsonify({"cleared": tier, "entries_removed": count})


# ── Compression Endpoints ────────────────────────

@app.route("/api/compress/<session_id>", methods=["POST"])
def compress_memory(session_id):
    """Manually trigger compression with a specified strategy."""
    data = request.json or {}
    strategy_str = data.get("strategy", "hybrid").lower()

    strategy_map = {
        "summarize":    CompressionStrategy.SUMMARIZE,
        "deduplicate":  CompressionStrategy.DEDUPLICATE,
        "score_prune":  CompressionStrategy.SCORE_PRUNE,
        "hierarchical": CompressionStrategy.HIERARCHICAL,
        "hybrid":       CompressionStrategy.HYBRID,
    }

    strategy = strategy_map.get(strategy_str, CompressionStrategy.HYBRID)
    target_budget = data.get("target_budget")

    engine = get_engine(session_id)
    stats = engine.compress(strategy=strategy, target_budget=target_budget)

    return jsonify({
        "compression_applied": True,
        "strategy": strategy_str,
        "stats": {
            "original_tokens": stats.original_tokens,
            "compressed_tokens": stats.compressed_tokens,
            "tokens_saved": stats.tokens_saved,
            "savings_pct": stats.savings_pct,
            "compression_ratio": stats.compression_ratio,
            "entries_before": stats.entries_before,
            "entries_after": stats.entries_after,
            "time_ms": stats.time_ms,
        },
        "memory_stats": engine.get_stats()
    })


@app.route("/api/compress/<session_id>/history")
def compression_history(session_id):
    """Get compression history for a session."""
    engine = get_engine(session_id)
    stats = engine.get_stats()
    return jsonify({
        "session_id": session_id,
        "compression_log": stats["compression_log"],
        "total_compressions": stats["total_compressions"]
    })


# ── Chat Endpoint ───────────────────────────────

@app.route("/api/chat/<session_id>", methods=["POST"])
def chat(session_id):
    """
    Full chat endpoint with memory-aware context injection.
    1. Add user message to memory
    2. Build compressed context window
    3. Send to LLM with context
    4. Add assistant response to memory
    5. Return response + memory stats
    """
    data = request.json or {}
    user_message = data.get("message", "")
    system_prompt = data.get("system_prompt", "You are a helpful AI assistant with persistent memory.")
    max_context_tokens = data.get("max_context_tokens", 3000)

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    engine = get_engine(session_id)

    if not engine.client:
        return jsonify({"error": "OpenAI API key not configured"}), 503

    # 1. Add user message
    engine.add_message("user", user_message)

    # 2. Build context window
    context_messages = engine.get_context_window(max_tokens=max_context_tokens)

    # 3. Build final message list for LLM
    llm_messages = [{"role": "system", "content": system_prompt}] + context_messages

    try:
        # 4. Call LLM
        response = engine.client.chat.completions.create(
            model=engine.model,
            messages=llm_messages,
            temperature=0.7,
            max_tokens=800
        )
        assistant_reply = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        # 5. Add assistant response to memory
        engine.add_message("assistant", assistant_reply)

        return jsonify({
            "reply": assistant_reply,
            "tokens_used": tokens_used,
            "context_messages": len(context_messages),
            "memory_stats": engine.get_stats()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Visualization Endpoint ──────────────────────

@app.route("/api/visualize/<session_id>")
def visualize(session_id):
    """Return data optimized for frontend visualization."""
    engine = get_engine(session_id)
    stats = engine.get_stats()
    all_memories = engine.get_all_memories()

    # Build timeline data
    timeline = []
    for tier, entries in all_memories.items():
        for e in entries:
            timeline.append({
                "id": e["id"],
                "tier": tier,
                "role": e["role"],
                "preview": e["content"][:100] + "..." if len(e["content"]) > 100 else e["content"],
                "tokens": e["token_count"],
                "importance": e["importance"],
                "compressed": e["compressed"],
                "timestamp": e["timestamp"],
                "access_count": e["access_count"],
            })

    timeline.sort(key=lambda x: x["timestamp"])

    # Token distribution
    token_dist = {
        tier: stats["tiers"][tier]["tokens"]
        for tier in stats["tiers"]
    }

    # Compression history for chart
    compression_chart = [
        {
            "label": f"#{i+1}",
            "strategy": log.get("strategy_used", ""),
            "before": log.get("original_tokens", 0),
            "after": log.get("compressed_tokens", 0),
            "savings_pct": log.get("savings_pct", 0)
        }
        for i, log in enumerate(engine.compression_log)
    ]

    return jsonify({
        "stats": stats,
        "timeline": timeline,
        "token_distribution": token_dist,
        "compression_chart": compression_chart,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
