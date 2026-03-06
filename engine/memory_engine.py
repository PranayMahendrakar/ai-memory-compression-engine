"""
AI Memory Compression Engine - Core Module
==========================================
Multi-strategy memory compression system for LLMs:
  1. Hierarchical Memory Tiers (working / episodic / semantic / archival)
  2. Importance Scoring (recency + frequency + semantic weight)
  3. LLM-powered Summarization Compression
  4. Semantic Deduplication (cosine similarity on embeddings)
  5. Token Budget Enforcement
"""

import json
import time
import math
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import openai
import os


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class MemoryTier(Enum):
    WORKING   = "working"    # Recent turns, full fidelity (~last 10 msgs)
    EPISODIC  = "episodic"   # Compressed summaries of recent sessions
    SEMANTIC  = "semantic"   # Distilled facts / long-term knowledge
    ARCHIVAL  = "archival"   # Highly compressed, rarely retrieved


class CompressionStrategy(Enum):
    SUMMARIZE    = "summarize"     # LLM-based summarization
    DEDUPLICATE  = "deduplicate"   # Remove semantically redundant entries
    SCORE_PRUNE  = "score_prune"   # Drop lowest-importance memories
    HIERARCHICAL = "hierarchical"  # Promote/demote across tiers
    HYBRID       = "hybrid"        # All strategies combined


@dataclass
class MemoryEntry:
    id: str
    role: str                       # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    tier: str = MemoryTier.WORKING.value
    importance: float = 1.0         # 0.0 – 1.0
    access_count: int = 0
    token_count: int = 0
    compressed: bool = False
    original_tokens: int = 0        # Tokens before compression
    summary_of: List[str] = field(default_factory=list)  # IDs summarized
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_llm_message(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("embedding", None)   # Don't serialize large embeddings to JSON
        return d


@dataclass
class CompressionStats:
    original_tokens: int = 0
    compressed_tokens: int = 0
    entries_before: int = 0
    entries_after: int = 0
    strategy_used: str = ""
    compression_ratio: float = 0.0
    time_ms: float = 0.0
    tokens_saved: int = 0

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round((self.tokens_saved / self.original_tokens) * 100, 1)


# ─────────────────────────────────────────────
# Token Counter (tiktoken-lite fallback)
# ─────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count. Uses tiktoken if available, else heuristic."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Rough heuristic: ~4 chars per token
        return max(1, len(text) // 4)


# ─────────────────────────────────────────────
# Importance Scorer
# ─────────────────────────────────────────────

class ImportanceScorer:
    """
    Scores memory entries using multiple signals:
      - Recency decay (exponential)
      - Access frequency
      - Semantic keyword boosting
      - Role weight (system > user > assistant)
    """

    ROLE_WEIGHTS = {"system": 1.5, "user": 1.0, "assistant": 0.9}
    BOOST_KEYWORDS = [
        "important", "remember", "must", "critical", "key", "decision",
        "agreed", "action", "todo", "deadline", "name", "always", "never",
        "requirement", "constraint", "goal", "objective", "priority"
    ]

    def __init__(self, decay_hours: float = 24.0):
        self.decay_hours = decay_hours

    def score(self, entry: MemoryEntry, now: float = None) -> float:
        if now is None:
            now = time.time()

        # 1. Recency decay
        age_hours = (now - entry.timestamp) / 3600
        recency = math.exp(-age_hours / self.decay_hours)

        # 2. Access frequency boost (log scale)
        frequency = math.log1p(entry.access_count) * 0.15

        # 3. Role weight
        role_w = self.ROLE_WEIGHTS.get(entry.role, 1.0)

        # 4. Keyword boost
        content_lower = entry.content.lower()
        keyword_hits = sum(1 for kw in self.BOOST_KEYWORDS if kw in content_lower)
        keyword_boost = min(0.3, keyword_hits * 0.05)

        # 5. Length penalty (very short entries are less valuable)
        length_factor = min(1.0, entry.token_count / 20) if entry.token_count > 0 else 0.5

        raw = (recency * 0.45 + frequency * 0.20 + keyword_boost * 0.15 + length_factor * 0.20) * role_w
        return min(1.0, round(raw, 4))

    def rescore_all(self, entries: List[MemoryEntry]) -> None:
        now = time.time()
        for e in entries:
            e.importance = self.score(e, now)


# ─────────────────────────────────────────────
# Semantic Deduplicator
# ─────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Detects and removes near-duplicate memory entries using:
      - Exact hash matching
      - Simple n-gram overlap (no embedding needed by default)
      - Optional embedding cosine similarity (if OpenAI client provided)
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    @staticmethod
    def _ngrams(text: str, n: int = 3) -> set:
        tokens = text.lower().split()
        return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set(tokens)

    def _jaccard(self, a: str, b: str) -> float:
        g1, g2 = self._ngrams(a), self._ngrams(b)
        if not g1 and not g2:
            return 1.0
        if not g1 or not g2:
            return 0.0
        intersection = len(g1 & g2)
        union = len(g1 | g2)
        return intersection / union if union > 0 else 0.0

    def find_duplicates(self, entries: List[MemoryEntry]) -> List[Tuple[str, str, float]]:
        """Return list of (id_a, id_b, similarity) for near-duplicates."""
        duplicates = []
        seen_hashes = {}

        for i, e1 in enumerate(entries):
            h = hashlib.md5(e1.content.strip().lower().encode()).hexdigest()
            if h in seen_hashes:
                duplicates.append((seen_hashes[h], e1.id, 1.0))
                continue
            seen_hashes[h] = e1.id

            for e2 in entries[i+1:]:
                sim = self._jaccard(e1.content, e2.content)
                if sim >= self.threshold:
                    duplicates.append((e1.id, e2.id, round(sim, 3)))

        return duplicates

    def deduplicate(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """Remove duplicates, keeping higher-importance entry. Returns (deduped, count_removed)."""
        pairs = self.find_duplicates(entries)
        to_remove = set()

        for id_a, id_b, _ in pairs:
            ea = next((e for e in entries if e.id == id_a), None)
            eb = next((e for e in entries if e.id == id_b), None)
            if ea and eb:
                remove_id = id_b if (ea.importance >= eb.importance) else id_a
                to_remove.add(remove_id)

        original_count = len(entries)
        deduped = [e for e in entries if e.id not in to_remove]
        return deduped, original_count - len(deduped)


# ─────────────────────────────────────────────
# LLM Summarizer
# ─────────────────────────────────────────────

class LLMSummarizer:
    """
    Uses OpenAI GPT to compress a block of conversation turns
    into a dense, information-preserving summary.
    """

    SYSTEM_PROMPT = """You are a memory compression specialist for AI systems.
Your task: compress the given conversation turns into a compact memory summary.

Rules:
- Preserve ALL key facts, decisions, names, numbers, and action items
- Remove filler, repetition, and low-value exchanges  
- Use dense, telegraphic prose (not bullet points unless listing facts)
- Target: 30-40% of original length while retaining 90%+ of information value
- Start with "[MEMORY SUMMARY]" tag
- End with "[/MEMORY SUMMARY]" tag
"""

    def __init__(self, client: openai.OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def summarize(self, entries: List[MemoryEntry], context: str = "") -> Tuple[str, int, int]:
        """
        Summarize a list of memory entries.
        Returns (summary_text, original_tokens, compressed_tokens)
        """
        conversation_text = ""
        original_tokens = 0
        for e in entries:
            line = f"[{e.role.upper()}]: {e.content}"
            conversation_text += line + "\n"
            original_tokens += e.token_count or count_tokens(e.content)

        prompt = f"""Compress these conversation turns into a memory summary:

{conversation_text}
{f"Additional context: {context}" if context else ""}

Produce a compressed memory summary that preserves essential information."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )

        summary = response.choices[0].message.content.strip()
        compressed_tokens = count_tokens(summary)
        return summary, original_tokens, compressed_tokens

    def extract_facts(self, entries: List[MemoryEntry]) -> str:
        """Extract discrete facts from memory for the semantic tier."""
        conversation_text = "\n".join(
            f"[{e.role.upper()}]: {e.content}" for e in entries
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Extract a minimal set of key facts from this conversation as a compact JSON list. Each fact: {'fact': str, 'confidence': 0-1}."},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Core Memory Engine
# ─────────────────────────────────────────────

class MemoryCompressionEngine:
    """
    Main engine that manages memory across tiers and applies
    compression strategies to stay within a token budget.

    Architecture:
        Working Memory  → recent, full-fidelity turns
        Episodic Memory → LLM-summarized session blocks  
        Semantic Memory → extracted facts / knowledge distillation
        Archival Memory → heavily compressed, rarely accessed history
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        token_budget: int = 4000,
        working_window: int = 10,
        importance_threshold: float = 0.2,
        dedup_threshold: float = 0.85,
        decay_hours: float = 24.0,
        auto_compress: bool = True,
    ):
        self.model = model
        self.token_budget = token_budget
        self.working_window = working_window
        self.importance_threshold = importance_threshold
        self.auto_compress = auto_compress

        # OpenAI client
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=key) if key else None

        # Sub-components
        self.scorer = ImportanceScorer(decay_hours=decay_hours)
        self.deduplicator = SemanticDeduplicator(dedup_threshold)
        self.summarizer = LLMSummarizer(self.client, model) if self.client else None

        # Memory stores by tier
        self.memories: Dict[str, List[MemoryEntry]] = {
            MemoryTier.WORKING.value:  [],
            MemoryTier.EPISODIC.value: [],
            MemoryTier.SEMANTIC.value: [],
            MemoryTier.ARCHIVAL.value: [],
        }

        # Compression history
        self.compression_log: List[Dict] = []
        self._id_counter = 0

    # ── Entry Management ──────────────────────

    def _new_id(self) -> str:
        self._id_counter += 1
        return f"mem_{int(time.time())}_{self._id_counter}"

    def add_message(self, role: str, content: str, metadata: Dict = None) -> MemoryEntry:
        """Add a new message to working memory."""
        token_count = count_tokens(content)
        entry = MemoryEntry(
            id=self._new_id(),
            role=role,
            content=content,
            timestamp=time.time(),
            tier=MemoryTier.WORKING.value,
            token_count=token_count,
            original_tokens=token_count,
            importance=1.0,
            metadata=metadata or {}
        )
        self.memories[MemoryTier.WORKING.value].append(entry)

        if self.auto_compress:
            self._auto_manage()

        return entry

    def add_batch(self, messages: List[Dict[str, str]]) -> List[MemoryEntry]:
        """Add multiple messages at once."""
        entries = []
        for msg in messages:
            e = self.add_message(msg.get("role", "user"), msg.get("content", ""))
            entries.append(e)
        return entries

    def _all_entries(self) -> List[MemoryEntry]:
        result = []
        for tier_entries in self.memories.values():
            result.extend(tier_entries)
        return result

    def _total_tokens(self) -> int:
        return sum(e.token_count for e in self._all_entries())

    # ── Auto Management ────────────────────────

    def _auto_manage(self):
        """Triggered after each message. Promotes/compresses as needed."""
        working = self.memories[MemoryTier.WORKING.value]

        # Promote old entries from working when window is exceeded
        if len(working) > self.working_window:
            to_promote = working[:-self.working_window]
            self.memories[MemoryTier.WORKING.value] = working[-self.working_window:]

            for e in to_promote:
                e.tier = MemoryTier.EPISODIC.value
                self.memories[MemoryTier.EPISODIC.value].append(e)

        # Compress if over token budget
        if self._total_tokens() > self.token_budget:
            self.compress(strategy=CompressionStrategy.HYBRID)

    # ── Compression Strategies ─────────────────

    def compress(
        self,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        target_budget: Optional[int] = None
    ) -> CompressionStats:
        """Apply compression strategy to reduce token usage."""
        t0 = time.time()
        budget = target_budget or self.token_budget
        all_before = self._all_entries()
        tokens_before = sum(e.token_count for e in all_before)
        count_before = len(all_before)

        if strategy == CompressionStrategy.SCORE_PRUNE:
            self._strategy_score_prune(budget)
        elif strategy == CompressionStrategy.DEDUPLICATE:
            self._strategy_deduplicate()
        elif strategy == CompressionStrategy.SUMMARIZE:
            self._strategy_summarize()
        elif strategy == CompressionStrategy.HIERARCHICAL:
            self._strategy_hierarchical()
        elif strategy == CompressionStrategy.HYBRID:
            self._strategy_deduplicate()
            self._strategy_hierarchical()
            if self._total_tokens() > budget:
                self._strategy_summarize()
            if self._total_tokens() > budget:
                self._strategy_score_prune(budget)

        all_after = self._all_entries()
        tokens_after = sum(e.token_count for e in all_after)

        stats = CompressionStats(
            original_tokens=tokens_before,
            compressed_tokens=tokens_after,
            entries_before=count_before,
            entries_after=len(all_after),
            strategy_used=strategy.value,
            compression_ratio=round(tokens_after / max(1, tokens_before), 4),
            time_ms=round((time.time() - t0) * 1000, 1),
            tokens_saved=tokens_before - tokens_after,
        )

        self.compression_log.append({
            **stats.__dict__,
            "timestamp": time.time(),
            "savings_pct": stats.savings_pct
        })

        return stats

    def _strategy_score_prune(self, budget: int):
        """Rescore all memories and drop lowest-importance entries to meet budget."""
        self.scorer.rescore_all(self._all_entries())

        for tier_name in [MemoryTier.ARCHIVAL.value, MemoryTier.SEMANTIC.value, MemoryTier.EPISODIC.value]:
            entries = self.memories[tier_name]
            entries.sort(key=lambda e: e.importance)
            while self._total_tokens() > budget and entries:
                removed = entries.pop(0)
                if removed.importance < self.importance_threshold:
                    pass  # discarded
                else:
                    break  # stop if importance is high enough

    def _strategy_deduplicate(self):
        """Remove near-duplicate entries across all tiers."""
        for tier_name, entries in self.memories.items():
            if len(entries) < 2:
                continue
            deduped, removed = self.deduplicator.deduplicate(entries)
            self.memories[tier_name] = deduped

    def _strategy_summarize(self):
        """Compress episodic memory using LLM summarization."""
        if not self.summarizer:
            # Fallback: truncation-based compression
            self._truncation_compress()
            return

        episodic = self.memories[MemoryTier.EPISODIC.value]
        if len(episodic) < 3:
            return

        # Chunk into groups of ~5 for summarization
        chunk_size = 5
        new_episodic = []
        i = 0
        while i < len(episodic):
            chunk = episodic[i:i+chunk_size]
            if len(chunk) >= 3:
                try:
                    summary_text, orig_tokens, comp_tokens = self.summarizer.summarize(chunk)
                    summary_entry = MemoryEntry(
                        id=self._new_id(),
                        role="system",
                        content=summary_text,
                        timestamp=chunk[-1].timestamp,
                        tier=MemoryTier.EPISODIC.value,
                        token_count=comp_tokens,
                        original_tokens=orig_tokens,
                        compressed=True,
                        summary_of=[e.id for e in chunk],
                        importance=max(e.importance for e in chunk),
                    )
                    new_episodic.append(summary_entry)
                except Exception:
                    new_episodic.extend(chunk)  # fallback: keep originals
            else:
                new_episodic.extend(chunk)
            i += chunk_size

        self.memories[MemoryTier.EPISODIC.value] = new_episodic

    def _strategy_hierarchical(self):
        """Demote old episodic summaries to semantic/archival tier."""
        episodic = self.memories[MemoryTier.EPISODIC.value]
        self.scorer.rescore_all(episodic)
        episodic.sort(key=lambda e: e.importance)

        # Move lowest-importance episodic to archival
        cutoff = max(0, len(episodic) - 8)  # keep at most 8 episodic entries
        to_archive = episodic[:cutoff]
        keep = episodic[cutoff:]

        for e in to_archive:
            e.tier = MemoryTier.ARCHIVAL.value
            self.memories[MemoryTier.ARCHIVAL.value].append(e)

        self.memories[MemoryTier.EPISODIC.value] = keep

    def _truncation_compress(self):
        """Naive fallback: keep only the highest-importance entries."""
        episodic = self.memories[MemoryTier.EPISODIC.value]
        self.scorer.rescore_all(episodic)
        episodic.sort(key=lambda e: e.importance, reverse=True)
        # Keep top 50%
        keep_count = max(2, len(episodic) // 2)
        self.memories[MemoryTier.EPISODIC.value] = episodic[:keep_count]

    # ── Memory Retrieval ───────────────────────

    def get_context_window(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Build the context window for LLM inference.
        Returns list of {role, content} dicts sorted by relevance.
        """
        budget = max_tokens or self.token_budget
        messages = []
        used_tokens = 0

        # 1. Include all working memory (most recent, highest priority)
        for e in self.memories[MemoryTier.WORKING.value]:
            e.access_count += 1
            messages.append(e.to_llm_message())
            used_tokens += e.token_count

        # 2. Add episodic summaries if budget allows
        episodic = sorted(
            self.memories[MemoryTier.EPISODIC.value],
            key=lambda e: e.importance, reverse=True
        )
        for e in episodic:
            if used_tokens + e.token_count > budget:
                break
            e.access_count += 1
            messages.insert(0, e.to_llm_message())
            used_tokens += e.token_count

        # 3. Prepend semantic memories as system context
        semantic = self.memories[MemoryTier.SEMANTIC.value]
        for e in semantic:
            if used_tokens + e.token_count > budget:
                break
            messages.insert(0, {"role": "system", "content": f"[FACT] {e.content}"})
            used_tokens += e.token_count

        return messages

    def search_memory(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Simple keyword search across all memory tiers."""
        query_lower = query.lower()
        scored = []
        for e in self._all_entries():
            score = sum(1 for word in query_lower.split() if word in e.content.lower())
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # ── Stats & Reporting ──────────────────────

    def get_stats(self) -> Dict:
        """Return comprehensive memory statistics."""
        tier_stats = {}
        total_tokens = 0
        total_entries = 0

        for tier_name, entries in self.memories.items():
            tokens = sum(e.token_count for e in entries)
            compressed_entries = sum(1 for e in entries if e.compressed)
            tier_stats[tier_name] = {
                "entries": len(entries),
                "tokens": tokens,
                "compressed_entries": compressed_entries,
                "avg_importance": round(
                    sum(e.importance for e in entries) / max(1, len(entries)), 3
                )
            }
            total_tokens += tokens
            total_entries += len(entries)

        return {
            "total_tokens": total_tokens,
            "total_entries": total_entries,
            "token_budget": self.token_budget,
            "budget_used_pct": round(total_tokens / max(1, self.token_budget) * 100, 1),
            "tiers": tier_stats,
            "compression_log": self.compression_log[-10:],  # Last 10 compressions
            "total_compressions": len(self.compression_log),
        }

    def get_all_memories(self) -> Dict[str, List[Dict]]:
        """Return all memories serialized by tier."""
        return {
            tier: [e.to_dict() for e in entries]
            for tier, entries in self.memories.items()
        }

    def clear_tier(self, tier: str) -> int:
        """Clear a specific memory tier. Returns count cleared."""
        count = len(self.memories.get(tier, []))
        self.memories[tier] = []
        return count

    def reset(self):
        """Reset all memory."""
        for tier in self.memories:
            self.memories[tier] = []
        self.compression_log = []
        self._id_counter = 0
