/**
 * AI Memory Compression Engine - Frontend
 * Real-time memory visualization, chat, and compression controls
 */

// ── State ───────────────────────────────────────
let sessionId = 'demo_session';
let allTimeline = [];
let activeFilter = 'all';

// ── Init ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  sessionId = document.getElementById('sessionId').value;
  checkStatus();
  setInterval(refreshVisualize, 8000); // Auto-refresh every 8s
});

// ── API Status ──────────────────────────────────
async function checkStatus() {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  dot.className = 'status-dot checking';
  label.textContent = 'Checking...';
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    if (d.api_key_configured) {
      dot.className = 'status-dot online';
      label.textContent = 'API Ready';
    } else {
      dot.className = 'status-dot offline';
      label.textContent = 'API Key Missing';
    }
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'Offline';
  }
}

// ── Session ─────────────────────────────────────
async function createSession() {
  const newId = 'session_' + Date.now();
  document.getElementById('sessionId').value = newId;
  sessionId = newId;
  await fetch('/api/session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ session_id: newId })
  });
  document.getElementById('chatMessages').innerHTML = `
    <div class="chat-welcome">
      <div class="welcome-icon">🧠</div>
      <p>New session created: <code>${newId}</code></p>
    </div>`;
  refreshVisualize();
}

// ── Chat ─────────────────────────────────────────
function handleEnter(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
}

async function sendChat() {
  const input = document.getElementById('userInput');
  const message = input.value.trim();
  if (!message) return;

  sessionId = document.getElementById('sessionId').value;
  const systemPrompt = document.getElementById('systemPrompt').value;

  // Append user message to chat
  appendChatMessage('user', message);
  input.value = '';

  // Show thinking indicator
  const thinkingId = appendThinking();
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('chatMeta').textContent = 'Compressing memory & generating response...';

  try {
    const r = await fetch(`/api/chat/${sessionId}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message, system_prompt: systemPrompt })
    });
    const data = await r.json();

    removeThinking(thinkingId);

    if (data.error) {
      appendChatMessage('system-note', `⚠️ Error: ${data.error}`);
    } else {
      appendChatMessage('assistant', data.reply, data.tokens_used);
      document.getElementById('chatMeta').textContent =
        `${data.tokens_used || 0} tokens used · ${data.context_messages || 0} context messages · memory managed`;
      updateStatsFromResponse(data.memory_stats);
    }
  } catch (err) {
    removeThinking(thinkingId);
    appendChatMessage('system-note', `⚠️ Request failed: ${err.message}`);
  } finally {
    document.getElementById('sendBtn').disabled = false;
    refreshVisualize();
  }
}

function appendChatMessage(role, content, tokensUsed) {
  const msgs = document.getElementById('chatMessages');
  const welcome = msgs.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;

  const roleLabel = role === 'system-note' ? '' : `<div class="msg-role">${role}</div>`;
  const tokenInfo = tokensUsed ? `<div class="msg-tokens">${tokensUsed} tokens</div>` : '';
  div.innerHTML = roleLabel + `<div>${escapeHtml(content)}</div>` + tokenInfo;

  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function appendThinking() {
  const msgs = document.getElementById('chatMessages');
  const id = 'thinking_' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'chat-msg assistant';
  div.innerHTML = `<div class="thinking">
    <div class="thinking-dot"></div>
    <div class="thinking-dot"></div>
    <div class="thinking-dot"></div>
  </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Compression ──────────────────────────────────
async function compress(strategy) {
  sessionId = document.getElementById('sessionId').value;
  try {
    const r = await fetch(`/api/compress/${sessionId}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ strategy })
    });
    const data = await r.json();

    if (data.error) {
      alert('Compression error: ' + data.error);
      return;
    }

    const s = data.stats;
    // Show compression result
    document.getElementById('compressionResult').classList.remove('hidden');
    document.getElementById('crStrategy').textContent = strategy.toUpperCase();
    document.getElementById('crSaved').textContent = s.tokens_saved;
    document.getElementById('crRatio').textContent = (s.compression_ratio * 100).toFixed(0) + '%';
    document.getElementById('crTime').textContent = s.time_ms + 'ms';

    // Flash the result
    const cr = document.getElementById('compressionResult');
    cr.style.background = 'rgba(34,211,174,0.15)';
    setTimeout(() => cr.style.background = '', 800);

    updateStatsFromResponse(data.memory_stats);
    refreshVisualize();
    appendChatMessage('system-note', `🗜️ Compression [${strategy}] — saved ${s.tokens_saved} tokens (${s.savings_pct}%), ratio: ${s.compression_ratio}`);
  } catch (err) {
    alert('Failed: ' + err.message);
  }
}

// ── Memory Injection ─────────────────────────────
async function injectMemory() {
  sessionId = document.getElementById('sessionId').value;
  const role = document.getElementById('injectRole').value;
  const content = document.getElementById('injectContent').value.trim();
  if (!content) return;

  try {
    const r = await fetch(`/api/memory/${sessionId}/add`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ role, content })
    });
    const data = await r.json();

    if (data.added) {
      document.getElementById('injectContent').value = '';
      appendChatMessage('system-note', `✅ Injected [${role}] — ${data.token_count} tokens → tier: ${data.tier}`);
      updateStatsFromResponse(data.stats);
      refreshVisualize();
    }
  } catch (err) {
    alert('Injection failed: ' + err.message);
  }
}

// ── Search ───────────────────────────────────────
async function searchMemory() {
  sessionId = document.getElementById('sessionId').value;
  const query = document.getElementById('searchQuery').value.trim();
  if (!query) return;

  try {
    const r = await fetch(`/api/memory/${sessionId}/search?q=${encodeURIComponent(query)}&k=8`);
    const data = await r.json();
    renderSearchResults(data.results);
  } catch (err) {
    document.getElementById('searchResults').innerHTML = `<div style="color:var(--red);font-size:12px">${err.message}</div>`;
  }
}

function renderSearchResults(results) {
  const container = document.getElementById('searchResults');
  if (!results.length) {
    container.innerHTML = '<div style="font-size:12px;color:var(--text-3)">No results found.</div>';
    return;
  }
  container.innerHTML = results.map(r => `
    <div class="search-result-item">
      <div class="sr-tier">${r.tier} · ${r.role} · ${r.importance?.toFixed(2) || '-'} importance</div>
      <div class="sr-preview">${escapeHtml(r.content?.substring(0, 120) || '')}...</div>
    </div>
  `).join('');
}

// ── Clear ────────────────────────────────────────
async function clearTier(tier) {
  if (!confirm(`Clear all ${tier} memories?`)) return;
  sessionId = document.getElementById('sessionId').value;
  await fetch(`/api/memory/${sessionId}/clear/${tier}`, { method: 'DELETE' });
  appendChatMessage('system-note', `🗑️ Cleared ${tier} memory tier`);
  refreshVisualize();
}

async function clearAll() {
  if (!confirm('Clear ALL memory? This cannot be undone.')) return;
  sessionId = document.getElementById('sessionId').value;
  await fetch(`/api/memory/${sessionId}/clear/all`, { method: 'DELETE' });
  document.getElementById('chatMessages').innerHTML = `
    <div class="chat-welcome">
      <div class="welcome-icon">🧠</div>
      <p>Memory cleared. Start fresh!</p>
    </div>`;
  refreshVisualize();
}

// ── Visualizer ───────────────────────────────────
async function refreshVisualize() {
  sessionId = document.getElementById('sessionId').value;
  try {
    const r = await fetch(`/api/visualize/${sessionId}`);
    const data = await r.json();
    updateBudgetBar(data.stats);
    updateTierCards(data.stats);
    allTimeline = data.timeline || [];
    renderTimeline(allTimeline, activeFilter);
    renderCompressionChart(data.compression_chart || []);
    updateLiveStats(data.stats);
  } catch {}
}

function updateBudgetBar(stats) {
  const pct = stats.budget_used_pct || 0;
  const fill = document.getElementById('budgetFill');
  const label = document.getElementById('budgetLabel');
  const pctEl = document.getElementById('budgetPct');

  fill.style.width = Math.min(100, pct) + '%';
  fill.className = 'budget-bar-fill' + (pct > 90 ? ' danger' : pct > 70 ? ' warn' : '');
  label.textContent = `${stats.total_tokens} / ${stats.token_budget} tokens`;
  pctEl.textContent = pct + '%';
}

function updateTierCards(stats) {
  const tiers = stats.tiers || {};
  for (const [tier, info] of Object.entries(tiers)) {
    const countEl = document.getElementById('count-' + tier);
    const tokenEl = document.getElementById('tokens-' + tier);
    if (countEl) countEl.textContent = info.entries;
    if (tokenEl) tokenEl.textContent = info.tokens + ' tok';
  }
}

function filterTimeline(tier, btn) {
  activeFilter = tier;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTimeline(allTimeline, tier);
}

function renderTimeline(entries, filter) {
  const container = document.getElementById('memoryTimeline');
  const filtered = filter === 'all' ? entries : entries.filter(e => e.tier === filter);

  if (!filtered.length) {
    container.innerHTML = '<div class="empty-timeline">No memories in this tier.</div>';
    return;
  }

  container.innerHTML = [...filtered].reverse().map(e => {
    const importancePct = Math.round((e.importance || 0) * 100);
    const compBadge = e.compressed ? '<span class="compressed-badge">compressed</span>' : '';
    return `
      <div class="tl-entry" data-tier="${e.tier}">
        <span class="tl-tier-badge ${e.tier}">${e.tier.substring(0,3)}</span>
        <div class="tl-content">
          <div class="tl-preview">${escapeHtml(e.preview || '')}</div>
          <div class="tl-meta">${e.role} · ${e.tokens}tok · acc:${e.access_count} ${compBadge}</div>
        </div>
        <div class="tl-importance">
          <div class="importance-bar"><div class="importance-fill" style="width:${importancePct}%"></div></div>
          <span class="importance-val">${(e.importance || 0).toFixed(2)}</span>
        </div>
      </div>
    `;
  }).join('');
}

function renderCompressionChart(chartData) {
  const container = document.getElementById('compressionChart');
  if (!chartData.length) {
    container.innerHTML = '<div class="empty-chart">No compressions yet.</div>';
    return;
  }

  const maxBefore = Math.max(...chartData.map(d => d.before), 1);
  container.innerHTML = chartData.map(d => {
    const beforeH = Math.round((d.before / maxBefore) * 60);
    const afterH = Math.round((d.after / maxBefore) * 60);
    return `
      <div class="chart-bar-group">
        <div class="chart-bars">
          <div class="chart-bar before" style="height:${beforeH}px" title="Before: ${d.before} tokens"></div>
          <div class="chart-bar after" style="height:${afterH}px" title="After: ${d.after} tokens"></div>
        </div>
        <div class="chart-label">${d.label}</div>
        <div class="chart-savings">-${d.savings_pct}%</div>
      </div>
    `;
  }).join('');
}

function updateLiveStats(stats) {
  document.getElementById('statTotal').textContent = stats.total_tokens || 0;
  document.getElementById('statEntries').textContent = stats.total_entries || 0;
  document.getElementById('statCompressions').textContent = stats.total_compressions || 0;
  document.getElementById('statBudget').textContent = (stats.budget_used_pct || 0) + '%';
}

function updateStatsFromResponse(memStats) {
  if (!memStats) return;
  updateBudgetBar(memStats);
  updateTierCards(memStats);
  updateLiveStats(memStats);
}

// ── Utils ────────────────────────────────────────
function escapeHtml(text) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(String(text || '')));
  return d.innerHTML;
}
