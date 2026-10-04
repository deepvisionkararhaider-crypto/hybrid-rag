// core.js — shared RAG primitives for the Vanilla RAG Worker.
// Embeddings: Cloudflare Workers AI (@cf/baai/bge-small-en-v1.5, 384-dim)
// LLM: Groq OpenAI-compatible API (model from env)
// Vector store: D1 (vectors stored as JSON, exact cosine search in-Worker)

export const EMBED_MODEL = '@cf/baai/bge-small-en-v1.5';

// ── Chunking: sentence-aware, configurable size/overlap ───────────────────────
export function chunkText(text, size = 800, overlap = 120) {
  const clean = (text || '').replace(/\r/g, '').replace(/[ \t]+/g, ' ').trim();
  if (!clean) return [];
  const sentences = clean.match(/[^.!?\n]+[.!?]*/g) || [clean];
  const chunks = [];
  let buf = '';
  for (const s of sentences) {
    const piece = s.trim();
    if (!piece) continue;
    if ((buf + ' ' + piece).trim().length > size && buf) {
      chunks.push(buf.trim());
      buf = buf.slice(Math.max(0, buf.length - overlap)); // overlap tail
    }
    buf = (buf + ' ' + piece).trim();
  }
  if (buf.trim()) chunks.push(buf.trim());
  return chunks.filter((c) => c.length > 0);
}

// ── Embeddings via Workers AI ─────────────────────────────────────────────────
export async function embed(env, texts) {
  const out = await env.AI.run(EMBED_MODEL, { text: texts });
  return out.data; // array of arrays
}

// ── Cosine similarity ─────────────────────────────────────────────────────────
export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── Groq chat completion (OpenAI-compatible) ──────────────────────────────────
export async function chat(env, messages, { temperature = 0.1, max_tokens = 1024 } = {}) {
  const key = env.GROQ_API_KEY;
  if (!key) throw new Error('GROQ_API_KEY not configured');
  const model = env.LLM_MODEL || 'openai/gpt-oss-120b';
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model, messages, temperature, max_tokens }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Groq ${res.status}: ${t.slice(0, 300)}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? '';
}

// ── D1 schema bootstrap (idempotent) ──────────────────────────────────────────
export async function ensureSchema(env) {
  await env.DB.prepare(
    `CREATE TABLE IF NOT EXISTS documents (
       id TEXT PRIMARY KEY,
       filename TEXT NOT NULL,
       content TEXT NOT NULL,
       created_at INTEGER NOT NULL
     )`
  ).run();
  await env.DB.prepare(
    `CREATE TABLE IF NOT EXISTS chunks (
       id TEXT PRIMARY KEY,
       doc_id TEXT NOT NULL,
       filename TEXT NOT NULL,
       chunk_index INTEGER NOT NULL,
       content TEXT NOT NULL,
       embedding TEXT NOT NULL,
       created_at INTEGER NOT NULL
     )`
  ).run();
  await env.DB.prepare(`CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)`).run();
}

export function uid() {
  return crypto.randomUUID();
}
