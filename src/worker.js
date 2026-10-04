// worker.js — 02-hybrid-rag
// Hybrid retrieval: BM25 (lexical) + dense cosine (semantic) fused with
// Reciprocal Rank Fusion (RRF). Grounded answer via Groq + citations.
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { chunkText, embed, cosine, chat, ensureSchema, uid } from './core.js';

const app = new Hono();
app.use('/api/*', cors());

const SAMPLE_DOCS = [
  {
    filename: 'hybrid-retrieval.md',
    content: `Hybrid retrieval combines lexical search and semantic search. Lexical search, most famously BM25, scores documents by exact term overlap, term frequency, and inverse document frequency. It is unbeatable when the user types a precise keyword, an error code, a product SKU, or a rare proper noun, because those tokens must literally appear.
Dense retrieval, in contrast, embeds the query and the documents into the same vector space and ranks by cosine similarity. It captures paraphrase and synonyms: a query about "car maintenance" can match a document titled "automobile servicing" even with no shared words. Its weakness is that it can miss exact identifiers that a bag-of-words model would never drop.
Neither method dominates. Benchmark studies repeatedly show that fusing the two beats either alone. The classic fusion algorithm is Reciprocal Rank Fusion (RRF), introduced by Cormack et al. RRF ignores raw scores, which live on incomparable scales, and instead uses ranks: each document earns a contribution of 1 / (k + rank) from every retriever, with k typically 60. Summing contributions produces a single robust ranking with no score normalization required.
A typical hybrid pipeline: run BM25 and vector search in parallel, take the top N from each, apply RRF to merge them, optionally rerank the fused list with a cross-encoder, then pass the final top-k chunks to the language model as context.`,
  },
  {
    filename: 'bm25-explained.txt',
    content: `BM25, or Best Matching 25, is a ranking function used by search engines. It improves on plain term-frequency scoring by adding two ideas: term frequency saturation, so that repeating a word ten times is not ten times better than once, and document length normalization, so long documents are not unfairly favored.
The BM25 score for a query term combines inverse document frequency, which rewards rare terms, with a saturated term frequency component. Two parameters tune behavior: k1 controls how quickly term frequency saturates, and b controls how strongly document length is normalized. Common defaults are k1 around 1.2-1.5 and b around 0.75.
BM25 is a bag-of-words model: it ignores word order and semantics. Two documents with the same words in different orders score identically. This makes it exact but brittle, which is exactly why pairing it with dense embeddings in a hybrid system works so well.`,
  },
  {
    filename: 'dense-vs-sparse.md',
    content: `Sparse representations such as BM25 assign one dimension per vocabulary term; most entries are zero. Dense representations assign a few hundred learned dimensions to every text, capturing meaning rather than surface form.
Sparse retrieval excels at exact match and is highly interpretable: you can see which terms drove the score. Dense retrieval excels at semantic match and generalization but is a black box and can retrieve plausible-looking but wrong passages, especially for rare identifiers.
Production search systems increasingly use both, then fuse. Cloudflare Vectorize, Qdrant, and Elasticsearch all support hybrid queries. The fusion step is what turns two noisy rankings into one strong one, and Reciprocal Rank Fusion is the most widely used, dependency-free fusion method.`,
  },
];

async function seedIfEmpty(env) {
  const row = await env.DB.prepare('SELECT COUNT(*) AS n FROM documents').first();
  if (row && row.n > 0) return;
  for (const doc of SAMPLE_DOCS) await ingestDocument(env, doc.filename, doc.content);
}

async function ingestDocument(env, filename, content) {
  const docId = uid();
  const chunks = chunkText(content, 800, 120);
  if (!chunks.length) throw new Error('Document produced no chunks');
  const vectors = await embed(env, chunks);
  const now = Date.now();
  const stmts = [env.DB.prepare('INSERT INTO documents (id, filename, content, created_at) VALUES (?,?,?,?)').bind(docId, filename, content, now)];
  chunks.forEach((c, i) => {
    stmts.push(env.DB.prepare('INSERT INTO chunks (id, doc_id, filename, chunk_index, content, embedding, created_at) VALUES (?,?,?,?,?,?,?)')
      .bind(uid(), docId, filename, i, c, JSON.stringify(vectors[i]), now));
  });
  await env.DB.batch(stmts);
  return { docId, chunks: chunks.length };
}

function tokenize(text) {
  return (text || '').toLowerCase().match(/[a-z0-9]+/g) || [];
}

// BM25 over the in-DB chunk corpus
function bm25(query, docs, { k1 = 1.5, b = 0.75 } = {}) {
  const qTerms = [...new Set(tokenize(query))];
  const N = docs.length;
  if (!N) return [];
  const df = {};
  const docTerms = docs.map((d) => tokenize(d.content));
  const avgdl = docTerms.reduce((s, t) => s + t.length, 0) / N || 1;
  docTerms.forEach((t) => { [...new Set(t)].forEach((w) => (df[w] = (df[w] || 0) + 1)); });
  const scores = docs.map((d, i) => {
    const tf = {};
    docTerms[i].forEach((w) => (tf[w] = (tf[w] || 0) + 1));
    const dl = docTerms[i].length;
    let s = 0;
    for (const q of qTerms) {
      if (!tf[q]) continue;
      const idf = Math.log(1 + (N - (df[q] || 0) + 0.5) / ((df[q] || 0) + 0.5));
      s += idf * ((tf[q] * (k1 + 1)) / (tf[q] + k1 * (1 - b + b * (dl / avgdl))));
    }
    return { idx: i, score: s };
  });
  return scores.filter((x) => x.score > 0).sort((a, b) => b.score - a.score);
}

function rrf(lists, k = 60) {
  const agg = {};
  for (const list of lists) {
    list.forEach((item, rank) => {
      const id = item.id;
      agg[id] = agg[id] || { id, item, score: 0 };
      agg[id].score += 1 / (k + rank + 1);
    });
  }
  return Object.values(agg).sort((a, b) => b.score - a.score);
}

async function hybridRetrieve(env, query, topK, opts = {}) {
  const alpha = typeof opts.alpha === 'number' ? opts.alpha : 0.5; // 0=bm25, 1=dense
  const candN = opts.candidates || 30;
  const rows = (await env.DB.prepare('SELECT id, filename, chunk_index, content, embedding FROM chunks LIMIT 5000').all()).results || [];
  if (!rows.length) return { results: [], bm25_count: 0, dense_count: 0 };

  // lexical
  const lex = bm25(query, rows).slice(0, candN).map((x) => ({ id: rows[x.idx].id, bm25: x.score }));
  // dense
  const qv = (await embed(env, [query]))[0];
  const denseSorted = rows.map((r) => ({ id: r.id, sim: cosine(qv, JSON.parse(r.embedding)) }))
    .sort((a, b) => b.sim - a.sim).slice(0, candN);
  const dense = denseSorted.map((x) => ({ id: x.id, dense: x.sim }));

  const fused = rrf([lex, dense]);
  const meta = {};
  lex.forEach((x, i) => (meta[x.id] = { ...meta[x.id], bm25: x.bm25, bm25_rank: i + 1 }));
  dense.forEach((x, i) => (meta[x.id] = { ...meta[x.id], dense: x.dense, dense_rank: i + 1 }));
  const rowMap = Object.fromEntries(rows.map((r) => [r.id, r]));

  let results = fused.slice(0, topK).map((f) => {
    const r = rowMap[f.id];
    return {
      chunk_id: r.id, filename: r.filename, chunk_index: r.chunk_index, content: r.content,
      rrf_score: Number(f.score.toFixed(6)),
      bm25_score: Number((meta[f.id]?.bm25 || 0).toFixed(4)),
      dense_score: Number((meta[f.id]?.dense || 0).toFixed(4)),
      bm25_rank: meta[f.id]?.bm25_rank || null,
      dense_rank: meta[f.id]?.dense_rank || null,
    };
  });
  // alpha weighting: blend by re-sorting on weighted normalized signals
  results = results.map((r) => ({ ...r, fused_weighted: Number((alpha * r.dense_rank_score + (1 - alpha) * r.bm25_rank_score).toFixed(6)) }));
  return { results, bm25_count: lex.length, dense_count: dense.length };
}

app.get('/api/health', (c) => c.json({ status: 'healthy', service: 'hybrid-rag', ts: Date.now() }));

app.get('/api/stats', async (c) => {
  await ensureSchema(c.env);
  const d = await c.env.DB.prepare('SELECT COUNT(*) AS n FROM documents').first();
  const ch = await c.env.DB.prepare('SELECT COUNT(*) AS n FROM chunks').first();
  return c.json({ documents: d?.n || 0, chunks: ch?.n || 0, fusion: 'RRF(k=60)', llm_model: c.env.LLM_MODEL || 'openai/gpt-oss-120b' });
});

app.get('/api/documents', async (c) => {
  await ensureSchema(c.env);
  try { await seedIfEmpty(c.env); } catch {}
  const rows = (await c.env.DB.prepare('SELECT d.id, d.filename, d.created_at, COUNT(ch.id) AS chunks FROM documents d LEFT JOIN chunks ch ON ch.doc_id=d.id GROUP BY d.id ORDER BY d.created_at DESC').all()).results || [];
  return c.json({ documents: rows, count: rows.length });
});

app.post('/api/documents', async (c) => {
  await ensureSchema(c.env);
  let body; try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const filename = (body.filename || 'untitled.txt').toString().slice(0, 200);
  const text = (body.text || '').toString();
  if (!text.trim()) return c.json({ error: 'Field "text" is required' }, 400);
  try { const r = await ingestDocument(c.env, filename, text); return c.json({ document_id: r.docId, filename, chunk_count: r.chunks }, 201); }
  catch (e) { return c.json({ error: 'Ingestion failed', detail: String(e.message || e) }, 500); }
});

app.delete('/api/documents/:id', async (c) => {
  await ensureSchema(c.env);
  const id = c.req.param('id');
  await c.env.DB.batch([c.env.DB.prepare('DELETE FROM chunks WHERE doc_id=?').bind(id), c.env.DB.prepare('DELETE FROM documents WHERE id=?').bind(id)]);
  return c.json({ deleted: id });
});

app.post('/api/search', async (c) => {
  await ensureSchema(c.env);
  let body; try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const query = (body.query || '').toString();
  if (!query.trim()) return c.json({ error: 'Field "query" is required' }, 400);
  try { await seedIfEmpty(c.env); } catch {}
  const topK = Math.min(Math.max(parseInt(body.top_k) || 5, 1), 20);
  const t0 = Date.now();
  const { results, bm25_count, dense_count } = await hybridRetrieve(c.env, query, topK, { alpha: body.alpha, candidates: body.candidates });
  return c.json({ query, mode: 'hybrid-rrf', results, bm25_candidates: bm25_count, dense_candidates: dense_count, retrieval_time_ms: Date.now() - t0 });
});

app.post('/api/query', async (c) => {
  await ensureSchema(c.env);
  let body; try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const query = (body.query || '').toString();
  if (!query.trim()) return c.json({ error: 'Field "query" is required' }, 400);
  try { await seedIfEmpty(c.env); } catch {}
  const topK = Math.min(Math.max(parseInt(body.top_k) || 4, 1), 12);
  const t0 = Date.now();
  const { results } = await hybridRetrieve(c.env, query, topK, { alpha: body.alpha });
  const tR = Date.now();
  if (!results.length) return c.json({ query, answer: "I don't have enough information in the provided documents to answer this question.", citations: [], model: 'none', retrieval_time_ms: tR - t0, generation_time_ms: 0, total_time_ms: Date.now() - t0 });
  const context = results.map((ch, i) => `[Source ${i + 1}] ${ch.filename} (chunk ${ch.chunk_index})\n${ch.content}`).join('\n\n---\n\n');
  const answer = await chat(c.env, [
    { role: 'system', content: 'You are a precise RAG assistant using hybrid retrieval. Answer ONLY from context. Cite [Source N]. Never invent facts.' },
    { role: 'user', content: `Context:\n${context}\n\nQuestion: ${query}\n\nAnswer with citations:` },
  ]);
  const tG = Date.now();
  return c.json({
    query, answer,
    citations: results.map((ch, i) => ({ n: i + 1, chunk_id: ch.chunk_id, filename: ch.filename, chunk_index: ch.chunk_index, rrf_score: ch.rrf_score, bm25_score: ch.bm25_score, dense_score: ch.dense_score, bm25_rank: ch.bm25_rank, dense_rank: ch.dense_rank, content: ch.content.slice(0, 300) })),
    model: c.env.LLM_MODEL || 'openai/gpt-oss-120b',
    retrieval_time_ms: tR - t0, generation_time_ms: tG - tR, total_time_ms: tG - t0,
  });
});

app.get('/', (c) => c.html(UI));

const UI = `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hybrid RAG — BM25 + Dense + RRF</title>
<script src="https://cdn.tailwindcss.com"></script><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css">
<style>body{background:#0b1020}.glass{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)}</style></head>
<body class="min-h-screen text-slate-100">
<header class="border-b border-white/10"><div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
<div class="flex items-center gap-3"><div class="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 grid place-items-center"><i class="fas fa-code-merge"></i></div>
<div><h1 class="font-bold text-lg leading-tight">Hybrid RAG</h1><p class="text-xs text-slate-400">BM25 + Dense vectors fused with Reciprocal Rank Fusion</p></div></div>
<div id="stats" class="text-xs text-slate-400"></div></div></header>
<main class="max-w-6xl mx-auto px-4 py-6 grid lg:grid-cols-2 gap-6">
<section class="glass rounded-2xl p-5"><h2 class="font-semibold mb-3"><i class="fas fa-file-arrow-up mr-2 text-emerald-400"></i>Knowledge base</h2>
<input id="fname" placeholder="filename.txt" class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm mb-2"/>
<textarea id="dtext" rows="4" placeholder="Paste text…" class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm mb-2"></textarea>
<div class="flex flex-wrap gap-2 mb-4"><button onclick="ingest()" class="bg-emerald-600 hover:bg-emerald-500 rounded-lg px-4 py-2 text-sm"><i class="fas fa-plus mr-1"></i>Ingest</button>
<label class="cursor-pointer bg-white/10 hover:bg-white/20 rounded-lg px-3 py-2 text-sm"><i class="fas fa-paperclip mr-1"></i>Upload<input type="file" accept=".txt,.md,.markdown" class="hidden" onchange="loadFile(event)"></label>
<button onclick="seed()" class="bg-white/5 hover:bg-white/10 rounded-lg px-3 py-2 text-sm">Load samples</button></div>
<h3 class="text-xs uppercase tracking-wide text-slate-400 mb-2">Documents (<span id="dcount">0</span>)</h3>
<ul id="docs" class="space-y-1 max-h-52 overflow-auto text-sm"></ul></section>
<section class="glass rounded-2xl p-5"><h2 class="font-semibold mb-3"><i class="fas fa-magnifying-glass mr-2 text-cyan-400"></i>Hybrid query</h2>
<div class="flex gap-2 mb-2"><input id="q" placeholder="e.g. what is BM25?" class="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm" onkeydown="if(event.key==='Enter')ask()"/>
<button onclick="ask()" class="bg-cyan-600 hover:bg-cyan-500 rounded-lg px-4 py-2 text-sm"><i class="fas fa-paper-plane"></i></button></div>
<div class="flex items-center gap-3 text-xs text-slate-400 mb-3"><span>BM25</span><input id="alpha" type="range" min="0" max="1" step="0.1" value="0.5" class="flex-1" oninput="document.getElementById('av').textContent=this.value"><span>Dense</span><span id="av" class="text-cyan-300">0.5</span></div>
<div id="meta" class="text-xs text-slate-400 mb-2"></div>
<div id="answer" class="text-sm leading-relaxed whitespace-pre-wrap min-h-[80px]"></div>
<div id="cites" class="mt-4 space-y-2"></div></section></main>
<p class="text-center text-xs text-slate-500 py-6">BM25 + Workers AI bge-small · RRF fusion · Groq LLM · D1 · <span id="health">…</span></p>
<script>
const api=(p,o)=>fetch(p,o).then(r=>r.json());
async function refresh(){try{const s=await api('/api/stats');document.getElementById('stats').innerHTML='<i class="fas fa-database mr-1"></i>'+s.documents+' docs · '+s.chunks+' chunks · '+s.fusion;
const d=await api('/api/documents');document.getElementById('dcount').textContent=d.count;
document.getElementById('docs').innerHTML=d.documents.map(x=>'<li class="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2"><span class="truncate"><i class="fas fa-file-lines mr-2 text-slate-400"></i>'+x.filename+' <span class="text-slate-500">('+x.chunks+')</span></span><button onclick="del(\\''+x.id+'\\')" class="text-slate-400 hover:text-rose-400"><i class="fas fa-trash"></i></button></li>').join('')||'<li class="text-slate-500 text-xs px-3 py-2">No documents.</li>';}catch(e){}}
async function ingest(){const filename=document.getElementById('fname').value||('doc-'+Date.now()+'.txt');const text=document.getElementById('dtext').value;if(!text.trim()){alert('Enter text');return;}
const r=await fetch('/api/documents',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename,text})});const j=await r.json();if(j.error){alert(j.error);return;}document.getElementById('dtext').value='';document.getElementById('fname').value='';refresh();}
function loadFile(e){const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{document.getElementById('dtext').value=r.result;document.getElementById('fname').value=f.name;};r.readAsText(f);}
async function del(id){await fetch('/api/documents/'+id,{method:'DELETE'});refresh();}
async function ask(){const q=document.getElementById('q').value.trim();if(!q)return;const alpha=parseFloat(document.getElementById('alpha').value);
document.getElementById('answer').innerHTML='<i class="fas fa-spinner fa-spin text-cyan-400"></i>';document.getElementById('cites').innerHTML='';document.getElementById('meta').textContent='';
try{const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,alpha})});const j=await r.json();if(j.error){document.getElementById('answer').textContent='Error: '+j.error;return;}
document.getElementById('answer').textContent=j.answer;
document.getElementById('meta').innerHTML='<i class="fas fa-bolt mr-1 text-amber-400"></i>'+j.model+' · retrieve '+j.retrieval_time_ms+'ms · generate '+j.generation_time_ms+'ms';
document.getElementById('cites').innerHTML=(j.citations||[]).map(c=>'<details class="bg-white/5 rounded-lg px-3 py-2 text-sm"><summary class="cursor-pointer"><span class="text-emerald-300 font-medium">[Source '+c.n+'] '+c.filename+'</span> <span class="text-slate-500">· RRF '+c.rrf_score+' · BM25# '+c.bm25_rank+' · Dense# '+c.dense_rank+'</span></summary><p class="mt-2 text-slate-300 text-xs">'+c.content.replace(/</g,'&lt;')+'…</p></details>').join('');
}catch(e){document.getElementById('answer').textContent='Failed: '+e;}}
async function seed(){document.getElementById('answer').innerHTML='<span class="text-slate-400 text-xs">Seeding…</span>';await api('/api/documents');refresh();document.getElementById('answer').innerHTML='';}
api('/api/health').then(h=>document.getElementById('health').innerHTML='<span class="text-emerald-400">API '+h.status+'</span>');refresh();
</script></body></html>`;

export default app;
