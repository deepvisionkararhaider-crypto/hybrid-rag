#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://localhost:8787}"
pass=0; fail=0
check(){ if [ "$1" = "$2" ]; then echo "PASS: $3"; pass=$((pass+1)); else echo "FAIL: $3 (got '$1' want '$2')"; fail=$((fail+1)); fi; }
check "$(curl -s "$BASE/api/health" | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")" "healthy" "health"
check "$(curl -s "$BASE/api/stats" | python3 -c "import sys,json;print(json.load(sys.stdin)['fusion'])")" "RRF(k=60)" "stats fusion"
code=$(curl -s -o /tmp/h.json -w "%{http_code}" -X POST "$BASE/api/documents" -H 'Content-Type: application/json' -d '{"filename":"t.txt","text":"BM25 ranks by term frequency and inverse document frequency. Dense retrieval embeds text and ranks by cosine similarity."}')
check "$code" "201" "ingest"
# lexical query (BM25 should fire on exact token)
r=$(curl -s -X POST "$BASE/api/search" -H 'Content-Type: application/json' -d '{"query":"BM25 inverse document frequency"}' | python3 -c "import sys,json;d=json.load(sys.stdin);print(len(d['results']))")
if [ "$r" -ge 1 ]; then echo "PASS: hybrid search has results"; pass=$((pass+1)); else echo "FAIL: hybrid search"; fail=$((fail+1)); fi
b=$(curl -s -X POST "$BASE/api/search" -H 'Content-Type: application/json' -d '{"query":"BM25 inverse document frequency"}' | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['bm25_candidates'])")
if [ "$b" -ge 1 ]; then echo "PASS: BM25 produced candidates"; pass=$((pass+1)); else echo "FAIL: BM25 candidates"; fail=$((fail+1)); fi
check "$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/search" -H 'Content-Type: application/json' -d '{"query":""}')" "400" "empty query rejected"
if curl -s -X POST "$BASE/api/query" -H 'Content-Type: application/json' -d '{"query":"what is BM25?"}' | grep -q '"answer"'; then echo "PASS: query returns answer"; pass=$((pass+1)); else echo "FAIL: query"; fail=$((fail+1)); fi
echo "---- $pass passed, $fail failed ----"; [ "$fail" -eq 0 ]
