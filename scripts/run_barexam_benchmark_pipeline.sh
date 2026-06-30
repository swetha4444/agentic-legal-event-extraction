#!/usr/bin/env bash
# Legal-RAG benchmark (reglab/barexam_qa), QA-aligned:
#   build docs+qa from test.csv gold passages
#   -> facts.jsonl -> chunks.jsonl (one doc per gold passage, evaluable)
#
# Run from project root (GPU allocation recommended for step 1):
#   ./scripts/run_barexam_benchmark_pipeline.sh
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

OUT_DIR="data/outputs/legal_rag_benchmark"
NUM_QA="${NUM_QA:-100}"
QA_SPLIT="${QA_SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-data/models/lexlm_facts_us_courtlistener}"
MAX_LLM_CHUNK_SENTS="${MAX_LLM_CHUNK_SENTS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-40}"

echo "==> Step 0/3: build QA-aligned docs + qa.jsonl (gold_idx passages)"
python scripts/data/build_barexam_qa_aligned.py \
    --num-qa "$NUM_QA" \
    --qa-split "$QA_SPLIT" \
    --output-dir "$OUT_DIR"

echo "==> Step 1/3: facts extraction (LexLM US classifier)"
python src/run_facts_agent.py --agent bert \
    --checkpoint "$CHECKPOINT" \
    --input "$OUT_DIR/docs.jsonl" \
    --output "$OUT_DIR/facts.jsonl"

echo "==> Step 1b: chunk_text = extracted_facts or fallback to document_text (mbe passages)"
python - <<PY
import json, os
out_dir = "${OUT_DIR}"
docs = {json.loads(l)["case_id"]: json.loads(l) for l in open(f"{out_dir}/docs.jsonl")}
with open(f"{out_dir}/facts.jsonl") as fin, open(f"{out_dir}/facts_for_chunking.jsonl", "w") as fout:
    n = 0
    for line in fin:
        rec = json.loads(line)
        cid = rec.get("case_id") or ""
        doc_text = (docs.get(cid, {}).get("document_text") or "").strip()
        facts = (rec.get("extracted_facts") or "").strip()
        rec["chunk_text"] = facts if facts else doc_text
        fout.write(json.dumps(rec) + "\n")
        n += 1
print(f"Wrote {out_dir}/facts_for_chunking.jsonl ({n} records)")
PY

echo "==> Step 2/3: LLM chunking (meaningful chunks; 1 LLM call/doc via --dry-run)"
python -m src.run_events_agent \
    --input "$OUT_DIR/facts_for_chunking.jsonl" \
    --facts-field chunk_text \
    --chunk-strategy llm \
    --chunk-size "$CHUNK_SIZE" \
    --max-sentences-for-llm-chunking "$MAX_LLM_CHUNK_SENTS" \
    --chunks-only \
    --dry-run \
    --output "$OUT_DIR/chunks.jsonl"

echo "==> Done. Outputs in $OUT_DIR :"
ls -la "$OUT_DIR"
