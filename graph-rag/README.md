# Graph RAG

This folder contains a self-contained graph-native RAG prototype for the legal event graphs produced elsewhere in the repository.

Write boundary:
- All code and local artifacts for this prototype live inside this `graph-rag` folder.
- The code may read graph JSONL files from elsewhere in the repository.
- The code must not modify anything outside this folder.

Current components:
- `graphrag.loader`: loads merged graph JSONL and optional chunk graph JSONL.
- `graphrag.store`: stores graph data in a local SQLite database with FTS5 indexes.
- `graphrag.retrieval`: hybrid graph retrieval over event nodes, supporting fact nodes, entities, and edges.
- `graphrag.prompting`: builds graph-native prompts for downstream LLM answering.
- `graphrag.llm`: optional OpenAI-compatible answer generation.
- `graphrag.cli`: CLI entrypoint.

## Layout

```text
graph-rag/
  graphrag/
    cli.py
    llm.py
    loader.py
    models.py
    prompting.py
    retrieval.py
    store.py
  .artifacts/
    graphrag.sqlite   # created locally by indexing commands
```

## What gets indexed

From each merged graph row:
- document metadata
- entities
- events
- event participants
- temporal edges
- causal edges
- event evidence and source refs

If a matching chunk-graph JSONL is provided, the index also stores:
- chunk text
- chunk theme
- sentence ranges

Supporting facts are treated as first-class graph evidence when:
- `event_type == "SUPPORTING_FACT"`, or
- the event has no participants and no temporal/causal edges

## Quick start

Index a merged graph file and optional chunk graph file into a local database under this folder:

```bash
cd /work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/graph-rag
PYTHONDONTWRITEBYTECODE=1 python -m graphrag.cli index \
  --merged-jsonl ../data/outputs/hybrid_merge_local_qwen/mask_then_hybrid_merged_20260407_182232.jsonl \
  --chunk-jsonl ../data/outputs/chunk_event_graphs_rebuilt_1-5.jsonl \
  --db .artifacts/graphrag.sqlite
```

Preview retrieval:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m graphrag.cli query \
  --db .artifacts/graphrag.sqlite \
  --question "Why was the plaintiff terminated and what led to it?" \
  --top-k 8
```

Preview the prompt that would be sent to an LLM:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m graphrag.cli prompt \
  --db .artifacts/graphrag.sqlite \
  --question "Why was the plaintiff terminated and what led to it?" \
  --top-k 8
```

Optional answer generation with an OpenAI-compatible API:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m graphrag.cli answer \
  --db .artifacts/graphrag.sqlite \
  --question "Why was the plaintiff terminated and what led to it?" \
  --model gpt-4o-mini
```

`graph-rag` now follows the repo's existing config pattern:
- loads `<repo>/.env` when `python-dotenv` is installed
- prefers `AGENT_API_KEY`
- falls back to `OPENAI_API_KEY`
- reads default `llm.api_base` and `llm.model` from `<repo>/config/config.yaml`

## Retrieval design

The current retriever is graph-first:

1. Seed search over event/fact text, chunk text, and entity names using local FTS.
2. Seed scoring with lexical rank, token overlap, and supporting-fact awareness.
3. Graph expansion over:
   - temporal and causal edges
   - participant/entity overlap
   - chunk co-reference via `source_refs`
4. Final reranking over transparent features:
   - lexical score
   - graph proximity to seeds
   - entity overlap
   - temporal match
   - evidence richness

The scoring weights are intentionally easy to change in code while we evaluate the system.
