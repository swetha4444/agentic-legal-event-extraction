# Role-Aware Legal Event Extraction — Data Pipeline

Data pipeline from **CourtListener**: fetch case law (clusters + opinion text) or RECAP dockets; write to JSONL for downstream event extraction.

See **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** for Free Law Project, CourtListener APIs, and RECAP vs clusters.

## Setup

```bash
./scripts/setup.sh
source .venv/bin/activate
```

## Run pipeline

1. **Token:** [CourtListener](https://www.courtlistener.com/) API token (sign in, REST API access).
   ```bash
   export COURTLISTENER_API_TOKEN=your_token
   ```
   Or put the token in `.courtlistener_token` in the project root (script loads it; file is gitignored).

2. **Run (default: clusters API + opinion text):**
   ```bash
   ./scripts/run_courtlistener_pipeline.sh --config config/config.yaml
   ./scripts/run_courtlistener_pipeline.sh --max-cases 100 --out data/processed/courtlistener_recap.jsonl
   ```

**Options:** `--source clusters|recap`, `--max-cases N`, `--no-opinion-text` (skip fetching opinion text), `--nature-of-suit`, `--query`, `--filed-before`, `--out`. See **commands.txt**.

## Run LLM facts extraction

Uses **config/config.yaml** (model, max_calls, max_docs) and **.env** (`AGENT_API_KEY`). Install deps (including litellm) then run:

```bash
./scripts/setup.sh
source .venv/bin/activate
# Ensure .env has AGENT_API_KEY=your_key
./scripts/run_facts_agent.sh --input data/processed/courtlistener_recap.jsonl --output data/outputs/facts_extracted.jsonl --limit 5
```

Or without the wrapper:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/run_facts_agent.py --input data/processed/courtlistener_recap.jsonl --output data/outputs/facts_extracted.jsonl --limit 5
```

**Options:** `--input`, `--output`, `--limit` (overrides config `max_docs`). Budget is enforced from `config.yaml` (`llm.max_calls`).

## Config

**config/config.yaml:** `courtlistener.*`, `llm` (api_base, model, temperature, max_calls, max_docs), `paths.*`. API key: set in **.env** as `AGENT_API_KEY` (see [docs/CONFIG_AND_ENV.md](docs/CONFIG_AND_ENV.md)).

## Output

**data/processed/courtlistener_recap.jsonl** — one JSON per line: `case_id`, `cluster_id`, `docket_id`, `case_name`, `case_name_full`, `date_filed`, `nature_of_suit`, `document_text` (opinion or complaint snippet), `absolute_url`, etc.

## Structure

- **config/** — config.yaml (courtlistener, paths).
- **data/raw**, **data/processed** — inputs and outputs.
- **src/data/** — courtlistener_client.py (clusters + RECAP search API).
- **src/** — run_courtlistener_pipeline.py.
- **scripts/** — setup.sh, run_courtlistener_pipeline.sh, validate_config.sh.
- **commands.txt** — command reference.
