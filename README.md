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

## Config

**config/config.yaml:** `courtlistener.source` (clusters | recap), `courtlistener.include_opinion_text`, `courtlistener.nature_of_suit`, `courtlistener.query`, `paths.courtlistener_output_file`.

## Output

**data/processed/courtlistener_recap.jsonl** — one JSON per line: `case_id`, `cluster_id`, `docket_id`, `case_name`, `case_name_full`, `date_filed`, `nature_of_suit`, `document_text` (opinion or complaint snippet), `absolute_url`, etc.

## Structure

- **config/** — config.yaml (courtlistener, paths).
- **data/raw**, **data/processed** — inputs and outputs.
- **src/data/** — courtlistener_client.py (clusters + RECAP search API).
- **src/** — run_courtlistener_pipeline.py.
- **scripts/** — setup.sh, run_courtlistener_pipeline.sh, validate_config.sh.
- **commands.txt** — command reference.
