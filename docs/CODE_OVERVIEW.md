# Code overview: what each file does

This project has one pipeline: **cluster-first — fetch case data from CourtListener (only cases with court opinion), enrich with docket and parties by docket_id, write JSONL**. Below is what each file is for.

---

## Project root

| File | Purpose |
|------|--------|
| **README.md** | How to set up, run the pipeline, and where output goes. |
| **commands.txt** | Copy-paste commands: setup, salloc, run pipeline, options. |
| **requirements.txt** | Python deps: `pyyaml`, `requests` (no COLD/datasets). |
| **.gitignore** | Ignore venv, data files, cache, `.courtlistener_token`, etc. |

---

## `config/`

| File | Purpose |
|------|--------|
| **config.yaml** | Single place for pipeline options. **courtlistener:** `nature_of_suit`; optional `api_token` or `token_file`. **paths:** `courtlistener_output_file` (default JSONL path). Pipeline is cluster-first. CLI flags override these. |

---

## `src/` (Python)

| File | Purpose |
|------|--------|
| **src/__init__.py** | Package marker and `__version__`. No pipeline logic. |
| **src/data/__init__.py** | Re-exports `fetch_clusters`, `fetch_opinion_for_docket`, and `search_recap` from `courtlistener_client`. The pipeline uses only `fetch_clusters`. |
| **src/data/courtlistener_client.py** | **CourtListener API client.** All HTTP and response shaping lives here. |
| **src/run_courtlistener_pipeline.py** | **Pipeline entrypoint.** Reads config/CLI, calls the client, writes one JSON object per case to a JSONL file. |

---

## `src/data/courtlistener_client.py` (in detail)

This file talks to CourtListener’s REST API and returns plain Python dicts.

- **Constants**  
  `CLUSTERS_URL`, `DOCKETS_URL`, `PARTIES_URL`, `SEARCH_URL` — API bases. `APPELLATE_COURT_PREFIXES` — court ID prefixes (e.g. `ca*`) for district filtering (RECAP).

- **`_load_config()`**  
  Loads `config/config.yaml` and returns the parsed dict. Used for token and defaults.

- **`_is_district_court(court_id)`**  
  Returns True if `court_id` does not look like an appellate/bankruptcy court (e.g. not `ca*`, `uscfc`, etc.). Used only for RECAP to skip non-district results.

- **`_get_token(api_token=None)`**  
  Resolves the API token in order: argument → `COURTLISTENER_API_TOKEN` env → `courtlistener.api_token` in config → file at `courtlistener.token_file`. Raises if none set. Used by both `fetch_clusters` and `search_recap`.

- **`_fetch_opinion_text(opinion_url, headers)`**  
  GETs a single opinion URL (from a cluster’s `sub_opinions`). Asks only for `plain_text` and `html_with_citations`. Returns `plain_text` if present, otherwise strips HTML from `html_with_citations` and returns that. Used when `include_opinion_text=True` for clusters and by `fetch_opinion_for_docket`.

- **`fetch_opinion_for_docket(docket_id, api_token=None)`**  
  Optional helper: GETs clusters by docket_id and returns `(opinion_text, cluster_id)`. Not used by the current cluster-first pipeline.

- **`_fetch_docket(docket_id, headers)`**  
  GETs one docket by ID; returns metadata (docket_number, court_id, date_terminated, assigned_to_str, referred_to_str, cause). No parties in response.

- **`_fetch_parties_for_docket(docket_id, headers)`**  
  GETs parties for a docket (`/api/rest/v4/parties/?docket={docket_id}`). Returns list of party names; empty if 403 or no access.

- **`_docket_id_from_cluster(item)`**  
  Extracts numeric docket_id from cluster item (handles `docket_id` or `docket` URL).

- **`fetch_clusters(..., include_opinion_text=False, include_docket_parties=False)`**  
  **Clusters API (case law).** GETs `/api/rest/v4/clusters/` with pagination. For each cluster: builds a normalized record (`_normalize_cluster_result`) with `parties` from caption. If `include_opinion_text=True`, fetches the first `sub_opinion` and adds `opinion_text`. If `include_docket_parties=True`, fetches docket and parties by `docket_id` and merges into the record (parties from API override caption). Yields one dict per cluster.

- **`_parties_from_caption(case_name, case_name_full)`**  
  Parses caption for “X v. Y” (or “ v. ” / “ v ”) and returns a list of one or two party names. Used by `_normalize_cluster_result` so cluster records have a `parties` field.

- **`_normalize_cluster_result(item)`**  
  Maps one cluster API response to a flat dict: `cluster_id`, `docket_id`, `case_name`, `case_name_full`, `date_filed`, `nature_of_suit`, `absolute_url`, `slug`, `parties` (from caption), and empty/placeholder fields so the shape matches RECAP output where possible.

- **`search_recap(...)`**  
  **RECAP Search API.** GETs `/api/rest/v4/search/?type=r` with query params (`q`, `nature_of_suit`, `filed_before`, etc.). Paginates via `next`. For each result, skips if `district_only` and court is appellate; otherwise yields `_normalize_recap_result(item)`.

- **`_normalize_recap_result(item)`**  
  Maps one RECAP search hit to a flat dict with all useful fields: `docket_id`, `case_name`, `case_name_full`, `court_id`, `court`, `court_citation_string`, `docket_number`, `date_filed`, `date_terminated`, `date_argued`, `nature_of_suit`, `cause`, `documents` (description, snippet, entry_date_filed, etc.), `absolute_url` (from `docket_absolute_url`), **`parties`**, `party_id`, **`attorneys`**, `attorney_id`, **`firms`**, `firm_id`, **`assigned_to`**, `assigned_to_id`, **`referred_to`**, `referred_to_id`, **`jurisdiction_type`**, **`jury_demand`**, `pacer_case_id`.

So: **courtlistener_client** = “get token; fetch clusters (optionally with opinion text and docket/parties by docket_id); normalize to a common record shape. Also provides search_recap and fetch_opinion_for_docket for other use.”

---

## `src/run_courtlistener_pipeline.py` (in detail)

This is the **orchestrator**: config + CLI → client → JSONL.

- **`_load_config_file(path)`**  
  Loads a YAML config file and returns the dict (or `{}` if missing).

- **`_pick_document_text(rec, prefer_complaint)`**  
  Present but unused in the cluster-first pipeline. Would return a RECAP document snippet from a record's `documents` list if we used RECAP.

- **`run(...)`**  
  - Loads config (default or `--config`), merges with CLI (e.g. `--max-cases`, `--out`).  
  - Pipeline is **cluster-first**: runs `fetch_clusters(include_opinion_text=True, include_docket_parties=True)`. Every record has a court opinion; we enrich with docket metadata and parties by `docket_id`. `document_text` = opinion text.  
  - Writes one JSON line per record with: `case_id`, `cluster_id`, `docket_id`, `docket_number`, `court_id`, `court`, `court_citation_string`, `case_name`, `case_name_full`, `date_filed`, `date_terminated`, `date_argued`, `nature_of_suit`, `cause`, `document_text` (court opinion), `absolute_url`, **`parties`** (from Parties API or caption), `party_id`, `attorneys`, `attorney_id`, `firms`, `firm_id`, **`assigned_to`**, `assigned_to_id`, **`referred_to`**, `referred_to_id`, `jurisdiction_type`, `jury_demand`, `pacer_case_id`, `documents`. **`opinion_text`** is also written.  
  - Prints how many lines were written.

- **`main()`**  
  Defines argparse (--config, --max-cases, --nature-of-suit, --out), parses args, and calls `run(...)`.

So: **run_courtlistener_pipeline** = “config + CLI → clusters (opinion) + docket/parties by docket_id → one JSONL file with full metadata and parties.”

---

## `scripts/`

| File | Purpose |
|------|--------|
| **setup.sh** | Creates `.venv` if missing, activates it, upgrades pip, installs `requirements.txt`, creates `data/raw`, `data/processed`, `data/outputs`. Run once from project root. |
| **run_courtlistener_pipeline.sh** | Ensures you’re in project root, activates `.venv` if present, loads token from `.courtlistener_token` if `COURTLISTENER_API_TOKEN` is not set, then runs `python -m src.run_courtlistener_pipeline` with all passed arguments. |
| **validate_config.sh** | Quick check: imports `fetch_clusters` from `src.data.courtlistener_client`. Confirms the CourtListener pipeline code loads; does not call the API. |

---

## `data/`

| Path | Purpose |
|------|--------|
| **data/raw/** | Reserved for raw inputs; only `.gitkeep` is tracked. |
| **data/processed/** | Default location for the pipeline output JSONL (e.g. `courtlistener_recap.jsonl`). |
| **data/outputs/** | Reserved for later pipeline outputs; only `.gitkeep` is tracked. |
| **data/README.md** | Short description of these directories. |

---

## `docs/`

| File | Purpose |
|------|--------|
| **DATA_SOURCES.md** | Describes Free Law Project, CourtListener, and the cluster-first pipeline (docket/parties by docket_id). No code. |
| **CODE_OVERVIEW.md** | This file: what each file in the repo does. |

---

## End-to-end flow

1. You set a token (env or `.courtlistener_token` or config).
2. You run `./scripts/run_courtlistener_pipeline.sh` (or `python -m src.run_courtlistener_pipeline` with options).
3. **run_courtlistener_pipeline** reads config and CLI, then:
   - **Cluster-first:** `fetch_clusters(include_opinion_text=True, include_docket_parties=True)` → each record is a cluster with court opinion; we enrich with docket metadata and parties by `docket_id`.
4. Each record is written as one JSON line to the chosen output file (e.g. `data/processed/courtlistener_recap.jsonl`) with the full schema (parties, docket_number, assigned_to, `document_text` = opinion, etc.).

All “what to request” (URLs, params, district filter) is in **courtlistener_client**; all “what to write and where” is in **run_courtlistener_pipeline**. See **docs/DATA_SOURCES.md** for source summary and output fields.
