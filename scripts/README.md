# Scripts

Run from **project root**. Python: `PYTHONPATH=src .venv/bin/python scripts/...`

## Layout

```
scripts/
├── setup.sh              # Venv, deps, data dirs
├── data/compile_marro_uk_to_json.py # Compile UK MARRO docs into title/facts/nonfacts JSON
├── run_facts_agent.sh    # Facts extraction: --agent llm | bert
├── run_facts_legal_bert.sh
├── run_courtlistener_pipeline.sh
├── verify.sh
├── train/
│   ├── train_legal_bert_facts.py   # Core trainer (LegalBERT/LexLM)
│   ├── run_legal_bert_train.sh
│   ├── run_legal_bert_train_holdout.sh
│   ├── run_lexlm_train.sh
│   └── run_lexlm_train_holdout.sh
└── eval/
    ├── eval_legal_bert_on_marro_dir.py   # Single file or dir; micro/macro F1
    ├── compare_bert_llm_on_marro.py       # BERT vs LLM on same doc(s)
    └── eval_fact_modes_on_marro_json.py   # classifier | llm | hybrid on compiled MARRO JSON
```

## Data

- Training JSONL: `data/labeled/marro_fact_nonfact.jsonl` (FAC=1, rest=0). Build from MARRO with a convert script (`--exclude <file>` for hold-out).
- Hold-out: `data/labeled/marro_fact_nonfact_holdout.jsonl` (e.g. exclude `uksc-2011-0183.txt`).

## Quick ref

| What | Command |
|------|---------|
| Train LegalBERT | `./scripts/train/run_legal_bert_train.sh` |
| Train LexLM | `./scripts/train/run_lexlm_train.sh` |
| Eval one doc | `PYTHONPATH=src .venv/bin/python scripts/eval/eval_legal_bert_on_marro_dir.py --checkpoint <dir> --docs <path/to/doc.txt>` |
| Eval all UK-dataset | `--docs data/MARRO/UK-dataset` (add `--quiet` for aggregate only) |
| BERT vs LLM | `scripts/eval/compare_bert_llm_on_marro.py --bert <checkpoint> --doc <file> [--llm gpt4o]` |
| Compile UK MARRO to JSON | `python3 scripts/data/compile_marro_uk_to_json.py` |
| JSON eval (3 modes) | `PYTHONPATH=src .venv/bin/python scripts/eval/eval_fact_modes_on_marro_json.py --mode classifier --input-json data/processed_json/marro_uk_compiled.json` |
| JSON eval quick limit | `...eval_fact_modes_on_marro_json.py --mode <classifier|llm|hybrid> --max-docs 1 --max-sentences-per-doc 20` |

See **commands.txt** in project root for copy-paste.
