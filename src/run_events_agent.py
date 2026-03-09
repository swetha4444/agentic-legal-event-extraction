"""
Run event extraction on JSONL: chunk first, then run events agent per chunk.
Output: one record per document with extracted entities and events (and optional chunk list).
"""
import argparse
import json
import os
import sys

if __name__ == "__main__":
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)

from agents import LLMEventsExtractor, BudgetExceededError
from agents.config_loader import get_llm_config
from processing.chunking import chunk_document, Chunk


def _merge_chunk_results(chunk_results: list[dict], chunk_ids: list[str]) -> dict:
    """
    Merge per-chunk {entities, events} into one dict. Preserve chunk_id on each event.
    Does not dedupe entities/events; caller can do that if needed.
    """
    all_entities: list = []
    all_events: list = []
    seen_entity_ids = set()
    for res, cid in zip(chunk_results, chunk_ids):
        for e in res.get("entities") or []:
            eid = e.get("entity_id") or ""
            if eid and eid not in seen_entity_ids:
                seen_entity_ids.add(eid)
                all_entities.append(e)
            elif not eid:
                all_entities.append(e)
        for ev in res.get("events") or []:
            ev = dict(ev)
            ev["_chunk_id"] = cid
            all_events.append(ev)
    return {"entities": all_entities, "events": all_events}


def main():
    parser = argparse.ArgumentParser(
        description="Event extraction: chunk document, then extract entities and events per chunk.",
    )
    parser.add_argument("--input", default="data/processed/courtlistener_recap.jsonl", help="Input JSONL")
    parser.add_argument("--output", default=None, help="Output JSONL (default: data/outputs/events_<model>.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Max documents")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    # Chunking (first step)
    parser.add_argument(
        "--chunk-strategy",
        choices=("char", "paragraph", "sentence", "section", "llm"),
        default="char",
        help="Chunking strategy: char/paragraph/sentence/section = fixed; llm = LLM splits doc into meaningful chunks (with overlap)",
    )
    parser.add_argument("--chunk-size", type=int, default=12_000, help="Max chars per chunk (char/section) or max sentences (sentence)")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap in chars (char strategy)")
    parser.add_argument("--overlap-sentences", type=int, default=5, help="Overlap in sentences (sentence strategy)")
    parser.add_argument("--text-field", default="document_text", help="JSON field containing document text")
    parser.add_argument("--facts-field", default=None, help="If set, use this field instead of document_text (e.g. extracted_facts, predicted_extracted_facts)")
    parser.add_argument("--sentences-field", default=None, help="If set, use this array of {id, text[, pred_label]} for chunking; each chunk is sent to LLM as [S1] ... [S2] ... so LLM can return sentence_ids in evidence")
    parser.add_argument("--fact-only", action="store_true", help="When using --sentences-field, keep only sentences with pred_label==1 (fact sentences)")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only; write one output record with mock extracted_entities/extracted_events (no LLM calls)")
    parser.add_argument("--chunks-only", action="store_true", help="Output file: only doc id + chunks (no extracted_entities, extracted_events, sentences, etc.)")
    parser.add_argument("--max-sentences-for-llm-chunking", type=int, default=120, help="When chunk-strategy=llm, skip LLM chunking if doc has more sentences (fall back to sentence strategy)")
    args = parser.parse_args()

    cfg = get_llm_config()
    model = args.model or cfg.get("model")
    output_path = args.output or os.path.join("data", "outputs", f"events_{model or 'default'}.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # In dry-run we still create extractor when chunk-strategy is llm so we can run real LLM chunking (only event extraction is mocked)
    extractor = None if (args.dry_run and args.chunk_strategy != "llm") else LLMEventsExtractor(model_name=model)
    text_key = args.facts_field or args.text_field
    count = 0

    with open(args.input) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if args.limit is not None and count >= args.limit:
                print(f"Stopped at limit={args.limit}")
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get(text_key) or rec.get("document_text") or ""
            use_sentences_array = args.sentences_field and isinstance(rec.get(args.sentences_field), list)
            if not use_sentences_array and not (text or "").strip():
                continue
            if use_sentences_array and not rec.get(args.sentences_field):
                continue

            case_name = rec.get("case_name") or rec.get("title") or ""
            docket_number = rec.get("docket_number") or ""

            # 1) Chunking: LLM meaningful chunks, or from record's sentences array, or from raw text
            prefix = rec.get("case_id") or f"doc_{count}"
            formatted: list = []
            sentence_ids: list = []  # 1-based IDs for each item in formatted

            if args.sentences_field and isinstance(rec.get(args.sentences_field), list):
                sent_list = rec[args.sentences_field]
                if args.fact_only:
                    sent_list = [s for s in sent_list if s.get("pred_label") == 1]
                for s in sent_list:
                    sid = s.get("id") or s.get("sentence_id") or len(formatted) + 1
                    stext = (s.get("text") or "").strip()
                    if stext:
                        formatted.append(f"[S{sid}] {stext}")
                        sentence_ids.append(sid)
            else:
                import re
                raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                for i, stext in enumerate(raw_sents, start=1):
                    formatted.append(f"[S{i}] {stext}")
                    sentence_ids.append(i)

            if not formatted:
                chunks = []
            elif args.chunk_strategy == "llm" and extractor and len(formatted) <= args.max_sentences_for_llm_chunking:
                # LLM splits doc into meaningful chunks (with overlap); runs even in dry-run (only event extraction is mocked)
                full_text = "\n".join(formatted)
                try:
                    spec_list = extractor.get_meaningful_chunks(full_text, case_name=case_name)
                except BudgetExceededError:
                    raise
                if not spec_list:
                    # Fallback: one chunk
                    chunks = [Chunk(chunk_id=f"{prefix}_chunk_0", text=full_text, start_char=0, end_char=len(full_text), start_sentence_id=sentence_ids[0] if sentence_ids else 0, end_sentence_id=sentence_ids[-1] if sentence_ids else 0)]
                else:
                    chunks = []
                    for spec in spec_list:
                        start_sid = spec.get("start_sentence_id") or 0
                        end_sid = spec.get("end_sentence_id") or 0
                        indices = [i for i in range(len(sentence_ids)) if start_sid <= sentence_ids[i] <= end_sid]
                        if not indices:
                            continue
                        chunk_text = "\n".join(formatted[i] for i in indices)
                        cid = spec.get("chunk_id") or f"{prefix}_chunk_{len(chunks)}"
                        chunks.append(Chunk(
                            chunk_id=cid,
                            text=chunk_text,
                            start_char=0,
                            end_char=len(chunk_text),
                            start_sentence_id=sentence_ids[indices[0]],
                            end_sentence_id=sentence_ids[indices[-1]],
                            metadata={"theme": spec.get("theme") or ""},
                        ))
            elif formatted and (args.chunk_strategy == "sentence" or (args.chunk_strategy == "llm" and (args.dry_run or not extractor or len(formatted) > args.max_sentences_for_llm_chunking))):
                # Fixed-size sentence chunks (sentence strategy, or llm fallback when dry-run / too long)
                max_sent = args.chunk_size
                step = max(1, max_sent - args.overlap_sentences)
                chunks = []
                start = 0
                idx = 0
                while start < len(formatted):
                    end = min(start + max_sent, len(formatted))
                    segment = formatted[start:end]
                    chunk_text = "\n".join(segment)
                    chunks.append(Chunk(
                        chunk_id=f"{prefix}_chunk_{idx}",
                        text=chunk_text,
                        start_char=0,
                        end_char=len(chunk_text),
                        start_sentence_id=sentence_ids[start] if sentence_ids else start,
                        end_sentence_id=sentence_ids[end - 1] if sentence_ids else end - 1,
                    ))
                    idx += 1
                    start += step
                    if end >= len(formatted):
                        break
            elif args.chunk_strategy == "sentence":
                import re
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                chunks = chunk_document(
                    text,
                    strategy="sentence",
                    sentences=sentences,
                    max_sentences_per_chunk=args.chunk_size,
                    overlap_sentences=args.overlap_sentences,
                    chunk_id_prefix=prefix,
                )
            else:
                chunks = chunk_document(
                    text,
                    strategy=args.chunk_strategy,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    chunk_id_prefix=prefix,
                )

            if not chunks:
                if args.chunks_only:
                    out = {"doc_id": rec.get("case_id") or rec.get("title") or rec.get("id") or f"doc_{count}", "num_chunks": 0, "chunks": []}
                else:
                    out = {k: v for k, v in rec.items() if k not in ("document_text", "extracted_facts")}
                    out["extracted_entities"] = []
                    out["extracted_events"] = []
                    out["chunks"] = []
                f_out.write(json.dumps(out, default=str) + "\n")
                count += 1
                continue

            # 2) Events extraction per chunk (or mock if --dry-run)
            chunk_results = []
            chunk_ids = [c.chunk_id for c in chunks]
            if args.dry_run:
                # Mock one entity and one event per chunk so output format is visible
                for c in chunks:
                    chunk_results.append({
                        "entities": [
                            {"entity_id": "E1", "kind": "PERSON", "canonical_role": "PLAINTIFF", "name": "[mock]", "aliases": []},
                        ],
                        "events": [
                            {
                                "event_id": "EV1",
                                "event_type": "InternalComplaintFiled",
                                "trigger": {"span_text": "[mock]"},
                                "participants": [{"entity_id": "E1", "role": "INITIATOR", "mention": "[mock]"}],
                                "time": {"kind": "UNKNOWN", "raw_span": "", "normalized": None},
                                "evidence": {"sentence_ids": [], "snippets": [{"text": "[dry-run: no LLM call]"}]},
                            }
                        ],
                    })
            else:
                try:
                    for c in chunks:
                        res = extractor.extract(
                            text=c.text,
                            case_name=case_name,
                            docket_number=docket_number,
                            chunk_id=c.chunk_id,
                        )
                        chunk_results.append(res)
                except BudgetExceededError as e:
                    print(f"Budget exceeded: {e}")
                    break

            # 3) Build chunk list for output
            chunks_out = [
                {
                    "chunk_id": c.chunk_id,
                    "start_sentence_id": getattr(c, "start_sentence_id", None),
                    "end_sentence_id": getattr(c, "end_sentence_id", None),
                    "theme": (c.metadata or {}).get("theme", ""),
                    "text": c.text[:2000] + ("..." if len(c.text) > 2000 else ""),
                }
                for c in chunks
            ]
            if args.chunks_only:
                out = {
                    "doc_id": rec.get("case_id") or rec.get("title") or rec.get("id") or f"doc_{count}",
                    "num_chunks": len(chunks),
                    "chunks": chunks_out,
                }
            else:
                merged = _merge_chunk_results(chunk_results, chunk_ids)
                out = {k: v for k, v in rec.items() if k not in ("document_text", "extracted_facts")}
                out["extracted_entities"] = merged["entities"]
                out["extracted_events"] = merged["events"]
                out["num_chunks"] = len(chunks)
                out["chunk_ids"] = chunk_ids
                out["chunks"] = chunks_out
            f_out.write(json.dumps(out, default=str) + "\n")
            count += 1
            if count % 5 == 0:
                print(f"Processed {count} cases...")

    print(f"Done. Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
