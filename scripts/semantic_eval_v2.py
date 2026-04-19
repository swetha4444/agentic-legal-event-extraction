"""
Semantic faithfulness v2: holistic narrative comparison.

Instead of breaking into claims and keyword-matching each one,
compare the two narratives as wholes:
  1. What key FACTS does the human mention? Does the prediction cover each?
  2. What does the prediction add that the human didn't mention?
  3. Overall: does the prediction send the same message?

Each fact is a simple (subject, action, detail) tuple extracted manually
from sentence structure, then matched flexibly against the full prediction.
"""

import json, csv, re, textwrap
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).resolve().parent.parent / "data" / "outputs"

# ── Load data ─────────────────────────────────────────────────────────────────
preds_by_doc = {}
with open(DATA / "two_stage_best_5docs_20260331_162752_predictions.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        preds_by_doc[doc["doc_id"]] = doc

human_by_chunk = {}
with open(DATA / "human_annotation_summaries.csv") as f:
    for row in csv.DictReader(f):
        human_by_chunk[(row["Document"], row["Chunk"])] = row


def events_to_narrative(chunk):
    """Convert structured event graph into prose."""
    graph = chunk.get("graph", {})
    events = graph.get("events", [])
    entities = {e["entity_id"]: e for e in graph.get("entities", [])}
    causal = graph.get("causal_edges", [])

    def ent_name(eid):
        ent = entities.get(eid, {})
        name = ent.get("canonical_name", "")
        if not name:
            mentions = ent.get("mentions", [])
            name = mentions[0].get("span_text", eid) if mentions else eid
        return name

    sentences = []
    for ev in events:
        snippet = ""
        for s in ev.get("evidence", {}).get("snippets", []):
            if s.get("text"):
                snippet = s["text"]
                break

        if snippet:
            desc = snippet.strip()
        else:
            verb = ev.get("main_verb", "")
            parts = [ent_name(p.get("entity_id", "")) for p in ev.get("participants", [])]
            desc = f"{' and '.join(parts)} {verb}" if parts else ev.get("event_type", "")

        t = ev.get("time", {})
        if t and t.get("raw_span") and t["raw_span"].lower() not in desc.lower():
            desc += f" ({t['raw_span']})"

        sentences.append(desc)

    causal_notes = []
    for ce in causal:
        src = ce.get("source", ce.get("cause", ""))
        tgt = ce.get("target", ce.get("effect", ""))
        rel = ce.get("relation", ce.get("type", ""))
        causal_notes.append(f"{src} → {tgt} ({rel})")

    return " ".join(sentences), causal_notes


# ── Flexible semantic matching ────────────────────────────────────────────────

def normalize(text):
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_fact_phrases(summary):
    """
    Pull out the meaningful fact phrases from a human summary.
    Split on natural breakpoints: periods, commas followed by subjects,
    semicolons, 'and' connecting independent clauses.
    """
    # First split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', summary.strip())

    facts = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 10:
            continue
        # Further split long compound sentences at natural boundaries
        # but keep each piece meaningful (>15 chars)
        subparts = re.split(r';\s*|(?:,\s*(?:and\s+)?(?:also|which|who|that|then|but)\s)', part)
        for sp in subparts:
            sp = sp.strip().rstrip('.,;')
            if len(sp) >= 15:
                facts.append(sp)
            elif sp:
                # Too short on its own — merge with previous if possible
                if facts:
                    facts[-1] = facts[-1] + ", " + sp
                else:
                    facts.append(sp)
    return facts


def fact_is_covered(fact, pred_text_norm, threshold=0.45):
    """
    Check if a fact from the human summary is semantically present
    in the prediction narrative. Uses flexible n-gram matching.
    """
    fact_norm = normalize(fact)
    fact_words = [w for w in fact_norm.split() if len(w) > 2]

    if not fact_words:
        return True, 1.0  # trivial fact

    # Remove very common words that don't carry meaning
    stops = {'the','and','was','were','that','this','with','from','has','had',
             'have','for','are','not','but','who','also','been','she','her',
             'his','him','they','them','its','being','into','about','after',
             'before','when','where','which','while','than','then','there',
             'what','will','would','could','should','does','did'}
    content_words = [w for w in fact_words if w not in stops]
    if not content_words:
        content_words = fact_words  # fallback

    # Check what fraction of content words appear in prediction
    found = sum(1 for w in content_words if w in pred_text_norm)
    word_coverage = found / len(content_words)

    # Also check for key bigrams (consecutive word pairs)
    bigrams = [f"{content_words[i]} {content_words[i+1]}"
               for i in range(len(content_words)-1)]
    bigram_hits = sum(1 for bg in bigrams if bg in pred_text_norm) if bigrams else 0
    bigram_coverage = bigram_hits / len(bigrams) if bigrams else 1.0

    # Combined score: words matter most, bigrams add confidence
    combined = 0.6 * word_coverage + 0.4 * bigram_coverage

    return combined >= threshold, round(combined, 3)


# ── Run evaluation ────────────────────────────────────────────────────────────
print("=" * 90)
print("NARRATIVE FAITHFULNESS: Does the prediction convey the same message?")
print("=" * 90)

all_results = []

for doc_id, doc in sorted(preds_by_doc.items()):
    short_name = doc_id.split("_in_")[1].split("_courtlistener")[0] if "_in_" in doc_id else doc_id[:50]
    print(f"\n{'━' * 90}")
    print(f"  {short_name.upper()}")
    print(f"{'━' * 90}")

    for chunk in doc["chunks"]:
        cid = chunk["chunk_id"]
        key = (doc_id, cid)
        if key not in human_by_chunk:
            continue

        human = human_by_chunk[key]
        human_summary = human.get("Summary", "").strip()
        human_tags = human.get("Error Tags", "")
        if not human_summary:
            continue

        pred_narrative, causal_notes = events_to_narrative(chunk)
        pred_norm = normalize(pred_narrative + " " + chunk.get("text", ""))

        # Extract facts from human summary
        facts = extract_fact_phrases(human_summary)

        # Score each fact
        covered_facts = []
        partial_facts = []
        missed_facts = []

        for fact in facts:
            is_covered, score = fact_is_covered(fact, pred_norm)
            if is_covered:
                covered_facts.append((fact, score))
            elif score >= 0.25:
                partial_facts.append((fact, score))
            else:
                missed_facts.append((fact, score))

        total = len(facts)
        if total == 0:
            overall = 1.0
        else:
            overall = (len(covered_facts) + 0.5 * len(partial_facts)) / total

        result = {
            "doc_id": doc_id,
            "chunk_id": cid,
            "score": round(overall, 3),
            "total_facts": total,
            "covered": len(covered_facts),
            "partial": len(partial_facts),
            "missed": len(missed_facts),
            "covered_facts": [(f, s) for f, s in covered_facts],
            "partial_facts": [(f, s) for f, s in partial_facts],
            "missed_facts": [(f, s) for f, s in missed_facts],
            "human_summary": human_summary,
            "pred_narrative": pred_narrative,
            "human_tags": human_tags,
        }
        all_results.append(result)

        # Display
        bar = "█" * int(overall * 20) + "░" * (20 - int(overall * 20))
        print(f"\n  ┌─ {cid} ─ Score: {overall:.0%} [{bar}]")
        print(f"  │  Facts: {len(covered_facts)} covered, {len(partial_facts)} partial, {len(missed_facts)} missed (of {total})"
              + (f"  |  Human flagged: {human_tags}" if human_tags else ""))

        print(f"  │")
        print(f"  │  HUMAN:")
        for line in textwrap.wrap(human_summary, 82):
            print(f"  │    {line}")
        print(f"  │")
        print(f"  │  PREDICTION:")
        for line in textwrap.wrap(pred_narrative[:500], 82):
            print(f"  │    {line}")
        if len(pred_narrative) > 500:
            print(f"  │    [... +{len(pred_narrative)-500} chars]")

        # Show fact-level detail
        if covered_facts:
            print(f"  │")
            print(f"  │  ✓ COVERED:")
            for fact, sc in covered_facts[:4]:
                print(f"  │    ({sc:.0%}) {fact[:80]}")
        if partial_facts:
            print(f"  │  ~ PARTIAL:")
            for fact, sc in partial_facts[:3]:
                print(f"  │    ({sc:.0%}) {fact[:80]}")
        if missed_facts:
            print(f"  │  ✗ MISSED:")
            for fact, sc in missed_facts[:3]:
                print(f"  │    ({sc:.0%}) {fact[:80]}")

        print(f"  └{'─' * 85}")


# ── Summary ───────────────────────────────────────────────────────────────────
scores = [r["score"] for r in all_results]
avg = sum(scores) / len(scores)

print(f"\n{'=' * 90}")
print("OVERALL SUMMARY")
print(f"{'=' * 90}")
print(f"  Chunks evaluated:     {len(all_results)}")
print(f"  Avg faithfulness:     {avg:.1%}")
print(f"  Median:               {sorted(scores)[len(scores)//2]:.1%}")
print(f"  Best:                 {max(scores):.1%}")
print(f"  Worst:                {min(scores):.1%}")

# Per-doc
print(f"\n  Per-document:")
doc_scores = defaultdict(list)
for r in all_results:
    short = r["doc_id"].split("_in_")[1].split("_courtlistener")[0] if "_in_" in r["doc_id"] else r["doc_id"][:35]
    doc_scores[short].append(r["score"])
for doc, ss in sorted(doc_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
    a = sum(ss)/len(ss)
    bar = "█" * int(a * 20) + "░" * (20 - int(a * 20))
    print(f"    {doc[:45]:<45} {a:.0%} [{bar}] ({len(ss)} chunks)")

# Distribution
buckets = {"90-100%": 0, "75-89%": 0, "60-74%": 0, "40-59%": 0, "<40%": 0}
for s in scores:
    if s >= 0.9: buckets["90-100%"] += 1
    elif s >= 0.75: buckets["75-89%"] += 1
    elif s >= 0.6: buckets["60-74%"] += 1
    elif s >= 0.4: buckets["40-59%"] += 1
    else: buckets["<40%"] += 1
print(f"\n  Score distribution:")
for b, c in buckets.items():
    print(f"    {b}: {c} chunks {'█' * c}")

# What's consistently missed
all_missed = [f for r in all_results for f, _ in r["missed_facts"]]
print(f"\n  Total facts fully missed: {len(all_missed)} out of {sum(r['total_facts'] for r in all_results)}")

if all_missed:
    print(f"\n  Examples of missed facts:")
    for m in all_missed[:8]:
        print(f"    • {m[:100]}")

# Save
output_path = DATA / "semantic_eval_v2_results.json"
with open(output_path, "w") as f:
    json.dump({
        "summary": {
            "chunks_evaluated": len(all_results),
            "avg_score": round(avg, 4),
            "median": round(sorted(scores)[len(scores)//2], 4),
            "best": round(max(scores), 4),
            "worst": round(min(scores), 4),
        },
        "per_chunk": [{
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "score": r["score"],
            "facts_covered": r["covered"],
            "facts_partial": r["partial"],
            "facts_missed": r["missed"],
            "missed_facts": [f for f, _ in r["missed_facts"]],
            "human_summary": r["human_summary"],
            "pred_narrative": r["pred_narrative"],
        } for r in all_results],
    }, f, indent=2, ensure_ascii=False)
print(f"\n  Saved to: {output_path}")
