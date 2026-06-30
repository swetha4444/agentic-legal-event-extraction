#!/usr/bin/env python3
"""
LLM-based clustering: define K clusters from a corpus sample, then assign each document.

Silhouette is reported **in a fixed numeric embedding space** (Core6D or Core6D+hybrid),
using the **LLM-assigned labels**. That measures alignment between semantic grouping
and separation in the chosen structural embedding (not "LLM-native" geometry).

Requires AGENT_API_KEY (Keymaker) + openai, or ANTHROPIC_API_KEY + anthropic.
Optional CLUSTERING_COHERENCE_MODEL (see run_clustering_comparison._llm_coherence_completion).

Example:
  python3 clustering/scripts/run_llm_clustering.py --repo-root . --k 4 --seed 42

Recompute silhouette from a saved run (no LLM):
  python3 clustering/scripts/run_llm_clustering.py --repo-root . \\
    --from-json clustering/results/llm_clustering_labels.json --skip-llm
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_comparison_module():
    here = Path(__file__).resolve().parent / "run_clustering_comparison.py"
    spec = importlib.util.spec_from_file_location("clustering_compare", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise json.JSONDecodeError("no JSON object", text, 0)


def _excerpt(text: str, max_chars: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= max_chars else t[: max_chars - 3].rstrip() + "..."


def _repo_root(script_path: Path) -> Path:
    cc = _load_comparison_module()
    return cc._repo_root(script_path)  # type: ignore[attr-defined]


def _build_embeddings(
    cc,
    graphs: List[Dict],
    space: str,
    top_ent: int,
    top_evt: int,
    top_rl: int,
    hybrid_pca_dim: int,
) -> Tuple[np.ndarray, str]:
    if space == "core6d":
        X = cc.build_core6d_embedding(graphs)
        return X, "core6d_row_l2"
    if space == "hybrid":
        X = cc.build_core6d_plus_ekg_context_embedding(
            graphs, top_ent, top_evt, top_rl, context_pca_dim=int(hybrid_pca_dim)
        )
        return X, f"core6d_plus_ekgctx_pca{int(hybrid_pca_dim)}"
    raise SystemExit(f"Unknown --embedding-space {space!r} (use core6d or hybrid)")


def _phase1_cluster_definitions(
    cc,
    repo_root: Path,
    k: int,
    sample_excerpts: List[Tuple[str, str]],
    max_tokens: int,
) -> List[Dict[str, Any]]:
    lines = [f"Document {i + 1} (id={doc_id}):\n{ex}" for i, (doc_id, ex) in enumerate(sample_excerpts)]
    body = "\n\n---\n\n".join(lines)
    prompt = f"""You are clustering legal complaints into exactly {k} disjoint groups for analysis.

Below are {len(sample_excerpts)} representative excerpts (IDs are opaque strings).

{body}

Propose exactly {k} clusters that partition this kind of corpus. Each cluster must be distinct.

Respond ONLY with JSON:
{{
  "clusters": [
    {{"cluster_id": 0, "name": "<short label>", "criteria": "<1-3 sentences: who/what/when distinguishes this group>"}},
    ... exactly {k} objects with cluster_id 0..{k - 1} ...
  ]
}}"""
    raw = cc._llm_coherence_completion(prompt, repo_root, max_tokens=max_tokens)
    try:
        obj = _parse_json_object(raw)
    except json.JSONDecodeError:
        return [
            {"cluster_id": i, "name": f"Cluster {i}", "criteria": "Fallback: model JSON parse failed."}
            for i in range(k)
        ]
    clusters = obj.get("clusters")
    if not isinstance(clusters, list) or len(clusters) < k:
        return [
            {"cluster_id": i, "name": f"Cluster {i}", "criteria": "Fallback: invalid clusters array."}
            for i in range(k)
        ]
    # Use model order (prompt asks for cluster_id 0..k-1); force stable ids 0..k-1 for the menu.
    out: List[Dict[str, Any]] = []
    for i in range(k):
        row = clusters[i] if i < len(clusters) else {}
        if not isinstance(row, dict):
            row = {}
        out.append(
            {
                "cluster_id": i,
                "name": str(row.get("name") or f"Cluster {i}")[:200],
                "criteria": str(row.get("criteria") or "")[:1200],
            }
        )
    return out


def _format_cluster_menu(defs: List[Dict[str, Any]]) -> str:
    lines = []
    for d in defs:
        lines.append(
            f"- cluster_id={int(d['cluster_id'])}: {d.get('name', '')}\n  Criteria: {d.get('criteria', '')}"
        )
    return "\n".join(lines)


def _phase2_assign_batch(
    cc,
    repo_root: Path,
    k: int,
    menu: str,
    batch: List[Tuple[str, str]],
    max_tokens: int,
) -> Dict[str, int]:
    doc_lines = []
    for doc_id, ex in batch:
        safe_id = doc_id.replace("`", "'")
        doc_lines.append(f"DOC_ID: {safe_id}\nTEXT:\n{ex}")
    block = "\n\n---\n\n".join(doc_lines)
    ids_literal = ", ".join(repr(doc_id) for doc_id, _ in batch)
    prompt = f"""You assign each legal complaint to exactly one of {k} clusters.

Cluster menu:
{menu}

Documents to assign (you must output one row per DOC_ID below, same order as listed):
{block}

Rules:
- Use only cluster_id integers 0 through {k - 1}.
- Every listed DOC_ID must appear exactly once in assignments.

Respond ONLY with JSON:
{{"assignments": [{{"doc_id": "<exact DOC_ID string>", "cluster_id": <0-{k - 1}>}}, ...]}}

The assignments array must have length {len(batch)} and cover these doc_ids in order: [{ids_literal}]."""
    raw = cc._llm_coherence_completion(prompt, repo_root, max_tokens=max_tokens)
    try:
        obj = _parse_json_object(raw)
    except json.JSONDecodeError:
        return {}
    assign = obj.get("assignments")
    if not isinstance(assign, list):
        return {}
    out: Dict[str, int] = {}
    for item in assign:
        if not isinstance(item, dict):
            continue
        did = str(item.get("doc_id", "")).strip()
        try:
            cid = int(item.get("cluster_id"))
        except (TypeError, ValueError):
            continue
        cid = max(0, min(k - 1, cid))
        if did:
            out[did] = cid
    return out


def _assign_single_doc(
    cc,
    repo_root: Path,
    k: int,
    menu: str,
    doc_id: str,
    excerpt: str,
    max_tokens: int,
) -> int:
    safe_id = doc_id.replace("`", "'")
    prompt = f"""Assign this legal complaint to exactly one cluster.

Cluster menu:
{menu}

DOC_ID: {safe_id}
TEXT:
{excerpt}

Respond ONLY with JSON:
{{"doc_id": "{safe_id}", "cluster_id": <0-{k - 1}>}}"""
    raw = cc._llm_coherence_completion(prompt, repo_root, max_tokens=max_tokens)
    try:
        obj = _parse_json_object(raw)
        cid = int(obj.get("cluster_id", 0))
        return max(0, min(k - 1, cid))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0


def _mock_assign(k: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, k, size=n, dtype=np.int64)


def main() -> None:
    script_here = Path(__file__).resolve()
    p = argparse.ArgumentParser(description="LLM clustering + silhouette in Core6D/hybrid space.")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--ekg-jsonl", type=Path, default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--embedding-space",
        choices=("core6d", "hybrid"),
        default="core6d",
        help="Numeric space used for silhouette (LLM assigns from text, not this vector).",
    )
    p.add_argument("--hybrid-pca-dim", type=int, default=16)
    p.add_argument("--ekg-top-entity-types", type=int, default=40)
    p.add_argument("--ekg-top-event-types", type=int, default=40)
    p.add_argument("--ekg-top-roles", type=int, default=20)
    p.add_argument("--phase1-samples", type=int, default=12, help="Excerpts shown to define clusters.")
    p.add_argument("--text-chars-phase1", type=int, default=700)
    p.add_argument("--text-chars-assign", type=int, default=900)
    p.add_argument("--batch-size", type=int, default=4, help="Documents per assignment LLM call.")
    p.add_argument("--max-tokens-phase1", type=int, default=1200)
    p.add_argument("--max-tokens-assign", type=int, default=1200)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write full result (default: <repo>/clustering/results/llm_clustering_labels.json)",
    )
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Load doc_id -> cluster_id from this file; recompute silhouette only (--skip-llm implied).",
    )
    p.add_argument(
        "--mock-llm",
        action="store_true",
        help="Random labels (testing silhouette path without API keys).",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="With --from-json: skip API; only silhouette. Without --from-json: same as --mock-llm.",
    )
    ns = p.parse_args()

    cc = _load_comparison_module()
    root = ns.repo_root.expanduser().resolve() if ns.repo_root else _repo_root(script_here)
    cc._load_repo_dotenv(root)
    dr_ekg, dr_out = cc._defaults(root)
    ekg_path = ns.ekg_jsonl.expanduser().resolve() if ns.ekg_jsonl else dr_ekg
    out_path = (
        ns.output_json.expanduser().resolve()
        if ns.output_json
        else (root / "clustering" / "results" / "llm_clustering_labels.json")
    )

    k = max(2, int(ns.k))
    rng = np.random.RandomState(int(ns.seed))

    docs = cc.load_documents_and_graphs(ekg_path, limit=ns.max_docs)
    doc_ids = sorted(docs.keys())

    cluster_defs: List[Dict[str, Any]] = []
    labels_map: Dict[str, int] = {}

    if ns.from_json is not None:
        raw_obj = json.loads(Path(ns.from_json).expanduser().resolve().read_text(encoding="utf-8"))
        cluster_defs = list(raw_obj.get("cluster_definitions") or [])
        if "k" in raw_obj:
            k = max(2, int(raw_obj["k"]))
        assign = raw_obj.get("assignments")
        labels_list = raw_obj.get("labels")
        file_doc_ids = raw_obj.get("doc_ids")

        if isinstance(labels_list, list) and isinstance(file_doc_ids, list):
            fids = [str(x) for x in file_doc_ids]
            if len(fids) != len(labels_list):
                raise SystemExit("--from-json: doc_ids and labels must have the same length")
            doc_ids = fids
            labels_map = {doc_ids[i]: int(labels_list[i]) for i in range(len(doc_ids))}
        elif isinstance(assign, dict):
            labels_map = {str(kk): int(vv) for kk, vv in assign.items()}
            if isinstance(file_doc_ids, list) and file_doc_ids:
                cand = [str(x) for x in file_doc_ids]
                if all(d in labels_map for d in cand):
                    doc_ids = cand
                else:
                    doc_ids = sorted(labels_map.keys())
            else:
                doc_ids = sorted(labels_map.keys())
        elif isinstance(assign, list):
            for row in assign:
                if isinstance(row, dict) and "doc_id" in row:
                    labels_map[str(row["doc_id"])] = int(row["cluster_id"])
            if isinstance(file_doc_ids, list) and file_doc_ids:
                cand = [str(x) for x in file_doc_ids]
                if all(d in labels_map for d in cand):
                    doc_ids = cand
                else:
                    doc_ids = sorted(labels_map.keys())
            else:
                doc_ids = sorted(labels_map.keys())
        else:
            raise SystemExit(
                "--from-json: provide assignments {{doc_id: cluster_id}}, "
                "assignments list, or parallel doc_ids + labels arrays"
            )
        if not labels_map:
            raise SystemExit("--from-json: empty labels")
        for d in doc_ids:
            if d not in labels_map:
                raise SystemExit(f"--from-json: no cluster_id for doc_id={d!r}")
        missing = [d for d in doc_ids if d not in docs]
        if missing:
            raise SystemExit(f"--from-json: {len(missing)} doc_ids not in JSONL (e.g. {missing[:3]!r})")

    texts = [str(docs[d].get("text") or "") for d in doc_ids]
    graphs = [docs[d].get("merged_graph") or {} for d in doc_ids]
    n = len(doc_ids)
    if n < k + 1:
        raise SystemExit(f"Need at least k+1={k+1} documents; got {n}")

    X, emb_name = _build_embeddings(
        cc,
        graphs,
        ns.embedding_space,
        int(ns.ekg_top_entity_types),
        int(ns.ekg_top_event_types),
        int(ns.ekg_top_roles),
        int(ns.hybrid_pca_dim),
    )

    if ns.from_json is None and (ns.mock_llm or ns.skip_llm):
        print("Using random cluster assignments (--mock-llm or --skip-llm without --from-json).")
        labs = _mock_assign(k, n, int(ns.seed))
        labels_map = {doc_ids[i]: int(labs[i]) for i in range(n)}
        cluster_defs = [
            {"cluster_id": i, "name": f"Random cluster {i}", "criteria": "mock-llm"} for i in range(k)
        ]
    elif ns.from_json is None:
        # Phase 1: sample for cluster menu
        idxs = list(range(n))
        rng.shuffle(idxs)
        n_sample = min(int(ns.phase1_samples), n)
        sample_pairs = [
            (doc_ids[i], _excerpt(texts[i], int(ns.text_chars_phase1))) for i in idxs[:n_sample]
        ]
        print(f"Phase 1: requesting {k} cluster definitions from {n_sample} excerpts...", flush=True)
        cluster_defs = _phase1_cluster_definitions(
            cc, root, k, sample_pairs, max_tokens=int(ns.max_tokens_phase1)
        )
        menu = _format_cluster_menu(cluster_defs)
        print("Cluster menu:\n", menu[:2000], ("..." if len(menu) > 2000 else ""), flush=True)

        # Phase 2: batched assignment
        bs = max(1, int(ns.batch_size))
        for start in range(0, n, bs):
            chunk_ids = doc_ids[start : start + bs]
            batch = [(d, _excerpt(docs[d]["text"], int(ns.text_chars_assign))) for d in chunk_ids]
            print(f"Phase 2: batch {start // bs + 1}/{(n + bs - 1) // bs} ({len(chunk_ids)} docs)...", flush=True)
            got = _phase2_assign_batch(
                cc, root, k, menu, batch, max_tokens=int(ns.max_tokens_assign)
            )
            for d in chunk_ids:
                if d in got:
                    labels_map[d] = int(got[d])
                else:
                    print(f"  Missing in batch response, single-doc fallback: {d}", flush=True)
                    labels_map[d] = _assign_single_doc(
                        cc,
                        root,
                        k,
                        menu,
                        d,
                        _excerpt(docs[d]["text"], int(ns.text_chars_assign)),
                        max_tokens=256,
                    )

    labels = np.asarray([labels_map[d] for d in doc_ids], dtype=np.int64)
    if len(set(labels.tolist())) < 2:
        print("Warning: fewer than 2 distinct cluster ids; silhouette may be undefined.", flush=True)

    np.random.seed(int(ns.seed))
    sil_sampled = float(cc.compute_silhouette_score(X, labels))
    sil_sklearn = cc.silhouette_metric_sklearn(X, labels)

    counts = Counter(int(x) for x in labels.tolist())
    print("\n=== LLM (or loaded) clustering ===", flush=True)
    print(f"Documents: {n}, k={k}, embedding_space={ns.embedding_space} ({emb_name})", flush=True)
    print(f"Cluster sizes: {dict(sorted(counts.items()))}", flush=True)
    print(f"Silhouette (sampled, same metric as comparison script): {sil_sampled:.4f}", flush=True)
    if sil_sklearn is not None:
        print(f"Silhouette (sklearn full, Euclidean): {sil_sklearn:.4f}", flush=True)
    else:
        print("Silhouette (sklearn): n/a (install sklearn or degenerate labels)", flush=True)

    payload = {
        "method": "llm_menu_then_assign",
        "k": k,
        "seed": int(ns.seed),
        "embedding_space": ns.embedding_space,
        "embedding_name": emb_name,
        "ekg_jsonl": str(ekg_path),
        "silhouette_sampled": sil_sampled,
        "silhouette_sklearn_full": sil_sklearn,
        "cluster_definitions": cluster_defs,
        "assignments": {d: int(labels_map[d]) for d in doc_ids},
        "doc_ids": doc_ids,
        "labels": labels.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

