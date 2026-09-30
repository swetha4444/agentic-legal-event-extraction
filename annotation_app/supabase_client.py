"""
supabase_client.py — Cloud sync layer for the Event Graph Annotator.

Uses the same Supabase project as the Fact/Non-Fact dashboard.
Handles: authentication, chunk upload, annotation CRUD, activity feed.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from supabase import create_client, Client
except ImportError:
    # Supabase not installed — cloud features disabled
    create_client = None
    Client = None

# ---------------------------------------------------------------------------
# Config — reads from the same config.js or env vars
# ---------------------------------------------------------------------------
_CONFIG_JS = Path(__file__).resolve().parent.parent / "annotation_dashboard" / "config.js"


def _parse_config_js() -> dict:
    """Extract SUPABASE_URL and SUPABASE_ANON_KEY from config.js."""
    if not _CONFIG_JS.exists():
        return {}
    text = _CONFIG_JS.read_text()
    out = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if "SUPABASE_URL" in line and ":" in line:
            out["url"] = line.split(":", 1)[1].strip().strip('"').strip("'")
        if "SUPABASE_ANON_KEY" in line and ":" in line:
            out["key"] = line.split(":", 1)[1].strip().strip('"').strip("'")
    return out


def get_supabase_client():
    """Return a Supabase client, or None if not configured or not installed."""
    if create_client is None:
        # supabase package not installed
        return None

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    # Try Streamlit secrets (used on Streamlit Cloud)
    if not url or not key:
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_ANON_KEY")
        except Exception:
            pass

    # Try config.js (used locally)
    if not url or not key:
        cfg = _parse_config_js()
        url = url or cfg.get("url")
        key = key or cfg.get("key")

    if not url or not key:
        return None

    try:
        client = create_client(url, key)
        # Store the URL on the client so we can use it for OAuth
        client._project_url = url.rstrip("/")
        return client
    except Exception:
        return None


def get_supabase_url(client) -> str:
    """Get the Supabase project URL from the client."""
    # Use our stored URL first
    if hasattr(client, "_project_url"):
        return client._project_url
    # Fallback: try common attribute names
    for attr in ("supabase_url", "url", "_supabase_url", "rest_url"):
        val = getattr(client, attr, None)
        if val and isinstance(val, str):
            # rest_url might be like https://xxx.supabase.co/rest/v1
            if "/rest/" in val:
                return val.split("/rest/")[0]
            return val.rstrip("/")
    return ""


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def sign_in_with_password(client, email: str, password: str):
    """Sign in with email/password. Returns session data."""
    return client.auth.sign_in_with_password({"email": email, "password": password})


def get_current_user(client):
    """Get the currently authenticated user, or None."""
    try:
        resp = client.auth.get_user()
        return resp.user if resp else None
    except Exception:
        return None


def set_session(client, access_token: str, refresh_token: str):
    """Restore a session from stored tokens."""
    try:
        client.auth.set_session(access_token, refresh_token)
        return True
    except Exception as e:
        import streamlit as st
        st.warning(f"set_session error: {e}")
        return False


# ---------------------------------------------------------------------------
# Chunk upload (one-time seed)
# ---------------------------------------------------------------------------

def upload_chunks(client, chunks: list[dict], source_file: str, user_id: str):
    """
    Upload chunks to Supabase (idempotent — skips existing chunk_keys).
    Also uploads any pre-existing annotations found in the chunks.
    Called once per source file to seed the DB.
    """
    existing = (
        client.table("event_chunks")
        .select("chunk_key")
        .eq("source_file", source_file)
        .execute()
    )
    existing_keys = {r["chunk_key"] for r in existing.data}

    new_rows = []
    pre_existing_annotations = []

    for i, c in enumerate(chunks):
        if c["chunk_key"] in existing_keys:
            continue
        new_rows.append({
            "source_file": source_file,
            "doc_title": c["doc_title"],
            "chunk_id": c["chunk_id"],
            "chunk_key": c["chunk_key"],
            "chunk_theme": c.get("chunk_theme", ""),
            "chunk_text": c["chunk_text"],
            "events_json": c.get("events", []),
            "entities_json": c.get("entities", []),
            "metadata": c.get("metadata", {}),
            "chunk_index": i,
        })

        # Collect pre-existing annotations from the JSONL
        ann = c.get("annotation", {})
        if ann.get("annotation_completed"):
            pre_existing_annotations.append({
                "chunk_key": c["chunk_key"],
                "user_id": user_id,
                "human_summary": ann.get("human_summary", ""),
                "alignment_label": ann.get("alignment_label", "aligned"),
                "annotation_reasoning": ann.get("annotation_reasoning", ""),
                "error_tags": ann.get("error_tags", []),
                "annotation_completed": True,
            })

    if new_rows:
        # Batch in groups of 100
        for start in range(0, len(new_rows), 100):
            batch = new_rows[start : start + 100]
            client.table("event_chunks").insert(batch).execute()

    # Upload pre-existing annotations (from previously annotated JSONL files)
    if pre_existing_annotations:
        for ann_row in pre_existing_annotations:
            try:
                client.table("event_annotations").upsert(
                    ann_row, on_conflict="chunk_key,user_id"
                ).execute()
            except Exception:
                pass  # Skip if it fails (e.g., duplicate)

    return len(new_rows)


# ---------------------------------------------------------------------------
# Load chunks from cloud
# ---------------------------------------------------------------------------

def load_chunks_from_cloud(client, source_file: str) -> list[dict]:
    """Load all chunks for a given source file, ordered by chunk_index."""
    resp = (
        client.table("event_chunks")
        .select("*")
        .eq("source_file", source_file)
        .order("chunk_index")
        .execute()
    )
    return resp.data


def load_all_annotations(client, source_file: str, ann_table: str = "event_annotations") -> dict:
    """
    Load all annotations for chunks in a source file.
    Returns {chunk_key: {user_id: annotation_dict, ...}, ...}

    ann_table: "event_annotations" (GEPA) or "framenet_annotations" (FrameNet)
    """
    # Get chunk keys for this file
    chunks = (
        client.table("event_chunks")
        .select("chunk_key")
        .eq("source_file", source_file)
        .execute()
    )
    chunk_keys = [r["chunk_key"] for r in chunks.data]

    if not chunk_keys:
        return {}

    # Fetch annotations in batches (Supabase has a filter limit)
    all_annotations: dict[str, dict] = {}
    for start in range(0, len(chunk_keys), 100):
        batch_keys = chunk_keys[start : start + 100]
        resp = (
            client.table(ann_table)
            .select("*")
            .in_("chunk_key", batch_keys)
            .execute()
        )
        for ann in resp.data:
            ck = ann["chunk_key"]
            uid = ann["user_id"]
            all_annotations.setdefault(ck, {})[uid] = ann

    return all_annotations


# ---------------------------------------------------------------------------
# Save / update annotation
# ---------------------------------------------------------------------------

def save_annotation(
    client,
    chunk_key: str,
    user_id: str,
    summary: str,
    label: str,
    reasoning: str,
    tags: list[str],
    ann_table: str = "event_annotations",
    activity_table: str = "event_activity",
):
    """Upsert an annotation for a chunk by this user."""
    row = {
        "chunk_key": chunk_key,
        "user_id": user_id,
        "human_summary": summary,
        "alignment_label": label,
        "annotation_reasoning": reasoning,
        "error_tags": tags,
        "annotation_completed": True,
    }
    client.table(ann_table).upsert(
        row, on_conflict="chunk_key,user_id"
    ).execute()

    # Log activity
    client.table(activity_table).insert({
        "user_id": user_id,
        "chunk_key": chunk_key,
        "action": "annotated",
    }).execute()


# ---------------------------------------------------------------------------
# Activity feed
# ---------------------------------------------------------------------------

def get_recent_activity(client, limit: int = 30, activity_table: str = "event_activity") -> list[dict]:
    """Get the most recent annotation activity with user info."""
    try:
        resp = (
            client.table(activity_table)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        # Enrich with profile info where possible
        user_ids = list(set(r["user_id"] for r in resp.data if r.get("user_id")))
        profiles = {}
        if user_ids:
            try:
                p_resp = client.table("profiles").select("id, email, full_name").in_("id", user_ids).execute()
                profiles = {p["id"]: p for p in p_resp.data}
            except Exception:
                pass
        for r in resp.data:
            r["profiles"] = profiles.get(r.get("user_id"), {})
        return resp.data
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source files available in cloud
# ---------------------------------------------------------------------------

def get_cloud_source_files(client) -> list[str]:
    """Get list of distinct source files that have been uploaded."""
    resp = (
        client.table("event_chunks")
        .select("source_file")
        .execute()
    )
    return sorted(set(r["source_file"] for r in resp.data))


# ---------------------------------------------------------------------------
# Progress stats from cloud
# ---------------------------------------------------------------------------

def get_cloud_progress(client, source_file: str, ann_table: str = "event_annotations") -> dict:
    """Get annotation progress from cloud data."""
    chunks_resp = (
        client.table("event_chunks")
        .select("chunk_key")
        .eq("source_file", source_file)
        .execute()
    )
    total = len(chunks_resp.data)
    chunk_keys = [r["chunk_key"] for r in chunks_resp.data]

    if not chunk_keys:
        return {"total": 0, "completed": 0, "remaining": 0, "pct": 0,
                "by_annotator": {}, "by_label": {}}

    # Count distinct chunk_keys that have at least one completed annotation
    try:
        ann_resp = (
            client.table(ann_table)
            .select("chunk_key, user_id, alignment_label")
            .eq("annotation_completed", True)
            .in_("chunk_key", chunk_keys)
            .execute()
        )
    except Exception:
        # No annotations yet or table doesn't exist
        return {"total": total, "completed": 0, "remaining": total, "pct": 0,
                "by_annotator": {}, "by_label": {}}

    # Look up profiles separately
    user_ids = list(set(a["user_id"] for a in ann_resp.data if a.get("user_id")))
    profiles = {}
    if user_ids:
        try:
            p_resp = client.table("profiles").select("id, email, full_name").in_("id", user_ids).execute()
            profiles = {p["id"]: p for p in p_resp.data}
        except Exception:
            pass

    annotated_keys = set()
    by_annotator: dict[str, int] = {}
    by_label: dict[str, int] = {}

    for ann in ann_resp.data:
        annotated_keys.add(ann["chunk_key"])
        # Count by annotator — use user_id directly (already readable names
        # after migration; legacy UUIDs handled by caller)
        name = ann.get("user_id", "unknown")
        by_annotator[name] = by_annotator.get(name, 0) + 1
        # Count by label
        lbl = ann.get("alignment_label", "unknown")
        by_label[lbl] = by_label.get(lbl, 0) + 1

    completed = len(annotated_keys)
    return {
        "total": total,
        "completed": completed,
        "remaining": total - completed,
        "pct": round(100 * completed / total, 1) if total else 0,
        "by_annotator": by_annotator,
        "by_label": by_label,
    }
