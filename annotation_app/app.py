"""
app.py — Streamlit-based collaborative annotation UI for human evaluation
of event graphs extracted from legal text chunks.

CLOUD-SYNCED VERSION: Uses Supabase for shared annotation state.
All teammates see each other's progress in real time.
"""

import os
import json
import re
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from utils import (
    load_data, make_annotation_dict,
    get_progress_stats, ERROR_TAGS, ALIGNMENT_LABELS
)
from supabase_client import (
    get_supabase_client,
    upload_chunks, load_chunks_from_cloud, load_all_annotations,
    save_annotation, get_recent_activity, get_cloud_source_files,
    get_cloud_progress,
)

# ---------------------------------------------------------------------------
# Team members — add / remove names here
# ---------------------------------------------------------------------------
TEAM = ["Heo", "Swetha", "Vishnu", "Sriram"]

# Map old UUID-based user_ids (from before name-picker migration) to names
LEGACY_UUID_MAP = {
    "218b4166-e5d5-4f64-86b6-2adf3c14ee70": "heo",
}

def _resolve_user_id(uid: str) -> str:
    """Map legacy UUID user_ids to readable names."""
    return LEGACY_UUID_MAP.get(uid, uid)

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
# We need to hash the Supabase client object to safely use @st.cache_data
# We do this by ignoring the client object entirely.
SB_HASH = {"_supabase.client.Client": lambda _: None}

# Decorators to prevent synchronous blocking on every rerun
@st.cache_data(ttl=10, show_spinner=False, hash_funcs=SB_HASH)
def cached_get_cloud_source_files(_sb):
    return get_cloud_source_files(_sb)

@st.cache_data(ttl=10, show_spinner=False, hash_funcs=SB_HASH)
def cached_load_chunks_from_cloud(_sb, source_file):
    return load_chunks_from_cloud(_sb, source_file)

@st.cache_data(ttl=10, show_spinner=False, hash_funcs=SB_HASH)
def cached_load_all_annotations(_sb, source_file, ann_table="event_annotations"):
    return load_all_annotations(_sb, source_file, ann_table=ann_table)

@st.cache_data(ttl=10, show_spinner=False, hash_funcs=SB_HASH)
def cached_get_cloud_progress(_sb, source_file, ann_table="event_annotations"):
    return get_cloud_progress(_sb, source_file, ann_table=ann_table)

@st.cache_data(ttl=15, show_spinner=False, hash_funcs=SB_HASH)
def cached_get_recent_activity(_sb, limit=15, activity_table="event_activity"):
    return get_recent_activity(_sb, limit, activity_table=activity_table)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
GEPA_GRAPH_DIR = OUTPUT_DIR / "gepa_rendered_graphs_two_stage"
FRAMENET_GRAPH_DIR = OUTPUT_DIR / "framenet_renders"

# ---------------------------------------------------------------------------
# Graph mode configs — each mode maps to its own graph dir, filename suffix,
# Supabase annotation table, and activity table.
# ---------------------------------------------------------------------------
GRAPH_MODES = {
    "GEPA": {
        "graph_dir": GEPA_GRAPH_DIR,
        "suffix": "event-only",
        "ann_table": "event_annotations",
        "activity_table": "event_activity",
        "label": "GEPA Event Graph",
    },
    "FrameNet": {
        "graph_dir": FRAMENET_GRAPH_DIR,
        "suffix": "graph",
        "ann_table": "framenet_annotations",
        "activity_table": "framenet_activity",
        "label": "FrameNet Graph",
    },
}


def _load_graph_html(chunk_key: str, graph_mode: str = "GEPA") -> str | None:
    """
    Load a pre-rendered vis.js graph HTML for a chunk.
    chunk_key format: "doc_id::chunk_id"
    graph_mode: "GEPA" or "FrameNet"
    Returns the full HTML string, or None if not found.
    """
    if "::" not in chunk_key or graph_mode not in GRAPH_MODES:
        return None
    cfg = GRAPH_MODES[graph_mode]
    doc_id, chunk_id = chunk_key.split("::", 1)
    fname = f"{doc_id}__{chunk_id}__{cfg['suffix']}.html"
    path = cfg["graph_dir"] / fname
    if path.exists():
        html = path.read_text(encoding="utf-8")
        # Adapt for embedding: shrink network height, dark theme
        html = html.replace("height: 880px", "height: 520px")
        html = re.sub(r'min-height:\s*\d+px;?', '', html)
        html = html.replace("background: #faf8f2", "background: #0e1117")
        html = html.replace("background: #ffffff", "background: #161b22")
        html = html.replace("color: #444", "color: #ccc")
        html = html.replace('font-size: 20px; font-weight: 600;',
                            'font-size: 14px; font-weight: 600; color: #ccc;')
        return html
    return None


def _has_graph(chunk_key: str, graph_mode: str) -> bool:
    """Check if a graph file exists for this chunk in the given mode."""
    if "::" not in chunk_key or graph_mode not in GRAPH_MODES:
        return False
    cfg = GRAPH_MODES[graph_mode]
    doc_id, chunk_id = chunk_key.split("::", 1)
    fname = f"{doc_id}__{chunk_id}__{cfg['suffix']}.html"
    return (cfg["graph_dir"] / fname).exists()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Event Graph Annotator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .chunk-text {
        font-size: 1.05rem;
        line-height: 1.7;
        max-height: 520px;
        overflow-y: auto;
        padding: 1rem 1.2rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        background: #f8f9fa;
        color: #1a1a1a;
        white-space: pre-wrap;
    }
    .event-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
        background: #f0f2f5;
    }
    .stat-box {
        text-align: center;
        padding: 0.6rem;
        border-radius: 8px;
        background: #f0f2f5;
        border: 1px solid #ddd;
    }
    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .chunk-text {
            border-color: #333;
            background: #0e1117;
            color: #e6edf3;
        }
        .event-card {
            border-color: #444;
            background: #161b22;
        }
        .stat-box {
            background: #161b22;
            border-color: #333;
        }
    }
    /* Streamlit sets data-theme on root — use that too */
    [data-theme="dark"] .chunk-text,
    .stApp[data-testid="stAppViewContainer"]:has([data-theme="dark"]) .chunk-text {
        border-color: #333;
        background: #0e1117;
        color: #e6edf3;
    }
    [data-theme="dark"] .event-card {
        border-color: #444;
        background: #161b22;
    }
    [data-theme="dark"] .stat-box {
        background: #161b22;
        border-color: #333;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-badge.done {
        background: #238636; color: #fff;
    }
    .status-badge.mine {
        background: #1f6feb; color: #fff;
    }
    .activity-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid #ddd;
        font-size: 0.85rem;
    }
    .team-stats {
        padding: 0.5rem;
        background: #f0f2f5;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .activity-item { border-bottom-color: #222; }
        .team-stats { background: #161b22; border-color: #333; }
    }
    [data-theme="dark"] .activity-item { border-bottom-color: #222; }
    [data-theme="dark"] .team-stats { background: #161b22; border-color: #333; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Defaults
# ---------------------------------------------------------------------------
for key, default in {
    "current_idx": 0,
    "user": None,        # will hold {"name": "Heo", "id": "heo"}
    "cloud_ready": False,
    "graph_mode": "GEPA",  # "GEPA" or "FrameNet"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Supabase client init (for cloud sync — no auth needed)
# ---------------------------------------------------------------------------
if "sb" not in st.session_state:
    st.session_state.sb = get_supabase_client()
sb = st.session_state.sb

if sb:
    st.session_state.cloud_ready = True


# ===================================================================
# LOGIN PAGE — simple name picker (when2meet style)
# ===================================================================
def show_login_page():
    st.markdown("## ⚖️ Legal Event Graph Annotator")
    st.markdown("### Collaborative annotation for your team")
    st.markdown("---")
    st.markdown("**Who are you?** Pick your name to start annotating.")

    cols = st.columns(len(TEAM))
    for i, name in enumerate(TEAM):
        with cols[i]:
            if st.button(name, use_container_width=True, type="primary"):
                st.session_state.user = {
                    "name": name,
                    "id": name.lower().replace(" ", "_"),
                }
                st.rerun()

    if not sb:
        st.warning("Supabase not configured. Annotations won't sync to the cloud.")


# ===================================================================
# MAIN ANNOTATOR UI (Cloud or Local)
# ===================================================================
def show_annotator():
    user = st.session_state.user
    is_cloud = st.session_state.cloud_ready and sb is not None
    user_id = user["id"]
    user_name = user["name"]

    # -----------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------
    # Read current graph mode from session state (selector is in the router)
    graph_mode = st.session_state.graph_mode
    mode_cfg = GRAPH_MODES[graph_mode]
    ann_table = mode_cfg["ann_table"]
    activity_table = mode_cfg["activity_table"]

    with st.sidebar:
        # ------- File / dataset selection -------
        if is_cloud:
            st.markdown("**Mode: Cloud Sync**")

            # Show local files that can be uploaded
            local_files = sorted([
                f.name for f in OUTPUT_DIR.glob("chunk_event_graphs_*.jsonl")
            ])
            cloud_files = cached_get_cloud_source_files(sb)

            # Upload button for local files not yet in cloud
            not_uploaded = [f for f in local_files if f not in cloud_files]
            if not_uploaded:
                st.markdown("**Upload local data to cloud:**")
                for fname in not_uploaded:
                    if st.button(f"Upload {fname}", key=f"upload_{fname}"):
                        with st.spinner(f"Uploading {fname}..."):
                            chunks = load_data(str(OUTPUT_DIR / fname))
                            n = upload_chunks(sb, chunks, fname, user_id)
                            st.cache_data.clear() # Invalidate cache
                            st.success(f"Uploaded {n} new chunks!")
                            st.rerun()
                st.divider()

            all_files = sorted(set(cloud_files + local_files))

            # If no files anywhere, show a file uploader
            if not all_files:
                st.markdown("**No data yet — upload a JSONL file:**")

            uploaded_file = st.file_uploader(
                "Upload JSONL file",
                type=["jsonl"],
                help="Upload a chunk_event_graphs_*.jsonl file to seed the database.",
            )
            if uploaded_file is not None:
                import tempfile
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save to temp file, parse, upload to Supabase
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".jsonl", delete=False
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    chunks = load_data(tmp_path)
                    os.unlink(tmp_path)
                    if chunks:
                        n = upload_chunks(sb, chunks, uploaded_file.name, user_id)
                        st.cache_data.clear() # Invalidate cache
                        st.success(f"Uploaded {n} chunks from {uploaded_file.name}!")
                        st.rerun()
                    else:
                        st.error("No chunks found in that file. Make sure it's a valid event graph JSONL.")

            if not all_files:
                st.stop()

            st.divider()

            selected_file = st.selectbox(
                "Source File",
                all_files,
                index=0,
            )

            # Load from cloud
            cloud_chunks = cached_load_chunks_from_cloud(sb, selected_file)
            if not cloud_chunks:
                # Not uploaded yet — offer to upload
                st.warning("This file hasn't been uploaded to the cloud yet.")
                if st.button("Upload Now", type="primary"):
                    chunks = load_data(str(OUTPUT_DIR / selected_file))
                    n = upload_chunks(sb, chunks, selected_file, user_id)
                    st.cache_data.clear() # Invalidate cache
                    st.success(f"Uploaded {n} chunks!")
                    st.rerun()
                st.stop()

            # Build chunks list from cloud data
            chunks = []
            for cc in cloud_chunks:
                chunks.append({
                    "chunk_key": cc["chunk_key"],
                    "doc_title": cc["doc_title"],
                    "chunk_id": cc["chunk_id"],
                    "chunk_theme": cc.get("chunk_theme", ""),
                    "chunk_text": cc["chunk_text"],
                    "events": cc.get("events_json", []),
                    "entities": cc.get("entities_json", []),
                    "metadata": cc.get("metadata", {}),
                })

            # Load all annotations (from the correct table for this graph mode)
            all_anns = cached_load_all_annotations(sb, selected_file, ann_table=ann_table)

            # Merge annotations: track both per-user and team-wide status
            # Remap legacy UUIDs so old annotations are recognized as ours
            for c in chunks:
                ck = c["chunk_key"]
                raw_dict = all_anns.get(ck, {})
                ann_dict = {_resolve_user_id(uid): a for uid, a in raw_dict.items()}
                my_ann = ann_dict.get(user_id, {})
                completed_by = [
                    uid for uid, a in ann_dict.items()
                    if a.get("annotation_completed")
                ]
                c["annotation"] = my_ann if my_ann else {}
                c["all_annotations"] = ann_dict
                c["any_annotated"] = len(completed_by) > 0
                c["i_annotated"] = my_ann.get("annotation_completed", False)
                c["annotator_count"] = len(completed_by)

            stats = cached_get_cloud_progress(sb, selected_file, ann_table=ann_table)

        # ------- Progress -------
        st.divider()
        st.subheader(f"Progress ({stats['pct']}%)")
        c1, c2 = st.columns(2)
        c1.metric("Done", stats["completed"])
        c2.metric("Left", stats["remaining"])
        st.progress(stats["pct"] / 100)

        # Per-annotator breakdown
        if stats.get("by_annotator"):
            st.markdown("**By annotator:**")
            for name, count in stats["by_annotator"].items():
                st.caption(f"  {name}: {count}")

        st.divider()

        # ------- Filters -------
        st.subheader("Filter")
        f_mode = st.selectbox(
            "Show",
            ["My Unannotated", "All", "My Annotated", "Nobody Annotated"],
            index=0,
        )

        indices = []
        for i, c in enumerate(chunks):
            if f_mode == "All":
                indices.append(i)
            elif f_mode == "My Unannotated" and not c["i_annotated"]:
                indices.append(i)
            elif f_mode == "My Annotated" and c["i_annotated"]:
                indices.append(i)
            elif f_mode == "Nobody Annotated" and not c["any_annotated"]:
                indices.append(i)

        st.caption(f"Viewing {len(indices)} chunks")

        # ------- Jump to next chunk I haven't annotated -------
        if st.button("Jump to My Next", use_container_width=True):
            for i, c_idx in enumerate(indices):
                if i <= st.session_state.current_idx:
                    continue
                c = chunks[c_idx]
                if not c["i_annotated"]:
                    st.session_state.current_idx = i
                    st.rerun()
                    break

        # ------- Activity Feed (cloud only) -------
        if is_cloud:
            st.divider()
            st.subheader("Recent Activity")
            activity = cached_get_recent_activity(sb, limit=15, activity_table=activity_table)
            for act in activity:
                raw_uid = act.get("user_id", "someone")
                name = _resolve_user_id(raw_uid)
                ts = act["created_at"][:16].replace("T", " ")
                ck = act["chunk_key"]
                # Show just the chunk_id part
                chunk_part = ck.split("::")[-1] if "::" in ck else ck
                # Hide timestamps for old annotations (before April 2026)
                hide_ts = ts < "2026-04"
                ts_html = "" if hide_ts else f'<br/><small style="color:#888">{ts}</small>'
                st.markdown(
                    f'<div class="activity-item">'
                    f'<b>{name}</b> annotated <code>{chunk_part}</code>'
                    f'{ts_html}</div>',
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------
    if not indices:
        st.info("No chunks match your filter. Try a different filter.")
        st.stop()

    if st.session_state.current_idx >= len(indices):
        st.session_state.current_idx = 0

    curr_chunk = chunks[indices[st.session_state.current_idx]]
    ann = curr_chunk.get("annotation", {})

    # ------- Header -------
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"Chunk {st.session_state.current_idx + 1} of {len(indices)}")
        st.caption(
            f"**Document:** {curr_chunk['doc_title']} | "
            f"**ID:** {curr_chunk['chunk_id']}"
        )

    with col2:
        n_annotators = curr_chunk.get("annotator_count", 0)
        i_did = curr_chunk.get("i_annotated", False)
        if i_did and n_annotators > 1:
            st.markdown(
                f'<span class="status-badge mine">YOU + {n_annotators - 1} other(s)</span>',
                unsafe_allow_html=True,
            )
        elif i_did:
            st.markdown(
                '<span class="status-badge mine">YOU ANNOTATED</span>',
                unsafe_allow_html=True,
            )
        elif n_annotators > 0:
            st.markdown(
                f'<span class="status-badge done">{n_annotators} annotated (not you)</span>',
                unsafe_allow_html=True,
            )
    st.divider()

    # Widget key for this chunk (used for form fields and graph toggle)
    wk = curr_chunk["chunk_key"]

    # ------- Columns: Text vs Graph -------
    L, R = st.columns(2, gap="large")

    with L:
        st.markdown("#### Source Text")
        st.markdown(
            f'<div class="chunk-text">{curr_chunk["chunk_text"]}</div>',
            unsafe_allow_html=True,
        )

    with R:
        chunk_key = curr_chunk.get("chunk_key", "")
        st.markdown(f"#### {mode_cfg['label']}")

        # Load the graph for the selected mode
        graph_html = _load_graph_html(chunk_key, graph_mode)

        if graph_html:
            # Strip the title and legend divs — we already have our own header
            graph_html = re.sub(r'<div class="title">.*?</div>', '', graph_html)
            graph_html = re.sub(r'<div class="legend">.*?</div>', '', graph_html)

            # Inject click handler + sentence lookup into the EXISTING script
            click_handler_js = """
            // --- Click to show detail panel ---
            network.on('click', function(params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    if (!node) return;
                    var panel = document.getElementById('detail-panel');
                    var title = document.getElementById('detail-title');
                    var body = document.getElementById('detail-body');

                    if (node.title) {
                        var raw = node.title
                            .replace(/&quot;/g, '"')
                            .replace(/&amp;/g, '&')
                            .replace(/&lt;/g, '<')
                            .replace(/&gt;/g, '>');
                        var parts = raw.split('<br>');
                        var eventType = '';
                        var html = '';
                        parts.forEach(function(part) {
                            var sep = part.indexOf(':');
                            if (sep > 0 && sep < 30) {
                                var key = part.substring(0, sep).trim();
                                var val = part.substring(sep + 1).trim();
                                if (key === 'event_type') { eventType = val; return; }
                                if (key === 'chunk' || key === 'event_id') return;
                                val = val.replace(/[{}]/g, '').replace(/"/g, '');
                                html += '<div style="margin:6px 0;"><b style="color:#8b949e;">' + key + ':</b> ' + val + '</div>';
                            }
                        });
                        title.textContent = eventType || node.label.split('\\n')[0];
                        body.innerHTML = html;
                    } else {
                        title.textContent = node.label.split('\\n')[0];
                        body.textContent = node.label;
                    }
                    panel.style.display = 'block';
                }
            });
            network.on('hoverNode', function() {
                document.body.style.cursor = 'pointer';
            });
            network.on('blurNode', function() {
                document.body.style.cursor = 'default';
            });

            // --- Sentence ID lookup ---
            // Build a map: sentence_id -> [nodeId, ...]
            var _sentMap = {};
            var _origColors = {};
            var _origEdges = {};
            nodes.forEach(function(n) {
                _origColors[n.id] = {
                    color: n.color ? JSON.parse(JSON.stringify(n.color)) : null,
                    borderWidth: n.borderWidth || 1,
                    shadow: n.shadow ? JSON.parse(JSON.stringify(n.shadow)) : false,
                    font: n.font ? JSON.parse(JSON.stringify(n.font)) : null,
                    size: n.size || null,
                    opacity: 1.0
                };
                // Extract sentence IDs from label and title
                var text = (n.label || '') + ' ' + ((n.title || '').replace(/&quot;/g, '"'));
                // Match patterns: 'sentence_id': 29 and sentence_ids: [29] and [29, 30]
                var re = /sentence_id[s'":\s\[]*(\d+)/g;
                var m;
                while ((m = re.exec(text)) !== null) {
                    var sid = m[1];
                    if (!_sentMap[sid]) _sentMap[sid] = [];
                    if (_sentMap[sid].indexOf(n.id) < 0) _sentMap[sid].push(n.id);
                }
            });
            // Save original edge styles
            edges.forEach(function(e) {
                _origEdges[e.id] = {
                    color: e.color ? JSON.parse(JSON.stringify(e.color)) : null,
                    width: e.width || 1
                };
            });

            function highlightBySentence(sentStr) {
                // Reset ALL nodes to original appearance
                nodes.forEach(function(n) {
                    var orig = _origColors[n.id];
                    if (orig) {
                        var upd = {id: n.id, borderWidth: orig.borderWidth, opacity: 1.0, shadow: orig.shadow};
                        if (orig.color) upd.color = orig.color;
                        if (orig.font) upd.font = orig.font;
                        if (orig.size) upd.size = orig.size;
                        nodes.update(upd);
                    }
                });
                // Reset ALL edges to original appearance
                edges.forEach(function(e) {
                    var orig = _origEdges[e.id];
                    if (orig) {
                        var upd = {id: e.id, width: orig.width};
                        if (orig.color) upd.color = orig.color;
                        edges.update(upd);
                    }
                });
                var countEl = document.getElementById('s-match-count');
                countEl.textContent = '';

                if (!sentStr) return;

                var matches = _sentMap[sentStr] || [];
                if (matches.length > 0) {
                    // Step 1: Dim ALL nodes and edges heavily
                    nodes.forEach(function(n) {
                        nodes.update({id: n.id, opacity: 0.12});
                    });
                    edges.forEach(function(e) {
                        edges.update({id: e.id, color: {color: 'rgba(100,100,100,0.1)', highlight: 'rgba(100,100,100,0.1)'}, width: 0.3});
                    });

                    // Step 2: Make matched nodes EXTREMELY prominent
                    matches.forEach(function(nid) {
                        nodes.update({
                            id: nid,
                            opacity: 1.0,
                            color: {background: '#ff0000', border: '#ffff00', highlight: {background: '#ff0000', border: '#ffff00'}},
                            borderWidth: 6,
                            shadow: {enabled: true, color: 'rgba(255,255,0,0.8)', size: 25, x: 0, y: 0},
                            font: {color: '#ffffff', size: 18, bold: true, strokeWidth: 3, strokeColor: '#000000'},
                            size: 45
                        });
                        // Also brighten edges connected to matched nodes
                        edges.forEach(function(e) {
                            if (e.from === nid || e.to === nid) {
                                edges.update({id: e.id, color: {color: '#ff6666', highlight: '#ff6666'}, width: 3});
                            }
                        });
                    });
                    network.focus(matches[0], {scale: 1.5, animation: {duration: 500}});
                    countEl.textContent = matches.length + ' node(s) found';
                    countEl.style.color = '#ff4444';
                    countEl.style.fontWeight = 'bold';
                } else {
                    countEl.textContent = 'no match';
                    countEl.style.color = '#f85149';
                    countEl.style.fontWeight = 'normal';
                }
            }

            // Defer event-listener wiring until overlay DOM elements exist
            // (they are injected after </script>, before </body>)
            setTimeout(function() {
                var lookupInput = document.getElementById('s-lookup');
                var lookupBtn = document.getElementById('s-lookup-btn');
                if (!lookupBtn || !lookupInput) return;  // safety
                lookupBtn.addEventListener('click', function() {
                    var val = lookupInput.value.replace(/[^0-9]/g, '');
                    highlightBySentence(val);
                });
                lookupInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        var val = this.value.replace(/[^0-9]/g, '');
                        highlightBySentence(val);
                    }
                });
                // Clear resets highlights
                lookupInput.addEventListener('input', function() {
                    if (!this.value.trim()) highlightBySentence(null);
                });
            }, 0);
            """
            # Insert before the LAST </script> (the main vis.js script, not CDN imports)
            last_idx = graph_html.rfind("</script>")
            if last_idx >= 0:
                graph_html = graph_html[:last_idx] + click_handler_js + graph_html[last_idx:]

            # Inject detail panel HTML + fullscreen button before </body>
            overlay_html = """
            <!-- Sentence lookup -->
            <div style="position:fixed; top:8px; left:8px; z-index:9999; display:flex; align-items:center; gap:4px;">
                <input id="s-lookup" type="text" placeholder="Sentence # (e.g. 63)"
                    style="width:130px; padding:5px 10px; border:1px solid #555; border-radius:6px 0 0 6px;
                    background:#161b22; color:#ccc; font-size:13px; outline:none;"
                />
                <button id="s-lookup-btn" style="
                    padding:5px 12px; border:1px solid #555; border-left:none; border-radius:0 6px 6px 0;
                    background:#2d333b; color:#ccc; cursor:pointer; font-size:13px; font-weight:600;
                ">Find</button>
                <span id="s-match-count" style="font-size:12px; margin-left:4px;"></span>
            </div>

            <!-- Fullscreen button -->
            <button id="fs-btn" style="
                position:fixed; top:8px; right:8px; z-index:9999;
                padding:6px 14px; border:1px solid #555; border-radius:6px;
                background:#161b22; color:#ccc; cursor:pointer;
                font-size:13px; font-weight:600;
            " onclick="toggleFS()">⛶ Fullscreen</button>

            <div id="detail-panel" style="
                display:none; position:fixed; bottom:0; left:0; right:0;
                max-height:45%; overflow-y:auto; z-index:9998;
                background:#1c2128; border-top:2px solid #58a6ff;
                padding:16px 20px; font-family:Arial,sans-serif;
                color:#e6edf3; font-size:14px; line-height:1.6;
                box-shadow:0 -4px 20px rgba(0,0,0,0.5);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span id="detail-title" style="font-size:16px; font-weight:700; color:#58a6ff;"></span>
                    <button onclick="closeDetail()" style="
                        background:none; border:1px solid #444; color:#ccc;
                        padding:2px 10px; border-radius:4px; cursor:pointer; font-size:13px;
                    ">✕ Close</button>
                </div>
                <div id="detail-body" style="white-space:pre-wrap;"></div>
            </div>

            <script>
            function toggleFS() {
                var el = document.documentElement;
                if (!document.fullscreenElement) {
                    el.requestFullscreen().catch(function(){});
                    document.getElementById('mynetwork').style.height = '100vh';
                    document.getElementById('fs-btn').textContent = '✕ Exit';
                } else {
                    document.exitFullscreen();
                    document.getElementById('mynetwork').style.height = '520px';
                    document.getElementById('fs-btn').textContent = '⛶ Fullscreen';
                }
            }
            document.addEventListener('fullscreenchange', function() {
                if (!document.fullscreenElement) {
                    document.getElementById('mynetwork').style.height = '520px';
                    document.getElementById('fs-btn').textContent = '⛶ Fullscreen';
                }
            });
            function closeDetail() {
                document.getElementById('detail-panel').style.display = 'none';
            }
            </script>
            """
            graph_html = graph_html.replace("</body>", overlay_html + "</body>")

            # Render the vis.js graph in an iframe — supports drag/zoom
            components.html(graph_html, height=580, scrolling=False)
        else:
            events = curr_chunk.get("events", [])
            if not events:
                pass  # No graph available — leave section empty
            else:
                for ev in events:
                    # Handle both nested graph format and flat format
                    if isinstance(ev, dict):
                        etype = ev.get("event_type", "Unknown")
                        trigger = ev.get("trigger", {})
                        if isinstance(trigger, dict):
                            trigger_text = trigger.get("span_text", "N/A")
                        else:
                            trigger_text = str(trigger)
                        participants = ev.get("participants", [])
                        roles_str = ", ".join([
                            f"{p.get('mention', {}).get('span_text', p.get('mention', 'N/A'))} "
                            f"({p.get('role', '?')})"
                            for p in participants
                            if isinstance(p, dict)
                        ])
                        st.markdown(f"""<div class="event-card">
                            <b style="color:#58a6ff">{etype}</b><br/>
                            <small><b>Trigger:</b> {trigger_text}</small><br/>
                            <small><b>Roles:</b> {roles_str if roles_str else 'N/A'}</small>
                        </div>""", unsafe_allow_html=True)

    st.divider()

    # ------- Other annotations on this chunk (cloud) -------
    if is_cloud and curr_chunk.get("all_annotations"):
        other_anns = {
            uid: a for uid, a in curr_chunk["all_annotations"].items()
            if uid != user_id and a.get("annotation_completed")
        }
        if other_anns:
            with st.expander(f"View other annotations ({len(other_anns)} teammate(s))", expanded=False):
                for uid, oa in other_anns.items():
                    st.markdown(f"**{uid}**")
                    st.markdown(f"- **Alignment:** {oa.get('alignment_label', 'N/A')}")
                    st.markdown(f"- **Summary:** {oa.get('human_summary', 'N/A')}")
                    st.markdown(f"- **Tags:** {', '.join(oa.get('error_tags', []))}")
                    st.markdown("---")

    # ------- Annotation Form -------
    st.markdown("### Your Annotation")

    C1, C2 = st.columns(2)
    with C1:
        h_summary = st.text_area(
            "Human Summary",
            value=ann.get("human_summary", ""),
            height=100,
            key=f"summary_{wk}",
        )
        h_label = st.radio(
            "Alignment",
            ALIGNMENT_LABELS,
            index=(
                ALIGNMENT_LABELS.index(ann["alignment_label"])
                if ann.get("alignment_label") in ALIGNMENT_LABELS else None
            ),
            horizontal=True,
            key=f"alignment_{wk}",
        )

    with C2:
        h_reasoning = st.text_area(
            "Reasoning (if not aligned, explain why — e.g. missing events, wrong roles, hallucinated details)",
            value=ann.get("annotation_reasoning", ""),
            height=100,
            key=f"reasoning_{wk}",
            placeholder="Optional if aligned. If misaligned, briefly explain what's off.",
        )
        h_tags = st.multiselect(
            "Error Tags",
            ERROR_TAGS,
            default=ann.get("error_tags", []),
            key=f"tags_{wk}",
        )

    st.divider()

    # ------- Action Buttons -------
    B1, B2, B3, B4 = st.columns([1, 1, 2, 1])
    with B1:
        if st.button("Prev", disabled=st.session_state.current_idx == 0):
            st.session_state.current_idx -= 1
            st.rerun()

    with B2:
        if st.button("Next", disabled=st.session_state.current_idx >= len(indices) - 1):
            st.session_state.current_idx += 1
            st.rerun()

    def _do_save():
        save_annotation(
            sb, curr_chunk["chunk_key"], user_id,
            h_summary, h_label, h_reasoning, h_tags,
            ann_table=ann_table,
            activity_table=activity_table,
        )
        st.cache_data.clear() # Invalidate cache

    with B3:
        if st.button("SAVE ANNOTATION", type="primary", use_container_width=True):
            _do_save()
            st.success("Saved!")
            st.rerun()

    with B4:
        if st.button(
            "Save & Next",
            use_container_width=True,
            disabled=st.session_state.current_idx >= len(indices) - 1,
        ):
            _do_save()
            st.session_state.current_idx += 1
            st.rerun()


# ===================================================================
# GLOBAL OVERVIEW PAGE
# ===================================================================
def show_overview():
    """Show all chunks across all files with their annotation status + export."""
    user = st.session_state.user
    is_cloud = st.session_state.cloud_ready and sb is not None
    user_id = user["id"]

    graph_mode = st.session_state.graph_mode
    mode_cfg = GRAPH_MODES[graph_mode]
    ann_table = mode_cfg["ann_table"]

    st.markdown(f"## {graph_mode} — Annotation Overview")

    # Get all source files
    cloud_files = cached_get_cloud_source_files(sb)
    if not cloud_files:
        st.info("No data uploaded yet. Go to the Annotator tab and upload a JSONL file.")
        return

    # File filter
    selected_files = st.multiselect(
        "Filter by source file",
        cloud_files,
        default=cloud_files,
    )

    if not selected_files:
        st.info("Select at least one source file.")
        return

    # Gather all chunks and annotations
    all_rows = []
    for sf in selected_files:
        cloud_chunks = cached_load_chunks_from_cloud(sb, sf)
        anns = cached_load_all_annotations(sb, sf, ann_table=ann_table)

        for cc in cloud_chunks:
            ck = cc["chunk_key"]
            raw_anns = anns.get(ck, {})
            chunk_anns = {_resolve_user_id(uid): a for uid, a in raw_anns.items()}
            # Collect all completed annotations
            completed = {
                uid: a for uid, a in chunk_anns.items()
                if a.get("annotation_completed")
            }
            n = len(completed)
            # Collect annotator IDs for display
            annotator_ids = list(completed.keys())

            all_rows.append({
                "source_file": sf,
                "doc_title": cc.get("doc_title", ""),
                "chunk_id": cc.get("chunk_id", ""),
                "chunk_key": ck,
                "chunk_text": cc.get("chunk_text", "")[:150] + "...",
                "status": f"✓ {n}/4" if n > 0 else "—",
                "annotator_ids": annotator_ids,
                "num_annotators": n,
                "_all_anns": chunk_anns,
                "_completed": completed,
                "_full_text": cc.get("chunk_text", ""),
                "_events": cc.get("events_json", []),
                "_entities": cc.get("entities_json", []),
                "_metadata": cc.get("metadata", {}),
            })

    # Annotator display — user IDs are already readable names
    for r in all_rows:
        r["annotators_display"] = ", ".join(r["annotator_ids"]) or "—"

    # Summary stats
    total = len(all_rows)
    done = sum(1 for r in all_rows if r["num_annotators"] > 0)
    full = sum(1 for r in all_rows if r["num_annotators"] >= 2)  # 2+ annotators
    st.markdown(
        f"**{done} / {total} chunks** have at least 1 annotation · "
        f"**{full}** have 2+ annotations"
    )
    st.progress(done / total if total else 0)

    # Status filter
    status_filter = st.radio(
        "Show", ["All", "Has annotations", "Needs more annotators", "No annotations"],
        horizontal=True,
    )

    filtered = all_rows
    if status_filter == "Has annotations":
        filtered = [r for r in all_rows if r["num_annotators"] > 0]
    elif status_filter == "Needs more annotators":
        filtered = [r for r in all_rows if 0 < r["num_annotators"] < 4]
    elif status_filter == "No annotations":
        filtered = [r for r in all_rows if r["num_annotators"] == 0]

    # Display table
    import pandas as pd
    display_cols = ["source_file", "doc_title", "chunk_id", "status", "annotators_display"]
    df = pd.DataFrame(filtered)[display_cols] if filtered else pd.DataFrame()
    if not df.empty:
        df.columns = ["File", "Document", "Chunk", "Status", "Annotators"]

    st.dataframe(
        df,
        use_container_width=True,
        height=min(600, 35 * len(filtered) + 40),
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "Annotators": st.column_config.TextColumn(width="medium"),
        },
    )

    st.divider()

    # ------- EXPORT -------
    st.markdown("### Export")

    jsonl_scope = st.radio(
        "JSONL export scope",
        ["All chunks", "Annotated only"],
        horizontal=True,
        help="Choose whether to include every chunk or only those that have been annotated.",
    )
    include_all = jsonl_scope == "All chunks"

    # Build JSONL export: group by source_file and doc_title, reconstruct documents
    import io
    export_docs: dict[str, dict] = {}
    export_chunk_count = 0
    for r in all_rows:
        # Add all annotations for this chunk
        annotations_list = []
        for uid, a in r["_all_anns"].items():
            if a.get("annotation_completed"):
                annotations_list.append({
                    "user_id": uid,
                    "human_summary": a.get("human_summary", ""),
                    "alignment_label": a.get("alignment_label", ""),
                    "annotation_reasoning": a.get("annotation_reasoning", ""),
                    "error_tags": a.get("error_tags", []),
                    "annotation_completed": True,
                })

        # If "annotated only", skip chunks with no annotations
        if not include_all and not annotations_list:
            continue

        doc_key = f"{r['source_file']}::{r['doc_title']}"
        if doc_key not in export_docs:
            export_docs[doc_key] = {
                "doc_id": r["doc_title"],
                "source_file": r["source_file"],
                "chunks": [],
            }

        chunk_export = {
            "chunk_id": r["chunk_id"],
            "chunk_key": r["chunk_key"],
            "text": r["_full_text"],
            "events": r["_events"],
            "entities": r["_entities"],
            "metadata": r["_metadata"],
            "annotations": annotations_list,
            "annotation_status": "annotated" if annotations_list else "unannotated",
        }

        export_docs[doc_key]["chunks"].append(chunk_export)
        export_chunk_count += 1

    # Build JSONL string
    export_buffer = io.BytesIO()
    for doc in export_docs.values():
        doc["num_chunks"] = len(doc["chunks"])
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        export_buffer.write(line.encode("utf-8"))
    export_buffer.seek(0)

    scope_label = "all" if include_all else "annotated"
    st.download_button(
        label=f"Download {scope_label} chunks ({export_chunk_count} chunks, JSONL)",
        data=export_buffer,
        file_name=f"annotated_chunks_export_{scope_label}.jsonl",
        mime="application/jsonl",
        use_container_width=True,
        type="primary",
    )

    # Also offer CSV export of the summary table
    if not df.empty:
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="Download Summary Table (CSV)",
            data=csv_buffer,
            file_name="annotation_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ===================================================================
# ROUTER
# ===================================================================
if st.session_state.user is None:
    show_login_page()
else:
    # Always show graph mode selector in sidebar (visible on all tabs)
    with st.sidebar:
        st.title("⚖️ Annotator")
        st.caption(f"Signed in as **{st.session_state.user['name']}**")

        if st.button("Switch User"):
            st.session_state.user = None
            st.rerun()

        st.divider()

        mode_options = list(GRAPH_MODES.keys())
        prev_mode = st.session_state.graph_mode
        graph_mode = st.radio(
            "Graph Type",
            mode_options,
            index=mode_options.index(st.session_state.graph_mode),
            horizontal=True,
            key="graph_mode_selector",
        )
        if graph_mode != prev_mode:
            st.session_state.graph_mode = graph_mode
            st.session_state.current_idx = 0
            st.rerun()

        st.divider()

    tab1, tab2 = st.tabs(["Annotator", "Overview & Export"])
    with tab1:
        show_annotator()
    with tab2:
        show_overview()
