# Legal Event Graph Annotation App

A Streamlit-based **collaborative** annotation tool for human evaluation of event graphs extracted from legal text chunks. Annotations sync through Supabase so the whole team shares progress in real time.

## How It Works

1. One team member uploads the JSONL data to the cloud (one click in the sidebar).
2. Everyone signs in with GitHub and sees the same chunk list.
3. When you annotate chunks 1-5, your teammates immediately see those as done and can pick up from chunk 6.
4. A "Jump to Next Unannotated" button takes you straight to the first chunk nobody has touched.
5. An activity feed in the sidebar shows who annotated what.

## Quick Start (for teammates)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd agentic-legal-event-extraction/annotation_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. Sign in with GitHub on the login page
```

That's it. The app reads Supabase credentials from `annotation_dashboard/config.js` automatically.

## First-Time Setup (admin only)

If you haven't set up Supabase yet:

1. Create a free project at [supabase.com](https://supabase.com).
2. In the SQL Editor, run `annotation_dashboard/schema.sql` (if using both tools).
3. Then run `annotation_app/event_graph_schema.sql` to create the event annotation tables.
4. Enable GitHub as an OAuth provider under Authentication > Providers > GitHub.
5. Add your redirect URLs (e.g. `http://localhost:8501` for local dev).
6. Update `annotation_dashboard/config.js` with your Supabase URL and anon key.

## Features

- **Cloud Sync**: Annotations stored in Supabase, visible to all teammates instantly.
- **GitHub OAuth**: Sign in with your GitHub account.
- **Side-by-Side Review**: Compare chunk text and extracted event graphs.
- **Smart Navigation**: Filter by All / Unannotated / Annotated / My Unannotated. Jump to next unannotated chunk.
- **Activity Feed**: See who annotated what in real time.
- **View Teammate Annotations**: Expand to see how others labeled the same chunk (inter-annotator comparison).
- **Local Fallback**: Works without Supabase for solo offline annotation.
- **Progress Tracking**: Per-annotator breakdown of completion stats.

## Annotation Workflow

1. Sign in with GitHub (or use local mode for solo work).
2. Select a source file in the sidebar.
3. If the file hasn't been uploaded to the cloud yet, click "Upload" (one-time).
4. Use "Jump to Next Unannotated" to find your next chunk.
5. Review the text and extracted events side by side.
6. Fill in your annotation: summary, alignment label, reasoning, error tags.
7. Click "Save & Next" to commit and move on.

## Project Structure

- `app.py` — Main Streamlit UI with login, annotation, and navigation.
- `utils.py` — Data loading from local JSONL files, field detection.
- `supabase_client.py` — Cloud sync layer (auth, CRUD, activity feed).
- `event_graph_schema.sql` — Supabase tables for event graph annotations.
- `requirements.txt` — Python dependencies.

## Environment Variables (optional)

Instead of relying on `config.js`, you can set:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key"
export STREAMLIT_OAUTH_REDIRECT="http://localhost:8501"
```
