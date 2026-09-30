# Event Graph Annotator — Teammate Quick Start

## Access the App

Open the annotation app in your browser:

**https://agentic-legal-event-extraction-annotation.streamlit.app/**

(If the URL above doesn't work, ask Heo for the correct Streamlit Cloud link.)

## Sign In

1. Click **Sign in with GitHub**.
2. Authorize the app when GitHub asks. You'll be redirected back to the annotator.
3. That's it — you're in. Your GitHub identity is used to track who annotated what.

## How Annotation Works

The app shows legal document chunks one at a time. Each chunk contains the original text and the events/entities that were automatically extracted by the LLM pipeline.

**Your job:** Read the chunk, review the extracted events, and write a short human summary of what actually happened in the text. Then click **Save & Next** to move on.

Here's the typical flow:

1. **Read the chunk text** on the left side of the screen.
2. **Review the extracted events** shown below the text (if any).
3. **Write your summary** in the text box — a concise description of the key facts and events in the chunk.
4. Click **Save & Next** to submit your annotation and move to the next unannotated chunk.

The app automatically skips chunks that have already been annotated by someone on the team, so you'll always land on the next chunk that needs work. If Person A annotates chunks 1–5, you'll start at chunk 6.

## Uploading Data (First-Time Only)

If no data appears when you sign in, one team member needs to upload the JSONL files:

1. In the sidebar, use the **Upload JSONL** file uploader.
2. Select one or more `.jsonl` files from the `data/outputs/` folder (e.g., `chunks_only_1-5.jsonl`, `chunks_only_6-10.jsonl`, `chunks_only_11-15.jsonl`).
3. The chunks are synced to the cloud — everyone else will see them immediately.

This only needs to happen once. After that, all team members share the same pool of chunks.

## Overview & Export Tab

Switch to the **Overview & Export** tab to see:

- A table of all chunks across all files, showing which are annotated and by whom.
- Filter by source file or annotation status.
- A progress bar showing how far the team has gotten.
- **Download JSONL** — exports all chunks with their annotations in JSONL format.
- **Download CSV** — exports a summary table as CSV.

## Tips

- You can use the **← Prev / Next →** buttons to navigate between chunks manually.
- The sidebar shows your current progress and recent team activity.
- Your annotations are saved to the cloud instantly — no need to worry about losing work.
- If the app looks stuck, just refresh the page. Your session will be restored automatically.

## Need Help?

If you run into issues, reach out to Heo. Common fixes:

- **"Supabase not configured"** → The Streamlit secrets may need updating. Ask Heo.
- **No chunks showing up** → Someone needs to upload the JSONL files first (see above).
- **OAuth error** → Make sure you're authorizing with your GitHub account when prompted.
