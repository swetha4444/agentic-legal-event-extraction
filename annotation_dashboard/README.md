# Annotation Dashboard

Cloud-backed annotation app for labeling sentences as fact or non-fact.

## Run locally

Option 1 (open directly):
- Open `annotation_dashboard/index.html` in your browser.

Option 2 (recommended static server):
```bash
cd /work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/annotation_dashboard
python -m http.server 8000
```
Then visit `http://localhost:8000`.

## Usage

1. Sign in first (GitHub OAuth via Supabase).
2. Choose a saved dataset, or upload a new CSV/JSONL file.
3. Select the text column and (optionally) enable **Split into sentences**.
4. Label each sentence as Fact or Non-Fact.
5. Resume later from the **Saved Datasets** list.
6. Export labeled data as structured JSON.

Notes:
- Saved datasets remain in cloud and can be reopened after future logins.
- Local autosave still works and is merged when resuming.
- Export is JSON-only and outputs one object per source document in original order.
- Each exported object keeps original fields and updates text fields to:
  - `document_text` is removed
  - `Extracted Facts`
  - `Extracted Non Facts`
- Even unannotated documents are included with empty parentheses.
- When sentence mode is enabled, extra fields are added: `sentence`, `sentence_index`, `sentence_count`, `source_row`.

## Free hosting (no backend required)

Pick any static hosting provider:

GitHub Pages
1. Create a repo and copy the `annotation_dashboard` contents to the repo root.
2. Enable GitHub Pages from the repo settings.

Cloudflare Pages
1. Create a new Pages project and connect your repo.
2. Use the root of the repo (or `/annotation_dashboard`) as the build output.
3. No build command is needed.

## Cloud backend (shared + GitHub login)

Provider choice: **Supabase (Postgres + Auth)**. It is free-tier friendly and supports GitHub OAuth.

1. Create a Supabase project.
2. Open the SQL editor and run `annotation_dashboard/schema.sql`.
3. Enable GitHub login:
   - Supabase Dashboard -> Authentication -> Providers -> GitHub.
   - Add your GitHub OAuth app client ID/secret.
4. Add redirect URLs:
   - Add your site URL (e.g. `http://localhost:8000`, your Cloudflare Pages URL).
5. Fill `annotation_dashboard/config.js` with your project URL and anon key.

Note: OAuth requires the app to be served over HTTP (not `file://`). Use `python -m http.server` during local development.

Once configured, the app will:
- Require sign-in for cloud sync.
- Store datasets, sentences, and per-user annotations.
- Show the latest label with annotator initials.
- Show saved-dataset progress and contributors.
- Show a per-dataset activity timeline (who labeled what and when).

### Hosting (recommended)

Use Cloudflare Pages (free):
1. Connect your repo.
2. Set the project root to `annotation_dashboard`.
3. No build step needed.
4. Add the deployed URL to Supabase Auth redirect URLs.

## Optional E2E tests (Playwright)

Requirements: Node.js + npm.

```bash
cd /work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/annotation_dashboard
npm install
npx playwright install
python -m http.server 8000
npm run test:e2e
```

The sample test uses `sample.csv` to validate the upload and annotation flow.
