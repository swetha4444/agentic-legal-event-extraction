# Supabase Setup for Event Graph Annotation

This guide walks you through setting up the cloud backend so your team can collaborate on annotations.

## Step 1: Go to Your Supabase Project

1. Open [app.supabase.com](https://app.supabase.com)
2. Log in with your account
3. Click on the project you've been using (the same one as the Fact/Non-Fact dashboard)
   - It should be the project whose URL is in `annotation_dashboard/config.js`

## Step 2: Open the SQL Editor

On the left sidebar, you should see:
- **Home**
- **SQL Editor** ← Click this
- **Database**
- **Authentication**
- etc.

Click **SQL Editor**.

## Step 3: Create a New Query

Once you're in the SQL Editor, you should see a big text area. There may be a button or link to create a new query. Look for:
- A `+` button or "New Query" button at the top
- Or just click in the blank SQL editor area

## Step 4: Copy & Paste the Schema

Open the file `annotation_app/event_graph_schema.sql` from your project.

Copy **all** of the SQL code (it's about 4,800 characters).

Paste it into the SQL Editor in Supabase.

## Step 5: Run the Query

You should see a **Run** button (looks like a play button ▶️) at the bottom right of the SQL editor, or sometimes at the top.

Click it.

**Wait 10-20 seconds.** You should see a green success message like:
```
Query executed successfully
```

If you see an error, scroll down to see the error message and let me know.

## Step 6: Verify the Tables Were Created

On the left sidebar, click **Database** → **Tables**.

You should now see these new tables:
- `event_chunks`
- `event_annotations`
- `event_activity`

(You may also see the existing tables from the dashboard: `profiles`, `datasets`, `sentences`, etc.)

## Done! 🎉

Your cloud backend is ready. Now teammates can:

1. `streamlit run app.py`
2. Sign in with GitHub
3. One person uploads the JSONL data (one click in the sidebar)
4. Everyone else sees the same chunks and can start annotating

---

## Troubleshooting

**Q: I got an error like "relation already exists"**
- This means the tables were already created. That's fine! Just skip this step.

**Q: I don't see the tables in the Database section**
- Try refreshing the page (Ctrl+R or Cmd+R)
- Make sure you clicked Run on the SQL query

**Q: The schema.sql file from the dashboard didn't work**
- That's for the Fact/Non-Fact dashboard. Use `event_graph_schema.sql` (the new file) instead.

**Q: Can I run both schema files?**
- Yes! If you're using both tools, you can run `schema.sql` first (for the dashboard), then `event_graph_schema.sql` (for the event graph annotator). They don't conflict.
