# GitHub Collaboration Guide

Since you've chosen to run the app via GitHub, follow this simple workflow to keepeveryone's annotations in sync within the single shared JSONL file.

## 👥 Team Workflow

To avoid technical conflicts, each teammate should follow these 3 steps:

1.  **Sync (Before starting)**: 
    Run `git pull` to get the latest annotations from your teammates.
2.  **Annotate**: 
    Run `streamlit run app.py`, do your work, and click **SAVE ANNOTATION**. Your work is saved directly to `human_annotated_...jsonl`.
3.  **Share (When finishing)**: 
    Commit and push the updated JSONL file:
    ```bash
    git add data/outputs/*.jsonl
    git commit -m "Vilma: annotated 10 chunks in long_docs"
    git push
    ```

---

## 🛠 Setup for Teammates

Send these instructions to your teammates to get them running in 2 minutes:

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/your-username/agentic-legal-event-extraction.git
    cd agentic-legal-event-extraction/annotation_app
    ```
2.  **Install Dependencies**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Launch**:
    ```bash
    streamlit run app.py
    ```

---

## 💡 Pro-Tips for GitHub Collaboration
- **Don't Overlap**: Since you're using a single file, try to ensure different teammates work on different documents or ranges of chunks to make Git merges simpler.
- **Commit Often**: Push your work after a session so others have the most up-to-date progress.
- **Atomic Saves**: The app is designed to update specific lines in the JSONL, so Git handles merges surprisingly well even if two people commit different chunks simultaneously!
