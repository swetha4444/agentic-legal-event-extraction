# Installing Dependencies

The annotation app has two modes:

1. **Local Mode** (offline, no Supabase) — requires only `streamlit`
2. **Cloud Mode** (collaborative, with Supabase) — requires `streamlit` + `supabase`

## Quick Install (Local Mode Only)

```bash
pip install streamlit
```

Then run:
```bash
streamlit run app.py
```

You'll see a login page with two options. Choose **Local Mode** to annotate offline without Supabase.

---

## Full Install (Cloud Mode + Local Mode)

If `pip` can't reach PyPI (proxy issues), try these alternatives:

### Option A: Use Conda (Often works better with proxies)

```bash
conda install streamlit supabase
streamlit run app.py
```

### Option B: Install from a Different Mirror

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ streamlit supabase
streamlit run app.py
```

### Option C: Install Offline

Ask your IT team for a wheel file (`supabase-2.0.0-py3-none-any.whl`), then:

```bash
pip install ./supabase-2.0.0-py3-none-any.whl
pip install streamlit
streamlit run app.py
```

### Option D: Use Docker

If you have Docker installed:

```bash
docker run -it --rm -v $(pwd):/app -w /app python:3.10 bash
pip install streamlit supabase
streamlit run app.py --server.headless true
```

---

## Verify Installation

Once installed, test that everything works:

```bash
python3 -c "import streamlit; import supabase; print('✓ All imports OK')"
```

If you see `✓ All imports OK`, you're good to go!

---

## Troubleshooting

**Q: I got "ModuleNotFoundError: No module named 'supabase'"**
- For **Local Mode**: that's fine, just use Local Mode on the login page
- For **Cloud Mode**: try one of the alternative installation methods above

**Q: Conda isn't installed**
- Download from [conda.io](https://conda.io/miniconda.html)
- Or try one of the other options

**Q: Still can't install?**
- Run `streamlit run app.py` anyway
- Use **Local Mode** on the login page
- Once in Local Mode, annotations will save to your local JSONL files
- You can always upgrade to Cloud Mode later when Supabase is installed
