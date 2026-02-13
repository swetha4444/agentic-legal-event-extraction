#!/bin/bash
# Verify the pipeline installation and run a sample test

PYTHON_CMD=".venv/bin/python"

echo "Checking environment..."
if [ ! -f "$PYTHON_CMD" ]; then
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

echo "Installing/Verifying dependencies..."
$PYTHON_CMD -m pip install spacy litellm tqdm python-dotenv
$PYTHON_CMD -m spacy download en_core_web_sm

echo "Running Verification Pipeline (NLP-only mode, Limit 5 cases)..."
$PYTHON_CMD src/run_pipeline.py \
    --input data/processed/courtlistener_recap.jsonl \
    --output data/processed/verification_sample.jsonl \
    --limit 5 \
    --no-llm

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Verification Successful! ==="
    echo "Output saved to data/processed/verification_sample.jsonl"
    echo ""
    echo "To run WITH LLM anonymization (requires API key):"
    echo "  $PYTHON_CMD src/run_pipeline.py --input data/processed/courtlistener_recap.jsonl --output data/processed/clean_facts.jsonl --limit 5"
else
    echo "Verification Failed."
fi
