import json
import re

def heuristic_a(text):
    # Aggressive Regex for Facts/Background
    match = re.search(r'(?i)\n\s*(?:I\.|II\.|III\.)?\s*(?:factual )?(?:background|history|facts)\s*\n', text)
    if match:
        return f"MATCH: Found '{match.group(0).strip()}' at index {match.start()}"
    return "NO MATCH"

def heuristic_b(text):
    # Semantic Start: Look for "Memorandum" or "Opinion"
    match = re.search(r'(?i)\n\s*(?:memorandum|opinion)\s+', text)
    if match:
        return f"MATCH: Found '{match.group(0).strip()}' at index {match.start()}"
    return "NO MATCH"

def test_heuristics(file_path, limit=5):
    with open(file_path, 'r') as f:
        print(f"--- Testing Heuristics on {file_path} ---")
        for i, line in enumerate(f):
            if i >= limit: break
            try:
                data = json.loads(line)
                text = data.get('document_text', '')
                print(f"\n[Case {i}]")
                print(f"  Length: {len(text)}")
                print(f"  Heuristic A (Facts Header): {heuristic_a(text)}")
                print(f"  Heuristic B (Opinion Start): {heuristic_b(text)}")
                
                # Print a snippet from the middle to see what it looks like
                mid = len(text) // 2
                print(f"  Mid-text snippet: {text[mid:mid+100]}...")
            except:
                continue

if __name__ == "__main__":
    test_heuristics('data/processed/courtlistener_recap.jsonl')
