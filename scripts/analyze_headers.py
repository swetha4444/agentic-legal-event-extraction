import json

def analyze_headers(file_path, limit=5):
    with open(file_path, 'r') as f:
        print(f"--- Deep Search in {file_path} ---")
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                data = json.loads(line)
                text = data.get('document_text', '')
                if not text: continue
                
                print(f"\n--- Case {i} ---")
                # Split into lines to find the header line
                doc_lines = text.split('\n')
                found = False
                for j, doc_line in enumerate(doc_lines):
                    clean_line = doc_line.strip()
                    # Check for short-ish lines with keywords
                    if 0 < len(clean_line) < 60 and any(k in clean_line.upper() for k in ['FACT', 'BACKGROUND', 'OPINION', 'DISCUSSION']):
                        print(f"  [Line {j}] '{clean_line}'")
                        found = True
                
                if not found:
                    print("  No obvious headers found.")
                    
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    analyze_headers('data/processed/courtlistener_recap.jsonl')
