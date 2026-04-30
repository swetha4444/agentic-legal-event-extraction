
"""
Fetch Complaints Script
Searches CourtListener RECAP for "Complaint" documents in employment cases,
downloads the PDFs, extracts text, and saves to JSONL.
"""

import os
import sys
import json
import argparse
import requests
import time
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from pypdf import PdfReader
from io import BytesIO

# Add src to path if needed
sys.path.append(os.getcwd())

load_dotenv()

COURTLISTENER_API_TOKEN = os.environ.get("COURTLISTENER_API_TOKEN")
SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
STORAGE_BASE_URL = "https://storage.courtlistener.com/"

def get_headers():
    if not COURTLISTENER_API_TOKEN:
        raise ValueError("COURTLISTENER_API_TOKEN not found in .env")
    return {"Authorization": f"Token {COURTLISTENER_API_TOKEN}"}

def search_complaints(query: str, limit: int = 10) -> List[Dict]:
    """
    Search for complaints using RECAP Search API.
    Returns list of document metadata with filepath_local.
    """
    print(f"Searching for: {query} (Limit: {limit})...")
    
    headers = get_headers()
    # Removed docket_id from fields as it might be causing issues
    params = {
        "type": "r",
        "q": f'{query} description:("complaint")',
        "district_only": "true",
        "fields": "id,caseName,court,dateFiled,filepath_local,filepath_ia,absolute_url,description",
        "order_by": "dateFiled desc",
    }
    
    results = []
    url = SEARCH_URL
    
    while url and len(results) < limit:
        try:
            print(f"GET {url}...", flush=True)
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            print(f"Response: {resp.status_code}", flush=True)
            resp.raise_for_status()
            data = resp.json()
            
            batch = data.get("results", [])
            
            for item in batch:
                # API returns Dockets, containing 'recap_documents' list
                docs = item.get("recap_documents") or []
                for doc in docs:
                    filepath = doc.get("filepath_local")
                    
                    # We are searching for 'complaint', but the Docket might contain other docs
                    # The query description:("complaint") filters Dockets that HAVE a complaint.
                    # Use description check again to be sure we pick the COMPLAINT document, 
                    # not an order or exhibit, unless we want all docs in that docket.
                    # The user wants "Complaints".
                    desc = (doc.get("description") or "").lower()
                    if filepath and "complaint" in desc:
                        # Found a complaint PDF!
                        results.append({
                            "docket_id": item.get("docket_id") or item.get("id"), # API uses id for dockets
                            "caseName": item.get("caseName"),
                            "dateFiled": item.get("dateFiled"),
                            "court": item.get("court"),
                            "filepath_local": filepath,
                            "doc_id": doc.get("id"),
                            "doc_description": doc.get("description")
                        })
                        if len(results) >= limit:
                            break
                if len(results) >= limit:
                    break
            
            url = data.get("next")
            params = None # args are in next url
            
            # Be nice to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error searching: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break
            
    return results

def download_and_extract_pdf(filepath: str) -> str:
    """
    Download PDF from CourtListener storage and extract text.
    Includes rate limit handling (429).
    """
    url = f"{STORAGE_BASE_URL}{filepath}"
    max_retries = 3
    backoff = 2
    
    for attempt in range(max_retries):
        try:
            # Be nice to the API - Sleep before every request
            time.sleep(2)
            
            resp = requests.get(url, timeout=30)
            
            if resp.status_code == 429:
                wait = backoff * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
                
            resp.raise_for_status()
            
            # Extract text from PDF bytes
            with BytesIO(resp.content) as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text.strip()
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error downloading/extracting {url}: {e}", flush=True)
            else:
                print(f"Error {e}. Retrying...", flush=True)
                
    return ""

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    parser = argparse.ArgumentParser(description="Fetch and extract complaints")
    parser.add_argument("--query", default="employment", help="Search query (default: employment)")
    parser.add_argument("--limit", type=int, default=10, help="Max documents to fetch")
    parser.add_argument("--output", default="data/processed/complaints.jsonl", help="Output JSONL file")
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Search
    items = search_complaints(args.query, args.limit)
    print(f"Found {len(items)} complaints with downloadable files.")
    
    if not items:
        return

    # 2. Download & Extract
    print(f"Processing {len(items)} items...", flush=True)
    with open(output_path, "w") as f:
        for i, item in enumerate(items):
            filepath = item.get("filepath_local")
            if not filepath:
                continue
            
            print(f"[{i+1}/{len(items)}] Downloading: {STORAGE_BASE_URL}{filepath}", flush=True)
            text = download_and_extract_pdf(filepath)
            
            if not text:
                print(f"Warning: No text extracted for {item.get('id')}", flush=True)
                continue
            
            print(f"Successfully extracted {len(text)} characters.", flush=True)
            
            # Create record
            record = {
                "case_id": item.get("docket_id"), # Using docket_id as case_id
                "case_name": item.get("caseName"),
                "date_filed": item.get("dateFiled"),
                "court": item.get("court"),
                "document_url": f"{STORAGE_BASE_URL}{filepath}",
                "document_text": text, # Compatible with pipeline
                "original_data": item
            }
            
            f.write(json.dumps(record) + "\n")
            
    print(f"\nDone! Saved to {args.output}")

if __name__ == "__main__":
    main()
