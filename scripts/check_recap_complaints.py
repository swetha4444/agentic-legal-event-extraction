
import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.getcwd())
from src.data.courtlistener_client import search_recap

load_dotenv()

print("Searching for recent employment complaints...")

# Direct API call to test 'fields' parameter
import requests
token = os.environ.get("COURTLISTENER_API_TOKEN")
headers = {"Authorization": f"Token {token}"}
url = "https://www.courtlistener.com/api/rest/v4/search/"
params = {
    "type": "r",
    "q": 'employment description:("complaint")',
    "district_only": "true",
    "fields": "id,caseName,court,dateFiled,filepath_local,filepath_ia,absolute_url,description",
    "order_by": "dateFiled desc"
}

print(f"Querying {url} with fields={params['fields']}...")
resp = requests.get(url, params=params, headers=headers)
if resp.status_code != 200:
    print(f"Error: {resp.status_code} {resp.text}")
    sys.exit(1)

results = resp.json().get('results', [])
for i, item in enumerate(results[:3]):
    print(f"\n--- Result {i+1} ---")
    print(json.dumps(item, indent=2))
