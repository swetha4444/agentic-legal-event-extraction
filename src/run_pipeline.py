
import json
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv
from processing.fact_extractor import HybridFactExtractor

load_dotenv()  # Load environment variables from .env file

def run_pipeline(input_file: str, output_file: str, limit: int = None, 
                 model: str = "gpt-4o-mini", use_llm: bool = True):
    """
    Main pipeline function:
    1. Read JSONL input
    2. Extract Facts (Heuristic)
    3. Split Sentences (Spacy)
    4. Anonymize Entities (NER or LLM)
    5. Write to Output JSONL
    """
    
    mode_label = f"Hybrid (model: {model})" if use_llm else "NLP-only (no LLM)"
    print(f"Initializing Pipeline in {mode_label} mode")
    extractor = HybridFactExtractor(model_name=model, use_llm=use_llm)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        if limit:
            lines = lines[:limit]
            
        print(f"Processing {len(lines)} cases...")
        
        for line in tqdm(lines):
            try:
                case_data = json.loads(line)
                
                if not case_data.get('document_text'):
                    continue
                    
                # 1. Extract Facts Section
                facts_text = extractor.extract_facts_section(case_data['document_text'])
                
                # 2. Build entity map for this case (consistent across sentences)
                entity_map = extractor.build_entity_map(facts_text)
                
                # 3. Split Sentences
                sentences = extractor.split_sentences(facts_text)
                
                # 4. Anonymize each sentence
                context = {
                    "case_name": case_data.get("case_name", "Unknown"),
                    "docket": case_data.get("docket_number", "Unknown")
                }
                
                anonymized_sentences = []
                for sent in sentences:
                    anon_sent = extractor.anonymize_entities(
                        sent, entity_map=entity_map, case_context=context
                    )
                    anonymized_sentences.append(anon_sent)
                
                # 5. Construct Output
                output_obj = {
                    "case_id": case_data.get("case_id"),
                    "case_name": case_data.get("case_name"),
                    "facts_clean": anonymized_sentences,
                    "entity_map": entity_map,  # Show what was replaced
                    "num_sentences": len(sentences),
                    "original_facts_snippet": facts_text[:200] + "..."
                }
                
                outfile.write(json.dumps(output_obj) + "\n")
                count += 1
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing case: {e}")
                continue
                
    print(f"Pipeline complete. Processed {count} cases. Output: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Legal Event Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--limit", type=int, help="Limit number of cases (for testing)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LiteLLM model name")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM (NER-only anonymization)")
    
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output, args.limit, args.model, use_llm=not args.no_llm)
