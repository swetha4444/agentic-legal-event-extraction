import re
import spacy
from typing import List, Dict, Optional
import os

# Try importing litellm, handle if missing
try:
    from litellm import completion
except ImportError:
    completion = None

class HybridFactExtractor:
    def __init__(self, model_name: str = "gpt-4o-mini", use_llm: bool = True):
        """
        Initialize the extractor.
        Args:
            model_name: LiteLLM model identifier.
            use_llm: If False, skip all LLM calls and use NER-only anonymization.
        """
        self.model_name = model_name
        self.use_llm = use_llm and (completion is not None)
        
        if not self.use_llm:
            print("Running in NLP-only mode (no LLM calls).")

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_facts_section(self, text: str) -> str:
        """
        Extracts the 'Facts' or 'Background' section using heuristics.
        Falls back to returning the whole text if no clear section is found.
        """
        # Strategy 1: Look for "Facts" / "Background" headers (Generic)
        headers = [
            r"(?i)\n\s*(?:I\.|II\.|III\.)?\s*(?:factual )?(?:background|history|facts)\s*\n",
            r"(?i)\n\s*relevant facts\s*\n"
        ]
        
        for pattern in headers:
            match = re.search(pattern, text)
            if match:
                return text[match.end():].strip()

        # Strategy 2: Look for "MEMORANDUM" or "OPINION" to strip caption
        start_markers = [
            r"(?i)\n\s*memorandum\s*",
            r"(?i)\n\s*opinion\s*",
            r"(?i)\n\s*decision\s*"
        ]
        
        for pattern in start_markers:
            match = re.search(pattern, text)
            if match:
                clean_body = text[match.end():].strip()
                discussion_match = re.search(r"(?i)\n\s*(?:discussion|analysis)\s*\n", clean_body)
                if discussion_match:
                     return clean_body[:discussion_match.start()].strip()
                return clean_body

        return text

    def split_sentences(self, text: str) -> List[str]:
        """Uses Spacy to split text into sentences."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

    def build_entity_map(self, full_text: str) -> Dict[str, str]:
        """
        Scans the full case text and builds a consistent entity map.
        Returns a dict like {"John Smith": "[PERSON_1]", "Acme Corp": "[ORG_1]"}.
        Entities are sorted longest-first to avoid partial replacement issues.
        """
        doc = self.nlp(full_text)
        
        # Track entities by type with counters
        type_counters = {"PERSON": 0, "ORG": 0, "GPE": 0}
        entity_map = {}
        
        for ent in doc.ents:
            if ent.label_ in type_counters and ent.text.strip() not in entity_map:
                # Skip very short entities (likely noise)
                if len(ent.text.strip()) < 2:
                    continue
                type_counters[ent.label_] += 1
                tag = f"[{ent.label_}_{type_counters[ent.label_]}]"
                entity_map[ent.text.strip()] = tag
        
        return entity_map

    def anonymize_entities(self, text: str, entity_map: Dict[str, str] = None, 
                           case_context: Dict = None) -> str:
        """
        Anonymizes entities in text.
        - NLP mode: Uses pre-built entity_map for consistent replacement.
        - LLM mode: Sends to LLM for intelligent role mapping.
        """
        if not entity_map:
            entity_map = {}
        
        # ---- NLP-Only Path (Free) ----
        if not self.use_llm:
            anonymized = text
            # Sort by length descending to replace "John Smith" before "John"
            for name in sorted(entity_map.keys(), key=len, reverse=True):
                anonymized = anonymized.replace(name, entity_map[name])
            return anonymized
        
        # ---- LLM Path (Paid) ----
        doc = self.nlp(text)
        people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        if not people:
            return text

        unique_people = list(set(people))
        
        prompt = f"""
        You are a legal data assistant. 
        Context: Case involved {case_context.get('case_name', 'Unknown Case')}
        Text: "{text}"
        
        Identify the legal role for each person: {', '.join(unique_people)}.
        Roles: Plaintiff, Defendant, Judge, Witness, Appellant, Appellee, etc.
        
        Rewrite the text replacing names with roles (e.g. "Smith" -> "Plaintiff").
        Keep meaning exactly the same. Only output the rewritten text.
        """

        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            # Fallback to NLP-only if LLM fails
            anonymized = text
            for name in sorted(entity_map.keys(), key=len, reverse=True):
                anonymized = anonymized.replace(name, entity_map[name])
            return anonymized
