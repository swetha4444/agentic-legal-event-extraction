
import sys
print(f"Python version: {sys.version}")
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
    import spacy
    print(f"Spacy version: {spacy.__version__}")
    nlp = spacy.load("en_core_web_sm")
    print("Spacy model loaded successfully")
except Exception as e:
    print(f"Error: {e}")
