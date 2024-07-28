from keybert import KeyBERT
from typing import List
import sys
import warnings
warnings.filterwarnings("ignore")

def extract_keywords_with_scores(doc: str) -> List[str]:
    kw_model = KeyBERT()
    keywords_with_scores = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english')
    return keywords_with_scores

def extract_only_keywords(doc: str) -> List[str]:
    kw_model = KeyBERT()
    keywords_with_scores = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english')
    keywords = [keyword for keyword, score in keywords_with_scores if score >= 0.1]
    return keywords

def extract_multi_keywords(doc: str,n: int) -> List[str]:
    kw_model = KeyBERT()
    keywords_with_scores = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, n), stop_words='english')
    keywords = [keyword for keyword, _ in keywords_with_scores]
    return keywords

def extract_multi_keywords_with_scores(doc: str,n: int) -> List[str]:
    kw_model = KeyBERT()
    keywords_with_scores = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, n), stop_words='english')
    return keywords_with_scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process a string using a minimal method for keyword extraction with BERT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "prompt", 
        type=str, 
        help="Prompt string to process with the image."
    )

    args = parser.parse_args()
    
    try:
        keywords = extract_only_keywords(args.image_path)
        print("Extracted keywords :",keywords)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
