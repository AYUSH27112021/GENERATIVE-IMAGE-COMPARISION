# from keybert import KeyBERT

# doc = """
#          A picture of areoplane in a thunderstrom.
#       """
# kw_model = KeyBERT()
# keywords = kw_model.extract_keywords(doc)

# print(keywords)
# combined_words=kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english')
# print(combined_words)
from keybert import KeyBERT
from typing import List

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

# print(extract_only_keywords("A picture of areoplane in a thunderstrom."))
# print(extract_keywords_with_scores("A picture of areoplane in a thunderstrom."))