# app/utils_text.py

import string
from Levenshtein import distance as levenshtein_distance

# Define synonym mappings
SYNONYMS = {
    "tv": {"television"},
    "sofa": {"couch"},
    "cell phone": {"phone", "mobile"},
    "car": {"auto", "automobile"},
    "person": {"people", "man", "woman"},
}

def normalize(word):
    return word.strip(string.punctuation).lower()

def unify_synonym(word: str) -> str:
    """Define synonym mappings"""
    for main_word, syns in SYNONYMS.items():
        if word == main_word or word in syns:
            return main_word
    return word

def fuzzy_match(word: str, correct_words: set, max_dist=1): 
    """Define synonym mappings"""
    for cw in correct_words:
        if levenshtein_distance(word, cw) <= max_dist:
            return cw
    return None

def fuzzy_process_words(recognized_words, correct_words, max_dist=1):
    """Process the recognition results for normalization + synonym normalization + fuzzy matching"""
    matched = set()
    norm_correct = set(normalize(w) for w in correct_words)
    for rw in recognized_words:
        rw_norm = normalize(rw)
        rw_syn = unify_synonym(rw_norm)
        fm = fuzzy_match(rw_syn, norm_correct, max_dist=max_dist)
        if fm:
            matched.add(fm)
    return matched