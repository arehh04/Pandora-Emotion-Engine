"""
Advanced preprocessing pipeline.
Adds: contraction expansion, negation tagging, language detection, Flesch score.
"""
import re
import unicodedata
import pandas as pd
import spacy


# ── Contraction map ────────────────────────────────────────────────────────────
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "I'm": "I am", "I've": "I have", "I'll": "I will", "I'd": "I would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "let's": "let us", "who's": "who is", "what's": "what is",
    "n't": " not",
}

_CONTRACTION_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def expand_contractions(text: str) -> str:
    def _replace(match):
        token = match.group(0)
        return CONTRACTIONS.get(token, CONTRACTIONS.get(token.lower(), token))
    return _CONTRACTION_RE.sub(_replace, text)


# ── Negation tagger ────────────────────────────────────────────────────────────
_NEGATION_WORDS = {"not", "no", "never", "neither", "nobody", "nothing",
                   "nowhere", "nor", "cannot", "n't"}
_CLAUSE_PUNCT   = re.compile(r'[,;:.!?]')


def tag_negations(tokens: list[str]) -> list[str]:
    """
    Prefix each token in a negation scope with NOT_.
    Scope ends at clause-boundary punctuation or after 5 tokens.
    """
    result   = []
    negating = False
    neg_count = 0
    for tok in tokens:
        if tok.lower() in _NEGATION_WORDS:
            negating  = True
            neg_count = 0
            result.append(tok)
            continue
        if negating:
            if _CLAUSE_PUNCT.search(tok) or neg_count >= 5:
                negating  = False
                neg_count = 0
                result.append(tok)
            else:
                result.append(f"NOT_{tok}")
                neg_count += 1
        else:
            result.append(tok)
    return result


# ── Language detection (lightweight heuristic) ─────────────────────────────────
_COMMON_ENGLISH = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
}


def is_english(text: str, threshold: float = 0.15) -> bool:
    """Simple token-overlap heuristic. Returns True if likely English."""
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    if not tokens:
        return False
    overlap = sum(1 for t in tokens if t in _COMMON_ENGLISH)
    return (overlap / len(tokens)) >= threshold


# ── Flesch Reading Ease ────────────────────────────────────────────────────────
def count_syllables(word: str) -> int:
    """Approximate syllable count using vowel-group heuristic."""
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    vowels  = re.findall(r'[aeiouy]+', word)
    count   = len(vowels)
    if word.endswith('e'):
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease score.
    206.835 − 1.015*(words/sentences) − 84.6*(syllables/words)
    Higher = easier to read. Typical range: 0–100.
    """
    sentences = max(1, len(re.split(r'[.!?]+', text)))
    words     = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    asl       = len(words) / sentences            # avg sentence length
    asw       = syllables / len(words)            # avg syllables per word
    score     = 206.835 - 1.015 * asl - 84.6 * asw
    return round(max(0.0, min(100.0, score)), 2)


# ── Full pipeline ──────────────────────────────────────────────────────────────
def clean_advanced(text: str, nlp=None) -> dict:
    """
    Run the full advanced cleaning pipeline on a single text.
    Returns a dict with cleaned text variants and the Flesch score.
    """
    # 1. Unicode normalise
    text = unicodedata.normalize("NFKC", text)
    # 2. Strip HTML/URLs
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # 3. Expand contractions
    expanded = expand_contractions(text)
    # 4. Language check
    if not is_english(expanded):
        return None                               # caller should drop row
    # 5. Flesch score on original (before lemmatization)
    flesch = flesch_reading_ease(expanded)
    # 6. Lowercase + tokenize for classical features
    clean  = re.sub(r'[^a-z\s]', ' ', expanded.lower())
    tokens = clean.split()
    # 7. Negation tagging
    neg_tokens = tag_negations(tokens)
    neg_text   = " ".join(neg_tokens)

    return {
        "text_raw":    text,          # keep original case for BERT
        "text_clean":  expanded,      # contraction-expanded for BERT input
        "text_neg":    neg_text,      # negation-tagged for TF-IDF / NRC
        "flesch":      flesch,
        "is_english":  True,
    }


def clean_dataframe(df: pd.DataFrame, text_col: str = "text",
                    target_col: str = "extraversion") -> pd.DataFrame:
    """Apply clean_advanced to every row, drop non-English rows."""
    records = []
    for _, row in df.iterrows():
        result = clean_advanced(str(row[text_col]))
        if result is None:
            continue
        result[target_col] = row[target_col]
        records.append(result)
    return pd.DataFrame(records)


if __name__ == "__main__":
    sample = "I don't like big parties, they're too loud and I can't focus."
    out = clean_advanced(sample)
    print("Raw:          ", out["text_raw"])
    print("Expanded:     ", out["text_clean"])
    print("Negation:     ", out["text_neg"])
    print("Flesch Score: ", out["flesch"])
