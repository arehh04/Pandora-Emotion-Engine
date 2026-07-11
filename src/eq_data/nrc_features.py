"""Self-contained NRC Word-Emotion-Association-Lexicon loader and
text-feature scorer for the EQ pivot. Deliberately self-contained -- does
NOT import src/extract_classical_features.py's load_nrc_lexicon, since that
module is the separate, actively-changing legacy classical-ML pipeline this
project has repeatedly chosen to stay decoupled from (see CLAUDE.md).

Score weights (50/49 split between emotion-word density and positivity
skew) are a defensible starting point pending literature citation, same
citation_needed convention as src/eq_data/proxy_labels.py's trait weights.
"""


def load_nrc_lexicon(path):
    lexicon = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word_sense, emotion, flag = parts
            if int(flag) != 1:
                continue
            word = word_sense.split("--")[0]
            lexicon.setdefault(word, set()).add(emotion)
    return lexicon


def compute_nrc_text_score(text, nrc_lexicon):
    words = text.lower().split()
    if not words:
        return 0.0

    total_words = len(words)
    emotion_word_count = 0
    positive_count = 0
    negative_count = 0
    for word in words:
        emotions = nrc_lexicon.get(word)
        if emotions:
            emotion_word_count += 1
            if "positive" in emotions:
                positive_count += 1
            if "negative" in emotions:
                negative_count += 1

    emotion_word_density = emotion_word_count / total_words
    positive_ratio = positive_count / (positive_count + negative_count + 1)

    return min(99.0, (emotion_word_density * 50.0) + (positive_ratio * 49.0))
