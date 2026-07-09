"""Self-contained feature extraction for the Fuzzy Logic Engine and ML-prior
tool. Deliberately independent of src/extract_classical_features.py and
backend/main.py, which are under active, unstable revision elsewhere in
this project — this module owns its own small, stable 8-feature contract.
"""

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}


def load_nrc_lexicon(filepath):
    emotions_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                word_sense, emotion, score = parts
                word = word_sense.split("--")[0]
                if int(score) == 1:
                    emotions_dict.setdefault(word, set()).add(emotion)
    return emotions_dict


def extract_features_for_text(text, nlp, nrc_dict):
    doc = nlp(text)
    lemmas = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct]
    word_count = len(doc)

    positive_hits = sum(1 for lemma in lemmas if lemma in nrc_dict and "positive" in nrc_dict[lemma])
    negative_hits = sum(1 for lemma in lemmas if lemma in nrc_dict and "negative" in nrc_dict[lemma])
    total_lemmas = len(lemmas)
    positive_ratio = positive_hits / total_lemmas if total_lemmas > 0 else 0.0
    negative_ratio = negative_hits / total_lemmas if total_lemmas > 0 else 0.0
    semantic_polarity = (positive_ratio - negative_ratio) / (positive_ratio + negative_ratio + 1e-5)

    exclamation_count = text.count("!")
    question_count = text.count("?")
    verbs = sum(1 for t in doc if t.pos_ == "VERB")
    pronouns_sg = sum(1 for t in doc if t.pos_ == "PRON" and t.text.lower() in FIRST_PERSON_SINGULAR)
    pronouns_pl = sum(1 for t in doc if t.pos_ == "PRON" and t.text.lower() in FIRST_PERSON_PLURAL)

    return {
        "positive": positive_ratio,
        "negative": negative_ratio,
        "semantic_polarity": semantic_polarity,
        "behav_exclamation_ratio": exclamation_count / word_count if word_count > 0 else 0.0,
        "behav_question_ratio": question_count / word_count if word_count > 0 else 0.0,
        "behav_verb_ratio": verbs / word_count if word_count > 0 else 0.0,
        "behav_1st_sg_pronoun_ratio": pronouns_sg / word_count if word_count > 0 else 0.0,
        "behav_1st_pl_pronoun_ratio": pronouns_pl / word_count if word_count > 0 else 0.0,
    }
