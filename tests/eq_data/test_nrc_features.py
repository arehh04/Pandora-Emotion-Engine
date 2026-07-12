import pytest

from src.eq_data.nrc_features import compute_nrc_text_score, load_nrc_lexicon


def _write_fixture_lexicon(path):
    # Mirrors the real file's tab-separated (word--sense, emotion, flag) format.
    rows = [
        ("happy--joyful", "positive", 1), ("happy--joyful", "joy", 1),
        ("happy--joyful", "negative", 0), ("happy--joyful", "sadness", 0),
        ("wonderful--great", "positive", 1), ("wonderful--great", "joy", 1),
        ("wonderful--great", "negative", 0),
        ("sad--unhappy", "negative", 1), ("sad--unhappy", "sadness", 1),
        ("sad--unhappy", "positive", 0), ("sad--unhappy", "joy", 0),
    ]
    path.write_text("\n".join(f"{w}\t{e}\t{f}" for w, e, f in rows), encoding="utf-8")


def test_load_nrc_lexicon_parses_word_before_double_dash(tmp_path):
    fixture_path = tmp_path / "lexicon.txt"
    _write_fixture_lexicon(fixture_path)

    lexicon = load_nrc_lexicon(str(fixture_path))

    assert lexicon["happy"] == {"positive", "joy"}
    assert lexicon["wonderful"] == {"positive", "joy"}
    assert lexicon["sad"] == {"negative", "sadness"}


def test_load_nrc_lexicon_omits_words_with_no_flagged_emotions(tmp_path):
    fixture_path = tmp_path / "lexicon.txt"
    fixture_path.write_text("neutral--plain\tpositive\t0\nneutral--plain\tnegative\t0\n", encoding="utf-8")

    lexicon = load_nrc_lexicon(str(fixture_path))

    assert "neutral" not in lexicon


def test_compute_nrc_text_score_matches_hand_computed_value():
    lexicon = {
        "happy": {"positive", "joy"},
        "wonderful": {"positive", "joy"},
        "sad": {"negative", "sadness"},
    }
    text = "happy wonderful sad neutral word"

    # 5 words total; 3 are emotion words (happy, wonderful, sad) -> density = 3/5 = 0.6
    # positive_count=2 (happy, wonderful), negative_count=1 (sad)
    # positive_ratio = 2 / (2 + 1 + 1) = 0.5
    # score = 0.6*50 + 0.5*49 = 30 + 24.5 = 54.5
    score = compute_nrc_text_score(text, lexicon)

    assert score == pytest.approx(54.5)


def test_compute_nrc_text_score_handles_text_with_no_emotion_words():
    lexicon = {"happy": {"positive", "joy"}}
    text = "the quick brown fox jumps"

    score = compute_nrc_text_score(text, lexicon)

    assert score == pytest.approx(0.0)


def test_compute_nrc_text_score_caps_at_99():
    lexicon = {"joy": {"positive", "joy"}}
    text = "joy joy joy joy joy"  # density=1.0, positive_ratio=5/6=0.833; raw = 50 + 40.8... capped

    score = compute_nrc_text_score(text, lexicon)

    assert score <= 99.0


def test_compute_nrc_text_score_handles_empty_text():
    lexicon = {"happy": {"positive", "joy"}}

    score = compute_nrc_text_score("", lexicon)

    assert score == 0.0
