from src.eq_data.external_datasets import (
    normalize_emobank_row,
    normalize_empathetic_dialogues_row,
    normalize_goemotions_row,
    normalize_isear_row,
)


def test_normalize_goemotions_row_maps_label_indices_to_names():
    # Real verified schema: {'text': str, 'labels': list[int]}, indices into
    # a fixed 28-name list ending in "neutral" at index 27.
    row = {"text": "My favourite food is anything I didn't have to cook myself.", "labels": [27], "id": "eebbqej"}

    result = normalize_goemotions_row(row)

    assert result["text"] == row["text"]
    assert result["source"] == "goemotions"
    assert result["emotion_labels"] == ["neutral"]
    assert result["valence"] is None
    assert result["arousal"] is None
    assert result["dominance"] is None


def test_normalize_goemotions_row_handles_multi_label():
    row = {"text": "happy but nervous", "labels": [17, 19], "id": "x"}  # joy=17, nervousness=19

    result = normalize_goemotions_row(row)

    assert result["emotion_labels"] == ["joy", "nervousness"]


def test_normalize_isear_row_lowercases_and_strips_emotion():
    # Real verified schema: {'emotion': str, 'content': str, 'Unnamed: 2': None}
    row = {"emotion": "joy", "content": "On days when I feel close to my partner.  \n", "Unnamed: 2": None}

    result = normalize_isear_row(row)

    assert result["text"] == "On days when I feel close to my partner."
    assert result["source"] == "isear"
    assert result["emotion_labels"] == ["joy"]
    assert result["valence"] is None


def test_normalize_emobank_row_carries_vad_values():
    # Real verified schema: {'id', 'split', 'V', 'A', 'D', 'text'}
    row = {"id": "x", "split": "train", "V": 3.0, "A": 3.0, "D": 3.2, "text": 'Remember what she said?'}

    result = normalize_emobank_row(row)

    assert result["text"] == "Remember what she said?"
    assert result["source"] == "emobank"
    assert result["emotion_labels"] == []
    assert result["valence"] == 3.0
    assert result["arousal"] == 3.0
    assert result["dominance"] == 3.2


def test_normalize_empathetic_dialogues_row_uses_situation_as_text():
    # Real verified schema: {'conv_id', 'situation', 'emotion', 'conversations'}
    row = {
        "conv_id": "hit:0_conv:1",
        "situation": "I remember going to the fireworks with my best friend.",
        "emotion": "sentimental",
        "conversations": [{"content": "...", "role": "user"}],
    }

    result = normalize_empathetic_dialogues_row(row)

    assert result["text"] == row["situation"]
    assert result["source"] == "empathetic_dialogues"
    assert result["emotion_labels"] == ["sentimental"]
    assert result["valence"] is None
