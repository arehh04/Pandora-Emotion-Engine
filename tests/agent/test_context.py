import json
import os

import numpy as np
import pandas as pd
import spacy

from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.context import train_ml_prior_from_data, load_rag_context


def test_train_ml_prior_from_data_returns_working_model():
    nlp = spacy.load("en_core_web_sm")
    nrc_dict = load_nrc_lexicon(os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))

    model = train_ml_prior_from_data("data/train_clean.csv", nlp, nrc_dict, sample_size=30, seed=1)

    prediction = model.predict([[0.1, 0.0, 0.5, 0.05, 0.0, 0.2, 0.05, 0.05]])
    assert prediction.shape == (1,)


def test_load_rag_context_returns_none_when_artifacts_missing(tmp_path):
    result = load_rag_context(str(tmp_path))

    assert result is None


def test_load_rag_context_loads_corpus_with_injected_embedder(tmp_path):
    pd.DataFrame({
        "bert_text": ["I love parties"], "extraversion": [90], "tier": [6], "tier_label": ["Highly Extraverted"],
    }).to_csv(tmp_path / "exemplars_meta.csv", index=False)
    np.save(tmp_path / "exemplars_embeddings.npy", np.array([[1.0, 0.0]]))
    (tmp_path / "theory_meta.json").write_text(
        json.dumps([{"id": "a", "topic": "t", "text": "gregariousness", "citation_needed": "n/a"}]),
        encoding="utf-8",
    )
    np.save(tmp_path / "theory_embeddings.npy", np.array([[1.0, 0.0]]))

    class FakeEmbedder:
        def encode(self, texts):
            return np.array([[1.0, 0.0] for _ in texts])

    result = load_rag_context(str(tmp_path), embedder=FakeEmbedder())

    assert result is not None
    assert len(result["corpus"]["exemplars_df"]) == 1
    assert result["embedder"] is not None
