import json
import os

import numpy as np
import pandas as pd
import spacy

from src.agent.tools.classical_features import load_nrc_lexicon
from src.agent.tools.ml_prior import train_ml_prior
from src.agent.tool_schemas import TOOL_SCHEMAS, dispatch_tool_call


def _build_test_context(tmp_path=None):
    nlp = spacy.load("en_core_web_sm")
    nrc_dict = load_nrc_lexicon(os.path.join("data", "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))

    feature_rows = [
        {"positive": 0.3, "negative": 0.0, "semantic_polarity": 0.9, "behav_exclamation_ratio": 0.2,
         "behav_question_ratio": 0.0, "behav_verb_ratio": 0.3, "behav_1st_sg_pronoun_ratio": 0.0,
         "behav_1st_pl_pronoun_ratio": 0.2},
        {"positive": 0.0, "negative": 0.3, "semantic_polarity": -0.9, "behav_exclamation_ratio": 0.0,
         "behav_question_ratio": 0.0, "behav_verb_ratio": 0.05, "behav_1st_sg_pronoun_ratio": 0.2,
         "behav_1st_pl_pronoun_ratio": 0.0},
    ]
    scores = [90.0, 10.0]
    ml_model = train_ml_prior(feature_rows, scores)

    ctx = {"nlp": nlp, "nrc_dict": nrc_dict, "ml_model": ml_model, "rag": None}

    if tmp_path is not None:
        rag_dir = tmp_path / "rag"
        rag_dir.mkdir()
        pd.DataFrame({
            "bert_text": ["I love parties"], "extraversion": [90], "tier": [6], "tier_label": ["Highly Extraverted"],
        }).to_csv(rag_dir / "exemplars_meta.csv", index=False)
        np.save(rag_dir / "exemplars_embeddings.npy", np.array([[1.0, 0.0]]))
        (rag_dir / "theory_meta.json").write_text(
            json.dumps([{"id": "a", "topic": "t", "text": "gregariousness", "citation_needed": "n/a"}]),
            encoding="utf-8",
        )
        np.save(rag_dir / "theory_embeddings.npy", np.array([[1.0, 0.0]]))

        class FakeEmbedder:
            def encode(self, texts):
                return np.array([[1.0, 0.0] for _ in texts])

        from src.agent.tools.rag_retrieval import load_rag_corpus
        ctx["rag"] = {"corpus": load_rag_corpus(str(rag_dir)), "embedder": FakeEmbedder()}

    return ctx


def test_tool_schemas_have_five_entries_with_required_names():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {
        "fuzzy_logic_assessment", "ml_prior_assessment",
        "retrieve_similar_exemplars", "retrieve_relevant_theory", "submit_assessment",
    }
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_dispatch_fuzzy_logic_assessment_returns_valid_result():
    ctx = _build_test_context()

    result = dispatch_tool_call("fuzzy_logic_assessment", {"text": "I love parties and meeting people!"}, ctx)

    assert set(result.keys()) == {"fuzzy_score", "tier", "tier_label", "fired_rules"}


def test_dispatch_ml_prior_assessment_returns_valid_result():
    ctx = _build_test_context()

    result = dispatch_tool_call("ml_prior_assessment", {"text": "I love parties!"}, ctx)

    assert set(result.keys()) == {"score", "tier", "tier_label"}


def test_dispatch_rag_tools_return_error_when_corpus_absent():
    ctx = _build_test_context()  # ctx["rag"] is None

    exemplar_result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "hello"}, ctx)
    theory_result = dispatch_tool_call("retrieve_relevant_theory", {"text": "hello"}, ctx)

    assert "error" in exemplar_result
    assert "error" in theory_result


def test_dispatch_rag_tools_return_results_when_corpus_present(tmp_path):
    ctx = _build_test_context(tmp_path)

    result = dispatch_tool_call("retrieve_similar_exemplars", {"text": "party time", "k": 1}, ctx)

    assert "results" in result
    assert result["results"][0]["bert_text"] == "I love parties"


def test_dispatch_unknown_tool_returns_error():
    ctx = _build_test_context()

    result = dispatch_tool_call("not_a_real_tool", {}, ctx)

    assert "error" in result


def test_dispatch_never_raises_on_bad_arguments():
    ctx = _build_test_context()

    result = dispatch_tool_call("fuzzy_logic_assessment", {}, ctx)  # missing required "text"

    assert "error" in result
