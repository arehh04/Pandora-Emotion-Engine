import pandas as pd

from src.preprocessing.validate_augmentation import compute_augmentation_coverage


def test_compute_augmentation_coverage_full_and_balanced(tmp_path):
    clean_path = tmp_path / "train_clean.csv"
    augmented_path = tmp_path / "train_augmented.csv"

    pd.DataFrame({
        "bert_text": [f"text {i}" for i in range(10)],
        "extraversion": list(range(10)),
    }).to_csv(clean_path, index=False)

    # All 10 original rows present (type is NaN), plus balanced augmented rows.
    rows = []
    for i in range(10):
        rows.append({"bert_text": f"text {i}", "extraversion": i, "type": None, "expressiveness_bin": "Low"})
    for i in range(5):
        rows.append({"bert_text": f"aug low {i}", "extraversion": i, "type": "_AUGMENTED", "expressiveness_bin": "Medium"})
    for i in range(5):
        rows.append({"bert_text": f"aug high {i}", "extraversion": i, "type": "_AUGMENTED", "expressiveness_bin": "High"})
    pd.DataFrame(rows).to_csv(augmented_path, index=False)

    result = compute_augmentation_coverage(str(clean_path), str(augmented_path))

    assert result["total_original_rows"] == 10
    assert result["covered_original_rows"] == 10
    assert result["coverage_ratio"] == 1.0
    assert result["bin_counts"] == {"Low": 10, "Medium": 5, "High": 5}
    assert result["is_balanced"] is True


def test_compute_augmentation_coverage_partial_and_unbalanced(tmp_path):
    clean_path = tmp_path / "train_clean.csv"
    augmented_path = tmp_path / "train_augmented.csv"

    pd.DataFrame({
        "bert_text": [f"text {i}" for i in range(10)],
        "extraversion": list(range(10)),
    }).to_csv(clean_path, index=False)

    # Only 6 of the 10 original rows present, and bins are unbalanced (10 Low vs 2 High).
    rows = []
    for i in range(6):
        rows.append({"bert_text": f"text {i}", "extraversion": i, "type": None, "expressiveness_bin": "Low"})
    for i in range(4):
        rows.append({"bert_text": f"more low {i}", "extraversion": i, "type": None, "expressiveness_bin": "Low"})
    for i in range(2):
        rows.append({"bert_text": f"aug high {i}", "extraversion": i, "type": "_AUGMENTED", "expressiveness_bin": "High"})
    pd.DataFrame(rows).to_csv(augmented_path, index=False)

    result = compute_augmentation_coverage(str(clean_path), str(augmented_path))

    assert result["total_original_rows"] == 10
    assert result["covered_original_rows"] == 10  # all 10 rows have type NaN (none augmented in this fixture)
    assert result["coverage_ratio"] == 1.0
    assert result["bin_counts"] == {"Low": 10, "High": 2}
    assert result["is_balanced"] is False
