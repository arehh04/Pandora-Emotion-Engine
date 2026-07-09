"""Reports how complete and balanced a stratified-paraphrase augmentation run is.

Run directly to check data/train_augmented.csv against data/train_clean.csv:
    python -m src.preprocessing.validate_augmentation
"""
import os

import pandas as pd


def compute_augmentation_coverage(clean_path, augmented_path, bin_col="expressiveness_bin"):
    clean_df = pd.read_csv(clean_path)
    aug_df = pd.read_csv(augmented_path)

    if "type" in aug_df.columns:
        original_rows = aug_df[aug_df["type"].isna()]
    elif "is_augmented" in aug_df.columns:
        original_rows = aug_df[aug_df["is_augmented"] != True]  # noqa: E712
    else:
        original_rows = aug_df

    total_original = len(clean_df)
    covered_original = len(original_rows)
    coverage_ratio = covered_original / total_original if total_original else 0.0

    bin_counts = {}
    is_balanced = False
    if bin_col in aug_df.columns:
        bin_counts = aug_df[bin_col].value_counts().to_dict()
        counts = list(bin_counts.values())
        if counts and min(counts) > 0:
            is_balanced = (max(counts) / min(counts)) <= 1.5

    return {
        "total_original_rows": total_original,
        "covered_original_rows": covered_original,
        "coverage_ratio": coverage_ratio,
        "bin_counts": bin_counts,
        "is_balanced": is_balanced,
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")

    result = compute_augmentation_coverage(
        os.path.join(data_dir, "train_clean.csv"),
        os.path.join(data_dir, "train_augmented.csv"),
    )

    print(f"Original rows covered: {result['covered_original_rows']}/{result['total_original_rows']} "
          f"({result['coverage_ratio']:.1%})")
    print(f"Bin counts: {result['bin_counts']}")
    print(f"Balanced (max/min <= 1.5): {result['is_balanced']}")

    if result["coverage_ratio"] < 1.0:
        print("\nWARNING: train_augmented.csv does not cover all rows from train_clean.csv.")
        print("Re-run src/preprocessing/augment_gemma3_colab.py on Colab to completion before")
        print("retraining the legacy ML tool models or building the RAG exemplar corpus.")


if __name__ == "__main__":
    main()
