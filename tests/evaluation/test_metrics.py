import math

from src.evaluation.metrics import compute_regression_metrics, compute_tier_metrics


def test_compute_regression_metrics_perfect_prediction():
    result = compute_regression_metrics([10.0, 50.0, 90.0], [10.0, 50.0, 90.0])

    assert result["rmse"] == 0.0
    assert result["mae"] == 0.0
    assert result["r2"] == 1.0


def test_compute_regression_metrics_known_error():
    # y_true=[0, 10], y_pred=[2, 8] -> errors [2, -2], MAE=2, RMSE=2
    result = compute_regression_metrics([0.0, 10.0], [2.0, 8.0])

    assert result["mae"] == 2.0
    assert math.isclose(result["rmse"], 2.0, rel_tol=1e-9)


def test_compute_tier_metrics_perfect_prediction():
    result = compute_tier_metrics([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["weighted_kappa"] == 1.0
    assert len(result["confusion_matrix"]) == 6


def test_compute_tier_metrics_all_wrong_by_one_tier():
    # Every prediction is off by exactly one tier — weighted kappa should
    # still be well above 0 (near misses are penalized less than random),
    # while plain accuracy is 0.
    result = compute_tier_metrics([2, 3, 4, 5], [1, 2, 3, 4])

    assert result["accuracy"] == 0.0
    assert result["weighted_kappa"] > 0.0


def test_compute_tier_metrics_confusion_matrix_shape_matches_label_count():
    result = compute_tier_metrics([1, 1, 2], [1, 2, 2])

    assert len(result["confusion_matrix"]) == len(result["confusion_matrix"][0])
