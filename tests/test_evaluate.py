import numpy as np

from src.evaluate import evaluate_model


class _StubModel:
    """Fake classifier so evaluate_model can be tested without training a real one."""

    def __init__(self, predictions):
        self._predictions = np.array(predictions)

    def predict(self, X):
        return self._predictions


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    model = _StubModel(y_true)
    result = evaluate_model(model, X=None, y=y_true, dataset_name="perfect")

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_all_false_positives():
    y_true = [0, 0, 0, 0]
    model = _StubModel([1, 1, 1, 1])
    result = evaluate_model(model, X=None, y=y_true, dataset_name="all_fp")

    assert result["precision"] == 0.0  # zero_division=0, no crash on 0/0
    assert result["confusion_matrix"] == {"tn": 0, "fp": 4, "fn": 0, "tp": 0}


def test_missed_all_positives():
    y_true = [1, 1, 0, 0]
    model = _StubModel([0, 0, 0, 0])
    result = evaluate_model(model, X=None, y=y_true, dataset_name="missed_positives")

    assert result["recall"] == 0.0
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 2, "tp": 0}


def test_returns_dataset_name():
    result = evaluate_model(_StubModel([0]), X=None, y=[0], dataset_name="my_label")
    assert result["dataset_name"] == "my_label"
