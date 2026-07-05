# src/evaluate.py
# ----------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-26
# Description : This module contains functions for evaluating the trained Logistic Regression models on the test data, including calculating and printing accuracy, precision, recall, F1 score, and a confusion matrix.
# ----------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# sklearn : For machine learning utilities, including metrics for model evaluation.
# =================================================
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# =================================================
# evaluate_model Function
# --------------------------------------------
# This function takes a trained model and data, makes predictions, and calculates
# accuracy, precision, recall, F1 score, and a confusion matrix. Precision and the
# confusion matrix were added because for a moderation tool, the tradeoff between
# false positives (wrongly flagging safe content) and false negatives (missing
# real hate speech) is the actual decision that matters, and neither is visible
# from accuracy/recall/F1 alone.
# =================================================
def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Predicts and evaluates the model, returning core metrics as a dict."""
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    # labels=[0, 1] forces a 2x2 matrix even if y/y_pred happen to contain only
    # one class (e.g. a small or filtered batch) — without it, confusion_matrix
    # silently collapses to 1x1 and .ravel() fails to unpack into 4 values.
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    print(f"--- Evaluation: {dataset_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}\n")

    return {
        "dataset_name": dataset_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
