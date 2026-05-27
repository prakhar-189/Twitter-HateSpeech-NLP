# src/evaluate.py
# ----------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-26
# Description : This module contains functions for evaluating the trained Logistic Regression models on the test data, including calculating and printing accuracy, recall, and F1 score metrics.
# ----------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# sklearn : For machine learning utilities, including metrics for model evaluation.
# =================================================
from sklearn.metrics import accuracy_score, recall_score, f1_score


# =================================================
# evaluate_model Function
# --------------------------------------------
# This function takes a trained model and test data, makes predictions, and calculates accuracy, recall, and F1 score. It also prints the results in a formatted manner.
# =================================================
def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Predicts and evaluates the model, returning core metrics."""
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"--- Evaluation: {dataset_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    
    return acc, recall, f1