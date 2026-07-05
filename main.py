# main.py
# ----------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-26
# Description : This is the main script that orchestrates the entire Twitter Hate Speech NLP pipeline. It loads and preprocesses the data, trains multiple Logistic Regression models (baseline and balanced), performs hyperparameter tuning using Grid Search with Stratified K-Fold cross-validation, evaluates all model variants on the same held-out test set, and saves the best model, TF-IDF vectorizer, and metrics for future use in a Streamlit app.
# ----------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# os : For file path operations and directory management.
# json : For persisting evaluation metrics to disk.
# joblib : For saving and loading the trained model and vectorizer.
# src.preprocess : For loading and preprocessing the tweet data.
# src.train : For training Logistic Regression models and performing hyperparameter tuning.
# src.evaluate : For evaluating the trained models on the test data.
# =================================================
import json
import os

import joblib

from src.evaluate import evaluate_model
from src.preprocess import load_and_preprocess_data
from src.train import perform_grid_search, train_balanced_model, train_baseline_model


# =================================================
# main Function
# --------------------------------------------
# This function orchestrates the entire pipeline:
# 1. Loads and preprocesses the data.
# 2. Trains a baseline Logistic Regression model.
# 3. Trains a balanced Logistic Regression model.
# 4. Performs hyperparameter tuning using Grid Search with Stratified K-Fold cross-validation.
# 5. Evaluates all three model variants on the SAME held-out test set, so the
#    comparison between baseline / balanced / tuned is actually meaningful
#    (previously the first two were only evaluated on the training set, which
#    doesn't say anything about how they generalize).
# 6. Saves the best model, TF-IDF vectorizer, and all metrics for future use.
# =================================================
def main():
    data_path = os.path.join('data', 'Dataset.csv')

    if not os.path.exists(data_path):
        print(f"Error: Could not find data at {data_path}. Please ensure Dataset.csv is in the 'data' folder.")
        return

    print("=========================================")
    print(" Twitter Hate Speech NLP Pipeline")
    print("=========================================\n")

    print("[1/4] Loading and preprocessing data...")
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = load_and_preprocess_data(data_path)

    print("[2/4] Training baseline and balanced models...")
    baseline_model = train_baseline_model(X_train_tfidf, y_train)
    balanced_model = train_balanced_model(X_train_tfidf, y_train)

    print("[3/4] Hyperparameter tuning (Grid Search + Stratified K-Fold)...")
    best_model, best_params = perform_grid_search(X_train_tfidf, y_train)

    print("\n=========================================")
    print(" Evaluation (all models on the same held-out test set)")
    print("=========================================\n")
    all_metrics = {
        "baseline": evaluate_model(baseline_model, X_test_tfidf, y_test, "Baseline Model (Test Set)"),
        "balanced": evaluate_model(balanced_model, X_test_tfidf, y_test, "Balanced Model (Test Set)"),
        "best": evaluate_model(best_model, X_test_tfidf, y_test, "Best Estimator (Test Set)"),
        "best_grid_search_params": best_params,
    }

    print("[4/4] Saving model artifacts and metrics...")
    os.makedirs('models', exist_ok=True)

    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(best_model, 'models/best_model.pkl')

    with open('models/eval_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("[*] Successfully saved model, vectorizer, and metrics to /models directory!")


if __name__ == "__main__":
    main()
