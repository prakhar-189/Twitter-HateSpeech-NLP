# main.py
# ----------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-26
# Description : This is the main script that orchestrates the entire Twitter Hate Speech NLP pipeline. It loads and preprocesses the data, trains multiple Logistic Regression models (baseline and balanced), performs hyperparameter tuning using Grid Search with Stratified K-Fold cross-validation, evaluates the models on the test set, and saves the best model and TF-IDF vectorizer for future use in a Streamlit app.
# ----------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# os : For file path operations and directory management.
# joblib : For saving and loading the trained model and vectorizer.
# src.preprocess : For loading and preprocessing the tweet data.
# src.train : For training Logistic Regression models and performing hyperparameter tuning.
# src.evaluate : For evaluating the trained models on the test data.
# =================================================
import os
import joblib
from src.preprocess import load_and_preprocess_data
from src.train import train_baseline_model, train_balanced_model, perform_grid_search
from src.evaluate import evaluate_model


# =================================================
# main Function
# --------------------------------------------
# This function orchestrates the entire pipeline:
# 1. Loads and preprocesses the data.
# 2. Trains a baseline Logistic Regression model and evaluates it on the training set.
# 3. Trains a balanced Logistic Regression model and evaluates it on the training set.
# 4. Performs hyperparameter tuning using Grid Search with Stratified K-Fold cross-validation to find the best model.
# 5. Evaluates the best model on the test set.
# 6. Saves the best model and TF-IDF vectorizer for future use in a Streamlit app.
# =================================================
def main():
    data_path = os.path.join('data', 'Dataset.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find data at {data_path}. Please ensure Dataset.csv is in the 'data' folder.")
        return

    print("=========================================")
    print(" Twitter Hate Speech NLP Pipeline")
    print("=========================================\n")

    # 1. Loads and preprocesses the data.
    print("[1/3] Loading and preprocessing data...")
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = load_and_preprocess_data(data_path)

    # 2. Trains a baseline Logistic Regression model and evaluates it on the training set.
    print("[2/3] Training and evaluating baseline model...")
    baseline_model = train_baseline_model(X_train_tfidf, y_train)
    evaluate_model(baseline_model, X_train_tfidf, y_train, "Baseline Model (Train Set)")

    # 3. Trains a balanced Logistic Regression model and evaluates it on the training set.
    print("[3/3] Training and evaluating balanced model...")
    balanced_model = train_balanced_model(X_train_tfidf, y_train)
    evaluate_model(balanced_model, X_train_tfidf, y_train, "Balanced Model (Train Set)")

    # 4. Performs hyperparameter tuning using Grid Search with Stratified K-Fold cross-validation to find the best model.
    print("=========================================")
    print(" Hyperparameter Tuning & Final Evaluation")
    print("=========================================\n")
    best_model = perform_grid_search(X_train_tfidf, y_train)
    
    # 5. Evaluates the best model on the test set.
    evaluate_model(best_model, X_test_tfidf, y_test, "Best Estimator (Test Set)")

    print("\n=========================================")
    print(" Saving Model Artifacts for Streamlit")
    print("=========================================\n")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # 6. Saves the best model and TF-IDF vectorizer for future use in a Streamlit app.
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print("[*] Successfully saved to /models directory!")

if __name__ == "__main__":
    main()