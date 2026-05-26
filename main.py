import os
from src.preprocess import load_and_preprocess_data
from src.train import train_baseline_model, train_balanced_model, perform_grid_search
from src.evaluate import evaluate_model

def main():
    data_path = os.path.join('data', 'Dataset.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find data at {data_path}. Please ensure Dataset.csv is in the 'data' folder.")
        return

    print("=========================================")
    print(" Twitter Hate Speech NLP Pipeline")
    print("=========================================\n")

    # 1. Preprocessing
    print("[1/3] Loading and preprocessing data...")
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = load_and_preprocess_data(data_path)

    # 2. Baseline Model
    print("[2/3] Training and evaluating baseline model...")
    baseline_model = train_baseline_model(X_train_tfidf, y_train)
    evaluate_model(baseline_model, X_train_tfidf, y_train, "Baseline Model (Train Set)")

    # 3. Balanced Model
    print("[3/3] Training and evaluating balanced model...")
    balanced_model = train_balanced_model(X_train_tfidf, y_train)
    evaluate_model(balanced_model, X_train_tfidf, y_train, "Balanced Model (Train Set)")

    # 4. Hyperparameter Tuning (Grid Search)
    print("=========================================")
    print(" Hyperparameter Tuning & Final Evaluation")
    print("=========================================\n")
    best_model = perform_grid_search(X_train_tfidf, y_train)
    
    # 5. Final Evaluation on Test Set
    evaluate_model(best_model, X_test_tfidf, y_test, "Best Estimator (Test Set)")

if __name__ == "__main__":
    main()