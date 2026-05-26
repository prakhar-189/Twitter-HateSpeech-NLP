from sklearn.metrics import accuracy_score, recall_score, f1_score

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