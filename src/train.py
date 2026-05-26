from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def train_baseline_model(X_train, y_train):
    """Trains an ordinary Logistic Regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_balanced_model(X_train, y_train):
    """Trains a Logistic Regression model with class weights balanced."""
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def perform_grid_search(X_train, y_train):
    """Performs hyperparameter tuning using Stratified K-Fold."""
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        # CHANGED: Added solver='lbfgs' explicitly (it is the default, but good practice)
        estimator=LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42),
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1
    )
    
    print("[*] Running Grid Search. This may take a moment...")
    grid_search.fit(X_train, y_train)
    
    print(f"[*] Best Parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_