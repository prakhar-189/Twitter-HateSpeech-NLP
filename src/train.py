# src/train.py
# ----------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-26
# Description : This module contains functions for training Logistic Regression models on the preprocessed tweet data, including a baseline model, a balanced model, and hyperparameter tuning using Grid Search with Stratified K-Fold cross-validation.
# ----------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# sklearn : For machine learning utilities, including Logistic Regression, Grid Search, and Stratified K-Fold cross-validation.
# =================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# =================================================
# train_baseline_model Function
# --------------------------------------------
# This function trains a baseline Logistic Regression model on the provided training data without any class weighting.
# =================================================
def train_baseline_model(X_train, y_train):
    """Trains an ordinary Logistic Regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


# =================================================
# train_balanced_model Function
# --------------------------------------------
# This function trains a Logistic Regression model with class weights balanced.
# =================================================
def train_balanced_model(X_train, y_train):
    """Trains a Logistic Regression model with class weights balanced."""
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model


# =================================================
# perform_grid_search Function
# --------------------------------------------
# This function performs hyperparameter tuning using Stratified K-Fold cross-validation.
# Tunes both C and penalty (the original problem statement asks for both) — 'liblinear'
# is used as the solver since it's the one that supports both l1 and l2 penalties.
# =================================================
def perform_grid_search(X_train, y_train):
    """Performs hyperparameter tuning (C and penalty) using Stratified K-Fold."""
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42),
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1
    )

    print("[*] Running Grid Search. This may take a moment...")
    grid_search.fit(X_train, y_train)

    print(f"[*] Best Parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_
