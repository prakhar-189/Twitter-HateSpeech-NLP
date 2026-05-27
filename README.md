# 🛡️ Twitter Hate Speech Detection — NLP Pipeline

An end-to-end NLP system for detecting hate speech in tweets, combining a classical machine learning pipeline with a Streamlit web interface and an agentic LLM layer for deeper context analysis.

---

## Overview

This project tackles the problem of automated hate speech classification on Twitter (X). It uses **TF-IDF vectorization** and **Logistic Regression** as the core classifier, augmented by a **local LLM agent** (via LangChain + Ollama) that acts as a "Senior Trust & Safety Judge" to provide nuanced, natural-language explanations for borderline or flagged content.

The system is designed as an internal moderation tool — not just a binary classifier, but an explainable pipeline that surfaces *why* a tweet was flagged.

---

## Features

- **NLP preprocessing pipeline** — cleans raw tweet text (strips URLs, mentions, hashtags, stopwords)
- **TF-IDF + Logistic Regression** — fast, interpretable baseline classifier
- **Class imbalance handling** — trains a balanced model to address skewed label distributions
- **Hyperparameter tuning** — Grid Search with Stratified K-Fold cross-validation to find the optimal model
- **Model persistence** — saves the best model and vectorizer as `.pkl` artifacts via `joblib`
- **Streamlit UI** — interactive web app to classify arbitrary tweet text in real time
- **Lexical explainability** — visualizes per-term TF-IDF coefficients to show what drove each prediction
- **LLM agentic review** — routes flagged or borderline tweets to a local LLaMA 3 model for a 2–3 sentence policy violation explanation

---

## Project Structure

```
Twitter-HateSpeech-NLP/
│
├── src/
│   ├── preprocess.py       # Data loading, tweet cleaning, TF-IDF vectorization
│   ├── train.py            # Baseline model, balanced model, Grid Search tuning
│   └── evaluate.py         # Model evaluation (classification report, metrics)
│
├── data/
│   └── Dataset.csv         # Tweet dataset (not included — see Setup)
│
├── models/                 # Generated after running main.py
│   ├── tfidf_vectorizer.pkl
│   └── best_model.pkl
│
├── main.py                 # Orchestrates the full training pipeline
├── app.py                  # Streamlit web application
├── llm_agent.py            # LangChain + Ollama LLM reviewer
├── requirements.txt
├── Problem Statement.pdf
├── .gitignore
└── LICENSE
```

---

## How It Works

### Training Pipeline (`main.py`)

The pipeline runs in three stages:

1. **Load & Preprocess** — reads `data/Dataset.csv`, cleans tweet text, and produces TF-IDF feature matrices for train/test splits.
2. **Train & Evaluate** — fits a baseline Logistic Regression and a class-balanced variant, evaluating each on the training set.
3. **Hyperparameter Tuning** — runs Grid Search with Stratified K-Fold CV to find the best regularization parameters, evaluates the winner on the held-out test set, then saves the vectorizer and model to `models/`.

### Streamlit App (`app.py`)

- Accepts raw tweet text as input.
- Preprocesses → vectorizes → classifies using the saved model.
- Displays a styled result card (🚨 toxic / ✅ safe) with a confidence percentage.
- Shows a bar chart of the top TF-IDF coefficient contributions (lexical explainability).
- For **toxic** predictions or **borderline safe** predictions (toxic probability > 40%), automatically routes to the LLM agent for a secondary review.

### LLM Agent (`llm_agent.py`)

Uses **LangChain** with a local **Ollama** server (default model: `llama3`) to run a "Trust & Safety Moderator" persona. The agent is prompted to provide a 2–3 sentence explanation of *why* the content might violate platform policies, and whether sarcasm, reclaimed language, or slang may have caused a misclassification.

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally with the `llama3` model pulled:
  ```bash
  ollama pull llama3
  ```

### Installation

```bash
# Clone the repository
git clone https://github.com/prakhar-189/Twitter-HateSpeech-NLP.git
cd Twitter-HateSpeech-NLP

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place your tweet dataset as `data/Dataset.csv`. The file should contain a text column for tweets and a label column indicating hate speech (1) vs. safe (0). Adjust the column names in `src/preprocess.py` if needed.

---

## Usage

### Step 1 — Train the Model

```bash
python main.py
```

This will preprocess the data, train and tune the model, print evaluation results to the console, and save `models/tfidf_vectorizer.pkl` and `models/best_model.pkl`.

### Step 2 — Launch the App

```bash
streamlit run app.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`), paste or type a tweet, and click **Run Diagnostics**.

> **Note:** The LLM agent requires Ollama to be running in the background. If Ollama is not available, the ML classifier still works — the LLM step will display an error message gracefully.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML / NLP | scikit-learn, NLTK |
| Data | pandas, NumPy |
| Model Persistence | joblib |
| Web App | Streamlit |
| LLM Integration | LangChain, LangChain-Community |
| Local LLM | Ollama (LLaMA 3) |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Prakhar Srivastava**  
[github.com/prakhar-189](https://github.com/prakhar-189)