# src/preprocess.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-26
# Description : This module contains functions for loading cleaning & preprocessing the tweet data, including tokeniztion, stopward-removal & TF-IDF vectoriztion.
# -------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# Pandas : For data manipulation and analysis.
# re : For regular expression operations to clean tweets.
# nltk : For natural language processing tasks, including tokenization and stopword removal.
# sklearn : For machine learning utilities, including train-test splitting and TF-IDF vectorization.
# =================================================
import pandas as pd
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)


# =================================================
# clean_tweets Function
# --------------------------------------------
# This function takes a list of tweets and applies several NLP cleanup steps:
# 1. Converts tweets to lowercase.
# 2. Removes Twitter handles and URLs using regular expressions.
# 3. Tokenizes the cleaned tweets using NLTK's TweetTokenizer.
# 4. Removes stopwords and redundant terms like 'amp' and 'rt'.
# 5. Returns a list of cleaned tweets and a list of all tokens for analytics.
# =================================================
def clean_tweets(tweets_list):
    """Applies NLP cleanup steps to a list of tweets."""
    tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))
    redundant_terms = {'amp', 'rt'}
    
    cleaned_tweets_list = []
    all_tokens = []
    
    # 1. Converts tweets to lowercase
    for tweet in tweets_list:
        tweet = str(tweet).lower()

        # 2. Removes Twitter handles and URLs using regular expressions
        tweet = re.sub(r'@\w+', '', tweet) # Remove handles
        tweet = re.sub(r'http\S+|www\.\S+', '', tweet) # Remove URLs
        
        # 3. Tokenizes the cleaned tweets using NLTK's TweetTokenizer
        tokens = tokenizer.tokenize(tweet)
        cleaned_tokens = []
        
        for token in tokens:
            # 4. Removes stopwords and redundant terms like 'amp' and 'rt'
            token = token.replace('#', '') # Remove hashtag symbol but keep text
            if (token not in stop_words and 
                token not in redundant_terms and 
                len(token) > 1):
                cleaned_tokens.append(token)

        # 5. Returns a list of cleaned tweets and a list of all tokens for analytics        
        cleaned_tweets_list.append(' '.join(cleaned_tokens))
        all_tokens.extend(cleaned_tokens)
        
    return cleaned_tweets_list, all_tokens


# =================================================
# load_and_preprocess_data Function
# --------------------------------------------
# This function orchestrates the entire data loading and preprocessing pipeline:
# 1. Loads the dataset from a CSV file.
# 2. Cleans the tweets using the clean_tweets function.
# 3. Prints the top 10 most common terms across all cleaned tweets for analytics.
# 4. Splits the cleaned data into training and testing sets using an 80-20 split, ensuring stratification by label.
# 5. Applies TF-IDF vectorization to the cleaned tweets, limiting the feature set to the top 5000 terms.
# 6. Returns the TF-IDF vectors for the training and testing sets, the corresponding labels, and the fitted TF-IDF vectorizer for future use.
# =================================================
def load_and_preprocess_data(filepath):

    # 1. Loads the dataset from a CSV file.
    """Loads data, cleans it, splits it, and applies TF-IDF."""
    df = pd.read_csv(filepath)
    tweets_list = df['tweet'].tolist()
    
    # 2. Cleans the tweets using the clean_tweets function.
    cleaned_tweets, all_tokens = clean_tweets(tweets_list)
    df['cleaned_tweet'] = cleaned_tweets
    
    # 3. Prints the top 10 most common terms across all cleaned tweets for analytics.
    top_10 = Counter(all_tokens).most_common(10)
    print(f"[*] Top 10 most common terms: {top_10}\n")
    
    # 4. Splits the cleaned data into training and testing sets using an 80-20 split, ensuring stratification by label.
    X = df['cleaned_tweet']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Applies TF-IDF vectorization to the cleaned tweets, limiting the feature set to the top 5000 terms.
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 6. Returns the TF-IDF vectors for the training and testing sets, the corresponding labels, and the fitted TF-IDF vectorizer for future use.
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf