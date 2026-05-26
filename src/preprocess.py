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

def clean_tweets(tweets_list):
    """Applies NLP cleanup steps to a list of tweets."""
    tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))
    redundant_terms = {'amp', 'rt'}
    
    cleaned_tweets_list = []
    all_tokens = []
    
    for tweet in tweets_list:
        tweet = str(tweet).lower()
        tweet = re.sub(r'@\w+', '', tweet) # Remove handles
        tweet = re.sub(r'http\S+|www\.\S+', '', tweet) # Remove URLs
        
        tokens = tokenizer.tokenize(tweet)
        cleaned_tokens = []
        
        for token in tokens:
            token = token.replace('#', '') # Remove hashtag symbol but keep text
            if (token not in stop_words and 
                token not in redundant_terms and 
                len(token) > 1):
                cleaned_tokens.append(token)
                
        cleaned_tweets_list.append(' '.join(cleaned_tokens))
        all_tokens.extend(cleaned_tokens)
        
    return cleaned_tweets_list, all_tokens

def load_and_preprocess_data(filepath):
    """Loads data, cleans it, splits it, and applies TF-IDF."""
    df = pd.read_csv(filepath)
    tweets_list = df['tweet'].tolist()
    
    # Clean tweets
    cleaned_tweets, all_tokens = clean_tweets(tweets_list)
    df['cleaned_tweet'] = cleaned_tweets
    
    # Top 10 terms analytics
    top_10 = Counter(all_tokens).most_common(10)
    print(f"[*] Top 10 most common terms: {top_10}\n")
    
    # Train-Test Split
    X = df['cleaned_tweet']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf