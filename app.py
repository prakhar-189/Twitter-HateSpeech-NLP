# app.py
# ----------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-27
# Description : This is the Streamlit app that serves as the user interface for the Twitter Hate Speech NLP pipeline. It allows users to input tweet text, processes it through the trained model, and displays the classification results along with an explanation of the model's decision. Additionally, it integrates a local LLM agent to provide deeper context analysis for flagged tweets.
# ----------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# streamlit : For building the interactive web application.
# joblib : For loading the trained model and TF-IDF vectorizer.
# pandas : For data manipulation and display.
# src.preprocess : For cleaning the input tweet text before classification.
# llm_agent : For routing flagged tweets to a local LLM for deeper analysis and explanation.
# =================================================
import joblib
import pandas as pd
import streamlit as st

from llm_agent import analyze_with_llm
from src.preprocess import clean_tweets

# Page Configuration
st.set_page_config(page_title="Twitter Moderation Hub", page_icon="🛡️", layout="centered")


# =================================================
# Custom CSS Injection
# --------------------------------------------
# This function injects custom CSS styles into the Streamlit app to enhance the visual presentation of the results, including custom cards for toxic and safe classifications, typography styling, and a more polished UI.
# =================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* Typography and Header Styling */
        .twitter-header {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #1DA1F2;
            font-weight: 800;
            font-size: 38px;
            margin-bottom: 5px;
        }
        .sub-header {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #657786;
            font-size: 16px;
            margin-bottom: 25px;
        }
        
        /* Custom Result Cards */
        .result-card-toxic {
            background-color: rgba(224, 36, 94, 0.05); 
            border: 1px solid rgba(224, 36, 94, 0.2); 
            border-left: 6px solid #E0245E; 
            padding: 20px; 
            border-radius: 8px;
            margin-top: 15px;
            margin-bottom: 25px;
        }
        .result-card-safe {
            background-color: rgba(23, 191, 99, 0.05); 
            border: 1px solid rgba(23, 191, 99, 0.2); 
            border-left: 6px solid #17BF63; 
            padding: 20px; 
            border-radius: 8px;
            margin-top: 15px;
            margin-bottom: 25px;
        }
        
        /* Typography inside cards */
        .card-title { margin: 0; font-size: 22px; font-weight: bold; }
        .text-toxic { color: #E0245E; }
        .text-safe { color: #17BF63; }
        .confidence-text { font-size: 18px; margin-top: 8px; color: #14171a; }
        
        /* Code tag styling for the processed text */
        .processed-text {
            background-color: #f5f8fa;
            color: #1DA1F2;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

# Run the CSS injection
inject_custom_css()


# ==================================================
# Load Model Artifacts
# ---------------------------------------------
# This function loads the trained TF-IDF vectorizer and the best Logistic Regression model from the 'models' directory. It uses Streamlit's caching mechanism to avoid reloading the artifacts on every interaction, improving performance. If the artifacts are not found, it displays an error message and stops the app.
# ==================================================
@st.cache_resource 
def load_artifacts():
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/best_model.pkl')
    return vectorizer, model

try:
    tfidf_vectorizer, best_model = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found! Please run main.py first to generate the .pkl files.")
    st.stop()

# Using custom HTML for the title instead of st.title()
st.markdown('<div class="twitter-header">🛡️ Trust & Safety AI Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Internal tool for automated hate speech classification using TF-IDF and Logistic Regression.</div>', unsafe_allow_html=True)

# Text Input
user_input = st.text_area("Enter tweet payload for analysis:", placeholder="Type or paste tweet text here...", height=120)

if st.button("Run Diagnostics", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Payload empty. Please enter text to analyze.")
    else:
        with st.spinner("Processing through NLP pipeline..."):
            
            # Preprocess
            cleaned_text_list, _ = clean_tweets([user_input])
            cleaned_text = cleaned_text_list[0]
            
            if cleaned_text.strip() == "":
                st.info("The tweet was filtered out entirely by preprocessing (e.g., it only contained URLs, hashtags, or stop words).")
                st.stop()

            # Vectorize and Predict
            vectorized_text = tfidf_vectorizer.transform([cleaned_text])
            prediction = best_model.predict(vectorized_text)[0]
            probabilities = best_model.predict_proba(vectorized_text)[0]
            
            safe_prob = probabilities[0] * 100
            toxic_prob = probabilities[1] * 100

            # Custom HTML Output
            st.markdown("### Classification Result")
            
            if prediction == 1:
                html_result = f"""
                <div class="result-card-toxic">
                    <p class="card-title text-toxic">🚨 Policy Violation: Hate Speech Detected</p>
                    <p class="confidence-text">The model is <b>{toxic_prob:.1f}%</b> confident this content violates platform safety guidelines.</p>
                </div>
                """
                st.markdown(html_result, unsafe_allow_html=True)
                st.markdown("### 🤖 Agentic Policy Review")
                with st.spinner("Routing to LLM for deep context analysis..."):
                    llm_explanation = analyze_with_llm(user_input)
                    
                    st.info(f"**Senior AI Judge:** {llm_explanation}")
            else:
                html_result = f"""
                <div class="result-card-safe">
                    <p class="card-title text-safe">✅ Content Cleared: Safe</p>
                    <p class="confidence-text">The model is <b>{safe_prob:.1f}%</b> confident this content adheres to platform safety guidelines.</p>
                </div>
                """

                if toxic_prob > 40.0:
                    st.warning("⚠️ Borderline Content Detected. Routing to LLM for secondary review...")
                    with st.spinner("Analyzing nuance..."):
                        llm_explanation = analyze_with_llm(user_input)
                        st.info(f"**Secondary AI Review:** {llm_explanation}")    
            
            # Render the HTML card
            st.markdown(html_result, unsafe_allow_html=True)

            # Explainability Section
            st.markdown("### Lexical Explainability")
            st.markdown(f"Text as parsed by the model: <span class='processed-text'>{cleaned_text}</span>", unsafe_allow_html=True)
            st.write("") # Spacer
            
            vocab = tfidf_vectorizer.vocabulary_
            coefs = best_model.coef_[0]

            # set() dedupes repeated words in the tweet — without it, a word
            # appearing twice produced two identical rows, which set_index("Term")
            # below would turn into a duplicate index and render oddly in the chart.
            word_impacts = []
            for word in set(cleaned_text.split()):
                if word in vocab:
                    idx = vocab[word]
                    weight = coefs[idx]
                    word_impacts.append({"Term": word, "Toxicity Coefficient": weight})

            if word_impacts:
                df_impact = pd.DataFrame(word_impacts)
                df_impact = df_impact.sort_values(by="Toxicity Coefficient", ascending=False)
                
                # Streamlit's native bar chart is clean, but now it sits in a better UI structure
                st.caption("How specific terms influenced the model's decision (Positive = pushes toward Toxic, Negative = pushes toward Safe):")
                st.bar_chart(df_impact.set_index("Term"), color="#1DA1F2")
            else:
                st.write("None of the terms in this tweet were found in the model's trained vocabulary.")