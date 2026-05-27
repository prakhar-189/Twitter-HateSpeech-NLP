# src/llm_agent.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-27
# Description : This module contains a function to route flagged tweets to a local LLM (like LLaMA) acting as a Trust & Safety Judge. The function takes the tweet text as input and returns a natural language explanation of the policy violation, if any, based on the LLM's analysis.
# -------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# langchain_community.llms : For interfacing with local LLMs like LLaMA.
# langchain_core.prompts : For creating prompt templates to guide the LLM's analysis.
# =================================================
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


# =================================================
# analyze_with_llm Function
# --------------------------------------------
# This function routes the tweet to a local LLM to act as a Trust & Safety Judge. 
# It takes the tweet text and an optional model name (defaulting to "llama3") as input, and returns a natural language explanation of the policy violation based on the LLM's analysis.
# =================================================
def analyze_with_llm(tweet_text, model_name="llama3"):
    """
    Routes the tweet to a local LLM to act as a Trust & Safety Judge.
    Returns a natural language explanation of the policy violation.
    """
    # Initialize the local LLM
    llm = Ollama(model=model_name, temperature=0.2)
    
    # Define the persona and instructions for the LLM
    template = """
    You are a Trust & Safety AI Moderator for Twitter. 
    A lightweight machine learning model has flagged the following tweet for potential hate speech, toxicity, or policy violation.
    
    Your task is to act as a senior reviewer. Analyze the tweet and provide a concise, 2-3 sentence explanation of WHY it might violate safety policies. 
    If it uses sarcasm, reclaimed words, or slang that the standard model might have misunderstood, point that out.
    
    Tweet: "{tweet}"
    
    Moderator Analysis:
    """
    
    prompt = PromptTemplate(
        input_variables=["tweet"],
        template=template
    )
    
    # We pipe the prompt directly into the LLM
    chain = prompt | llm 
    
    try:
        # We use .invoke() instead of the old .run() method
        explanation = chain.invoke({"tweet": tweet_text})
        return explanation.strip()
    except Exception as e:
        return f"LLM Analysis failed: {str(e)}"