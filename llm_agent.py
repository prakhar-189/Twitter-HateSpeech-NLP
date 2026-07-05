# llm_agent.py
# -------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-05-27
# Description : This module contains a function to route flagged tweets to a local LLM (like LLaMA) acting as a Trust & Safety Judge. The function takes the tweet text as input and returns a natural language explanation of the policy violation, if any, based on the LLM's analysis.
# -------------------------------------------------


# =================================================
# Imports
# --------------------------------------------
# os : To read OLLAMA_MODEL / OLLAMA_BASE_URL overrides (needed when Ollama runs
#   somewhere other than localhost, e.g. in Docker).
# langchain_ollama : Current official LangChain integration for local Ollama models
#   (this used to import Ollama from langchain_community.llms, which is deprecated
#   in favor of the dedicated langchain-ollama package).
# langchain_core.prompts : For creating prompt templates to guide the LLM's analysis.
# =================================================
import os

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Ollama has no built-in request timeout, so a hung local server would otherwise
# block the Streamlit UI indefinitely with just a spinner and no way out.
REQUEST_TIMEOUT_SECONDS = 30

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
# OllamaLLM defaults to http://localhost:11434, which is wrong when Ollama runs
# outside this process's container (e.g. on the Docker host) — OLLAMA_BASE_URL
# lets that be overridden without touching code.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")


# =================================================
# analyze_with_llm Function
# --------------------------------------------
# This function routes the tweet to a local LLM to act as a Trust & Safety Judge.
# It takes the tweet text and an optional model name (defaulting to the
# OLLAMA_MODEL env var, or "llama3") as input, and returns a natural language
# explanation of the policy violation based on the LLM's analysis.
# =================================================
def analyze_with_llm(tweet_text, model_name=None):
    """
    Routes the tweet to a local LLM to act as a Trust & Safety Judge.
    Returns a natural language explanation of the policy violation.
    """
    model_name = model_name or DEFAULT_MODEL

    llm_kwargs = {"model": model_name, "temperature": 0.2, "timeout": REQUEST_TIMEOUT_SECONDS}
    if OLLAMA_BASE_URL:
        llm_kwargs["base_url"] = OLLAMA_BASE_URL

    # Initialize the local LLM
    llm = OllamaLLM(**llm_kwargs)

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
        explanation = chain.invoke({"tweet": tweet_text})
        return explanation.strip()
    except Exception as e:
        return f"LLM Analysis failed (is Ollama running with `ollama pull {model_name}`?): {str(e)}"
