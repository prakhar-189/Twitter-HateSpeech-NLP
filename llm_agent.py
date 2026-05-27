# src/llm_agent.py
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

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
    
    # --- THIS IS THE MODERN LCEL SYNTAX ---
    # We pipe the prompt directly into the LLM
    chain = prompt | llm 
    
    try:
        # We use .invoke() instead of the old .run() method
        explanation = chain.invoke({"tweet": tweet_text})
        return explanation.strip()
    except Exception as e:
        return f"LLM Analysis failed: {str(e)}"