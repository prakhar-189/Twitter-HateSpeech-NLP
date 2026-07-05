FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py llm_agent.py main.py pyproject.toml ./

EXPOSE 8501

# Requires data/Dataset.csv mounted for `main.py` (training) and models/*.pkl
# mounted for `app.py` (serving) — see README "Dataset" section for why these
# aren't baked into the image. The LLM review step needs a reachable Ollama
# server: set OLLAMA_BASE_URL (e.g. http://host.docker.internal:11434) if it's
# not running inside this container, and OLLAMA_MODEL to override "llama3".
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
