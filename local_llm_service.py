"""HTTP client for the local Ollama chat API."""
import os
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")


def ask_local_llm(prompt: str) -> str:
    """Send a prompt to Ollama and return the model response text."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The AI prompt cannot be empty.")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=(5, 180))
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return content
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The local AI model took too long to respond.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Could not connect to Ollama. Make sure Ollama is running and the model is available.") from exc
    except ValueError as exc:
        # Keep invalid server JSON distinguishable from connection errors.
        raise RuntimeError("Ollama returned an invalid response.") from exc
