"""Google Gemini client used by cleAIra's AI review services."""

import os

from google import genai
from google.genai import errors, types
from dotenv import load_dotenv


GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
load_dotenv(override=True)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the model response text."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The AI prompt cannot be empty.")

    api_key = os.environ.get(GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in the application environment."
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
    except errors.APIError as exc:
        status_code = getattr(exc, "code", "unknown")
        detail = str(getattr(exc, "message", "") or exc)
        raise RuntimeError(
            f"Gemini API request failed with status {status_code}: {detail}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError("The Gemini model took too long to respond.") from exc
    except OSError as exc:
        raise RuntimeError("Could not connect to the Gemini API.") from exc

    content = response.text
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return content
