import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_openai_client():
    """Create an OpenAI client using the API key from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Check your local .env file."
        )

    return OpenAI(api_key=api_key)