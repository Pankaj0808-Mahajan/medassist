import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.getenv("WEIGHTS_DIR", BASE / "weights"))
GUIDELINES = BASE / "data" / "guidelines"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
