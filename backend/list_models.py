import os
from google import genai
from config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

print("Available models:")
for m in client.models.list():
    print(f"- {m.name}")
