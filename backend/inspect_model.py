import os
from google import genai
from config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

print("Listing models and attributes:")
models = list(client.models.list())
if models:
    m = models[0]
    print(f"Model ID: {m.name}")
    print(f"Attributes: {dir(m)}")
    # If it's a model info object from the older API, it might have 'supported_generation_methods'
    # But for the new SDK, let's see what's there.
