import os
from google import genai
from config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

print("Inspecting first model in detail:")
models = list(client.models.list())
if models:
    m = models[0]
    print(f"Model ID: {m.name}")
    for attr in dir(m):
        if 'supported' in attr.lower() or 'method' in attr.lower():
            print(f" - {attr}: {getattr(m, attr, 'N/A')}")
else:
    print("No models found")
