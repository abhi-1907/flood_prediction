import os
from google import genai
from config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

models = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash"
]

for m in models:
    try:
        print(f"Testing {m}...")
        resp = client.models.generate_content(
            model=m,
            contents="Say hi"
        )
        print(f"  SUCCESS! {resp.text.strip()}")
    except Exception as e:
        err_str = str(e)
        if "RESOURCE_EXHAUSTED" in err_str:
            if "limit: 0" in err_str:
                print("  FAILED: limit 0 (No Free Tier)")
            else:
                print("  FAILED: Quota Exceeded (Limit > 0)")
        elif "NOT_FOUND" in err_str:
            print("  FAILED: 404 Not Found")
        else:
            print(f"  FAILED: {err_str[:150]}")
