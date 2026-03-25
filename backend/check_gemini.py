import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from backend/.env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

print(f"🔑 API Key loaded: {api_key[:8]}...{api_key[-4:]}")

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"❌ Failed to initialize Gemini Client: {e}")
    exit(1)

print("\n🧪 Testing 1-token generation on key models to check quota...")
models_to_test = [
    "gemini-2.5-flash", 
    "gemini-1.5-flash", 
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-2.5-pro"
]

for m in models_to_test:
    print(f"Testing {m:<20} ...", end=" ")
    try:
        # We send a trivial prompt and limit output to 5 tokens so it uses virtually zero quota.
        response = client.models.generate_content(
            model=m,
            contents="Hi",
            config=types.GenerateContentConfig(max_output_tokens=5, temperature=0.0)
        )
        print(f"✅ SUCCESS (Reply: {response.text.strip()})")
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "exhausted" in err_str or "quota" in err_str:
            print("❌ QUOTA EXHAUSTED (429)")
        elif "403" in err_str or "permission" in err_str:
            print("❌ NO ACCESS (403)")
        elif "404" in err_str or "not found" in err_str:
            print("❌ MODEL NOT FOUND (404)")
        else:
            print(f"❌ ERROR: {e}")

print("\nDone!")
