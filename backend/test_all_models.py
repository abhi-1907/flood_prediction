import os
import sys
import time
from google import genai
from google.genai import types
from config import settings

def test_models():
    """
    Dynamically lists all available Gemini models and tests their accessibility.
    """
    print("=" * 60)
    print("GEMINI MODEL TESTER")
    print("=" * 60)

    if not settings.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in configuration.")
        return

    print(f"API Key: {settings.GEMINI_API_KEY[:8]}...{settings.GEMINI_API_KEY[-4:]}")

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Gemini Client: {e}")
        return

    print("\nFetching available models...")
    try:
        # List all models
        all_models = list(client.models.list())
        
        # Filter for models that support text generation
        text_models = [
            m for m in all_models 
            if 'generateContent' in m.supported_actions
        ]
        
        if not text_models:
            print("No text generation models found.")
            return

        print(f"Found {len(text_models)} text generation models.\n")
        
        # Table Header
        header = f"{'Model Name':<35} | {'Status':<20}"
        print(header)
        print("-" * len(header))

        for model in text_models:
            model_id = model.name
            # Simplified name for display if it starts with 'models/'
            display_name = model_id.replace("models/", "")
            
            print(f"{display_name:<35} | ", end="", flush=True)
            
            try:
                # Test with a minimal prompt to check quota/access
                # Using 1 max token to be as cheap as possible
                response = client.models.generate_content(
                    model=model_id,
                    contents="Hi",
                    config=types.GenerateContentConfig(
                        max_output_tokens=1,
                        temperature=0.0
                    )
                )
                print("✅ SUCCESS")
            except Exception as e:
                err_msg = str(e).upper()
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    print("❌ QUOTA EXHAUSTED (429)")
                elif "PERMISSION_DENIED" in err_msg or "403" in err_msg:
                    print("🚫 ACCESS DENIED (403)")
                elif "NOT_FOUND" in err_msg or "404" in err_msg:
                    print("❓ NOT FOUND (404)")
                else:
                    # Print a truncated error message for others
                    clean_err = str(e).replace("\n", " ")[:30]
                    print(f"⚠️ ERROR: {clean_err}...")

            # Small sleep to be nice to the API
            time.sleep(0.1)

    except Exception as e:
        print(f"\nCritical Error during model listing: {e}")

    print("\n" + "=" * 60)
    print("Test Completed.")
    print("=" * 60)
    print("Note: 'Credits left' is not directly exposed by the SDK.")
    print("Status success indicates you have remaining quota for that model.")

if __name__ == "__main__":
    test_models()
