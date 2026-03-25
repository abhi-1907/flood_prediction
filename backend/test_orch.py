import asyncio
import logging
from agents.orchestration.orchestrator import orchestrator
from utils.logger import logger

# Set logger to DEBUG so we see everything
logging.getLogger().setLevel(logging.DEBUG)

async def main():
    try:
        res = await orchestrator.run("flood in mumbai", user_type="general")
        import json
        
        # print parts of the response to avoid huge output
        print("--- RESPONSE KEYS ---")
        print(list(res.keys()))
        print("--- STEPS SUMMARY ---")
        print(json.dumps(res.get("steps_summary", []), indent=2))
        if "error" in res:
            print("ERROR IN RESPONSE:", res["error"])
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
