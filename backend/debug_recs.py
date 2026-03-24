import urllib.request
import json
import traceback

payload = json.dumps({
    "location": "Kochi",
    "risk_level": "high",
    "user_type": "general_public",
    "has_elderly": False,
    "has_children": False,
    "has_disability": False,
    "vehicle_access": True
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8000/recommendations", 
    data=payload,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        resp_bytes = response.read()
        print("HTTP 200 OK")
        
        data = json.loads(resp_bytes.decode("utf-8"))
        print("\n--- TYPES LIST ---")
        print(f"data.recommendations type: {type(data.get('recommendations'))}")
        
        if data.get("recommendations"):
            first_rec = data["recommendations"][0]
            print(f"first_rec type: {type(first_rec)}")
            for k, v in first_rec.items():
                print(f"  {k} : {type(v).__name__} = {repr(v)[:50]}")
                if isinstance(v, list) and len(v) > 0:
                    print(f"    [0] type: {type(v[0]).__name__}")
        
        print(f"\ndata.summary type: {type(data.get('summary'))}")
        print(f"data.safety_message type: {type(data.get('safety_message'))}")
        
except urllib.error.HTTPError as e:
    err_body = e.read().decode("utf-8")
    print(f"HTTP Error {e.code}:\n{err_body}")
except Exception as e:
    traceback.print_exc()
