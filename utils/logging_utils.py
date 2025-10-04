import json, os, time
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "entropy.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_entropy(feature: str, payload: dict):
    entry = {
        "feature": feature,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **payload
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
