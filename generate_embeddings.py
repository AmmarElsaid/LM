import os
import json
from dotenv import load_dotenv
from embedding_manager import EmbeddingManager

# Load environment variables
load_dotenv()

# Pick backend
backend = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()
use_openai = backend == "openai"

# Init embedder
embedder = EmbeddingManager(use_openai=use_openai)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "lessons_with_embeddings.json")

lessons_with_embeddings = []

# Load lessons from JSON files in data folder
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".json"):
        path = os.path.join(DATA_FOLDER, filename)
        with open(path, "r", encoding="utf-8") as f:
            lesson = json.load(f)
            # Combine fields for stronger embedding
            text_to_embed = " ".join([
                lesson.get("topic", ""),
                lesson.get("explanation", ""),
                lesson.get("code", "")
            ])
            lesson["embedding"] = embedder.embed_text(text_to_embed)
            lessons_with_embeddings.append(lesson)

# Save all lessons with embeddings
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(lessons_with_embeddings, f, indent=2)

print(f"✅ Generated embeddings for {len(lessons_with_embeddings)} lessons.")
print(f"Saved to {OUTPUT_FILE}")

