import os
import json
from embedding_manager import EmbeddingManager

DATA_FOLDER = "data"
OUTPUT_FILE = "lessons_with_embeddings.json"

embedder = EmbeddingManager(use_openai=False)  # HuggingFace
lessons_with_embeddings = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".json"):
        filepath = os.path.join(DATA_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lesson = json.load(f)
            lesson_text = lesson.get("lesson_text") or lesson.get("explanation") or ""
            lesson["embedding"] = embedder.embed_text(lesson_text)
            lessons_with_embeddings.append(lesson)

# Save all lessons with embeddings
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(lessons_with_embeddings, f, indent=2)

print(f"✅ Processed {len(lessons_with_embeddings)} lessons and saved to {OUTPUT_FILE}")