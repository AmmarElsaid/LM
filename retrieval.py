from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from embedding_manager import EmbeddingManager

# Load lessons
with open("lessons_with_embeddings.json", "r") as f:
    lessons = json.load(f)

# Create embedder (same as used for generating embeddings)
embedder = EmbeddingManager(use_openai=False)

def find_best_match(query, lessons, embedder):
    # Get embedding (could be list of floats) and convert to 1D NumPy array
    query_vec = np.array(embedder.embed_text(query), dtype=np.float32)
    if query_vec.ndim > 1:
        query_vec = query_vec.flatten()  # make sure it’s 1D
    query_vec = query_vec.reshape(1, -1)  # reshape for cosine_similarity

    best_score = -1
    best_lesson = None

    for lesson in lessons:
        lesson_vec = np.array(lesson["embedding"], dtype=np.float32)
        if lesson_vec.ndim > 1:
            lesson_vec = lesson_vec.flatten()
        lesson_vec = lesson_vec.reshape(1, -1)

        score = cosine_similarity(query_vec, lesson_vec)[0][0]

        if score > best_score:
            best_score = score
            best_lesson = lesson

    return best_lesson, best_score


# Test queries
queries = ["What is gravity", "Basics of acids"]

for q in queries:
    lesson, score = find_best_match(q, lessons, embedder)  # <- pass embedder here!
    if lesson:
        print(f"Query: {q}\nBest match: {lesson['explanation']}\nScore: {score}\n")
    else:
        print(f"Query: {q}\nNo good match found. Score: {score}\n")