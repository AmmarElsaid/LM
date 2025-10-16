from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from embedding_manager import EmbeddingManager

EMBEDDING_DIM = 384  

# Load lessons
with open("lessons_with_embeddings.json", "r", encoding="utf-8") as f:
    lessons = json.load(f)

embedder = EmbeddingManager(use_openai=False, embedding_dim=EMBEDDING_DIM)

def find_best_match(query, lessons, embedder, embedding_dim=EMBEDDING_DIM):
    """
    Finds the most similar lesson to the query using cosine similarity.
    Skips lessons with empty or mismatched embeddings.
    Returns structured lesson with explanation, code (CS only), links.
    """
    query_vec = np.array(embedder.embed_text(query), dtype=np.float32).reshape(1, -1)

    best_score = -1
    best_lesson = None

    for lesson in lessons:
        lesson_embedding = lesson.get("embedding", [])
        if len(lesson_embedding) != embedding_dim:
            continue  # skip invalid embeddings

        lesson_vec = np.array(lesson_embedding, dtype=np.float32).reshape(1, -1)
        score = cosine_similarity(query_vec, lesson_vec)[0][0]

        if score > best_score:
            best_score = score
            best_lesson = lesson

    # Format structured lesson
    if best_lesson:
        # Ensure keys exist
        explanation = best_lesson.get("explanation", "No explanation available.")
        code = best_lesson.get("code", "") if best_lesson.get("subject") == "Computer Science" else ""
        links = best_lesson.get("links", [])
        if not links:
            links = [{"title": "See explanation above", "link": "", "snippet": "Resources included in explanation"}]

        return {
            "topic": best_lesson.get("topic", "unknown"),
            "explanation": explanation,
            "code": code,
            "links": links
        }, best_score

    return None, 0.0

# ---------------- Test ----------------
if __name__ == "__main__":
    queries = ["Explain gravity", "Basics of acids", "Python arithmetic operators"]

    for q in queries:
        lesson, score = find_best_match(q, lessons, embedder)
        if lesson:
            print(f"\nQuery: {q}\nTopic: {lesson['topic']}\nExplanation: {lesson['explanation']}\nCode: {lesson['code']}\nLinks: {lesson['links']}\nScore: {score:.3f}")
        else:
            print(f"\nQuery: {q}\nNo good match found. Score: {score:.3f}")