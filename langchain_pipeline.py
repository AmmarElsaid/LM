# langchain_pipeline.py
import os
import json
import time
import numpy as np
from embedding_manager import EmbeddingManager
from langchain_agent import learnmate_agent
from agents import web_agent, run_pipeline as fallback_pipeline

LESSONS_FILE = "lessons_with_embeddings.json"
SIMILARITY_THRESHOLD = 0.65

class Pipeline:
    def __init__(self):
        # Load lessons
        if os.path.exists(LESSONS_FILE):
            with open(LESSONS_FILE, "r", encoding="utf-8") as f:
                self.lessons_with_embeddings = json.load(f)
        else:
            self.lessons_with_embeddings = []

        self.embedder = EmbeddingManager(use_openai=False)

    # ---------------- Helper Methods ----------------
    def cosine_similarity_vec(self, vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_similar_lesson(self, prompt_embedding):
        best_lesson = None
        best_score = 0.0
        for lesson in self.lessons_with_embeddings:
            existing_embedding = lesson.get("embedding", [])
            if not existing_embedding or all(v == 0 for v in existing_embedding):
                continue
            score = self.cosine_similarity_vec(prompt_embedding, existing_embedding)
            #print(f"Comparing to lesson: {lesson.get('explanation','')[:30]}... score={score:.3f}")
            if score > best_score:
                best_score = score
                best_lesson = lesson
        if best_score >= SIMILARITY_THRESHOLD:
            return best_lesson
        return None

    def save_lessons(self):
        with open(LESSONS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.lessons_with_embeddings, f, indent=2)

    # ---------------- Main Run Method ----------------
    def run(self, user_prompt: str, subject: str = "General") -> dict:
        start_time = time.time()
        try:
            prompt_embedding = self.embedder.embed_text(user_prompt)
            existing_lesson = self.find_similar_lesson(prompt_embedding)
            if existing_lesson:
                return {
                    "meta": {"completed": True, "method": "embedding_retrieval", "time_sec": time.time() - start_time, "fallback": False},
                    "lesson": existing_lesson
                }

            if learnmate_agent is None:
                # fallback to old pipeline
                return fallback_pipeline(user_prompt, subject)

            # LangChain agent call
            result = learnmate_agent.invoke({"input": user_prompt, "subject": subject})

            # Decide which formatter based on subject
            from formatters import get_formatter
            formatter = get_formatter(subject)
            new_lesson = formatter.format(result)

            self.lessons_with_embeddings.append(new_lesson)
            self.save_lessons()

            return {
                "meta": {"completed": True, "method": "langchain", "time_sec": time.time() - start_time},
                "lesson": new_lesson
            }

        except Exception as e:
            print(f"Pipeline failed: {e}")
            return fallback_pipeline(user_prompt, subject)


# ------------------- Module-level instance -------------------
pipeline = Pipeline()