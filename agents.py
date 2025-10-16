import os
import time
import re
import json
import wikipedia
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dotenv import load_dotenv
from utils.logging_utils import log_entropy
from embedding_manager import EmbeddingManager
from sklearn.metrics.pairwise import cosine_similarity


import spacy
nlp = spacy.load("en_core_web_sm")
STOP_WORDS = set(spacy.lang.en.stop_words.STOP_WORDS)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

#File Paths
DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "seed_lessons.csv")
WIKI_JSON_DIR = os.path.join(os.path.dirname(__file__), "data", "wikipedia_data")
os.makedirs(WIKI_JSON_DIR, exist_ok=True)

LESSONS_EMBED_PATH = os.path.join(os.path.dirname(__file__), "lessons_with_embeddings.json")
embedder = EmbeddingManager(use_openai=False)

#Topic Planning
def planner_agent(user_prompt: str) -> Dict[str, Any]:
    subtasks = [
        {"type": "explain", "description": "Explain the topic simply."},
        {"type": "code", "description": "Provide a small runnable Python example."},
        {"type": "web", "description": "Find 1-3 curated tutorials and summarize briefly."},
    ]
    return {"topic_hint": infer_topic(user_prompt), "subtasks": subtasks}

def extract_topic(prompt: str) -> str:
    doc = nlp(prompt.lower())
    for chunk in doc.noun_chunks:
        if any(token.text not in STOP_WORDS for token in chunk):
            return chunk.text.strip()
    for token in doc:
        if token.is_alpha and token.text not in STOP_WORDS:
            return token.text.strip()
    return prompt

def infer_topic(prompt: str) -> str:
    keywords = ["recursion", "for loop", "for loops", "loops", "gradient descent",
                "lists", "binary search", "gravity", "beach"]
    p_low = prompt.lower()
    for k in keywords:
        if k in p_low:
            return k.replace("for loop", "for loops")
    tokens = re.findall(r"[a-zA-Z]+", p_low)
    return tokens[0] if tokens else "topic"

# Wikipedia Agent
def wikipedia_agent(topic: str) -> Dict[str, str]:
    try:
        summary = wikipedia.summary(topic, sentences=5)
        return {"topic": topic, "content": summary, "source": "Wikipedia"}
    except wikipedia.exceptions.DisambiguationError as e:
        return {"topic": topic, "content": "", "error": f"Ambiguous topic. Options: {e.options}"}
    except wikipedia.exceptions.PageError:
        return {"topic": topic, "content": "", "error": "No page found for this topic."}
    except Exception as e:
        return {"topic": topic, "content": "", "error": str(e)}


def extract_code_and_explanation(content: str):
    explanation_lines, code_lines = [], []
    in_code_block = False
    for line in content.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
        else:
            explanation_lines.append(line)
    explanation = "\n".join(explanation_lines).strip()
    code = "\n".join(code_lines).replace('\\n', '\n').strip('"')
    return explanation, code

#code generator agent
def tutor_agent(topic: str, user_prompt: str) -> Dict[str, Any]:
    """
    Calls OpenAI to generate only a concise Python code example for a topic.
    No explanation is returned.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return {"code": "# No code example generated.", "prompts_used": 0, "elapsed_time": 0.0}

    try:
        client = OpenAI(api_key=api_key)
        # System prompt instructs to return only code
        system_prompt = (
            "You are a concise programming tutor. Provide only one concise Python example "
            "in a single fenced code block. Do NOT include any explanations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User request: {user_prompt}\nFocus topic: {topic}"}
        ]

        start_time = time.time()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        elapsed = time.time() - start_time

        content = resp.choices[0].message.content.strip()

        _, code_text = extract_code_and_explanation(content)

        return {
            "code": code_text or "# No code example generated.",
            "prompts_used": 1,
            "elapsed_time": elapsed
        }

    except Exception as e:
        print("OpenAI call failed:", e)
        return {"code": "# No code example generated.", "prompts_used": 0, "elapsed_time": 0.0}

def explainer_agent(topic: str, user_prompt: str) -> Dict[str, Any]:
    """
    Calls OpenAI to generate only an explanation of a topic, without any code.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return {"explanation": "", "prompts_used": 0, "elapsed_time": 0.0}

    try:
        client = OpenAI(api_key=api_key)
        system_prompt = (
            "You are a concise tutor. Explain the topic clearly in 3-5 sentences. "
            "Do NOT provide any code examples."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User request: {user_prompt}\nFocus topic: {topic}"}
        ]

        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        elapsed = time.time() - start_time

        explanation = response.choices[0].message.content.strip()

        return {
            "explanation": explanation,
            "prompts_used": 1,
            "elapsed_time": elapsed
        }

    except Exception as e:
        print("OpenAI call failed:", e)
        return {"explanation": "", "prompts_used": 0, "elapsed_time": 0.0}

# Web agent
def web_agent(subject: str):
    """Return curated links based only on subject, ignoring topic."""
    links = []

    preferred_sites = {
        "Computer Science": [
            ("GeeksforGeeks", "https://www.geeksforgeeks.org/", "Tutorials and explanations"),
            ("StackOverflow", "https://stackoverflow.com/", "Questions and answers")
        ],
        "Mathematics": [
            ("Khan Academy", "https://www.khanacademy.org/math", "Interactive lessons"),
            ("Wolfram MathWorld", "https://mathworld.wolfram.com/", "Detailed explanations")
        ],
        "Biology": [
            ("Khan Academy Biology", "https://www.khanacademy.org/science/biology", "Biology tutorials"),
            ("Nature", "https://www.nature.com/subjects/biology", "Research articles")
        ],
        "Chemistry": [
            ("Chemistry LibreTexts", "https://chem.libretexts.org/", "Chemistry tutorials"),
            ("Khan Academy Chemistry", "https://www.khanacademy.org/science/chemistry", "Interactive lessons")
        ],
        "Physics": [
            ("HyperPhysics", "http://hyperphysics.phy-astr.gsu.edu/", "Physics explanations"),
            ("Khan Academy Physics", "https://www.khanacademy.org/science/physics", "Interactive lessons")
        ]
    }

    for site, link, snippet in preferred_sites.get(subject, []):
        links.append({"title": site, "link": link, "snippet": snippet})

    # fallback if no subject matched
    if not links:
        links = [{"title": "See explanation above", "link": "", "snippet": "Resources included in explanation"}]

    return links

# Validator
def validator_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["explanation"] = payload.get("explanation","").strip() or "No explanation generated."
    payload["code"] = payload.get("code","").strip() or "# No code example generated."
    payload["links"] = payload.get("links",[]) or [{"title":"No links found","link":"","snippet":"Try refining query."}]
    return payload

# Load and save lessons
def save_lesson(topic: str, explanation: str, code: str, links: list):
    """Save lesson CSV, JSON, and update embeddings automatically."""
    # CSV
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
    else:
        df = pd.DataFrame(columns=["topic", "explanation", "code", "links"])
    if topic.lower() in df["topic"].str.lower().values:
        df.loc[df["topic"].str.lower() == topic.lower(), ["explanation", "code", "links"]] = [explanation, code, json.dumps(links)]
    else:
        df = pd.concat([df, pd.DataFrame([{
            "topic": topic,
            "explanation": explanation,
            "code": code,
            "links": json.dumps(links)
        }])], ignore_index=True)
    df.to_csv(DATA_CSV, index=False)

    # JSON
    json_path = os.path.join(WIKI_JSON_DIR, f"{topic.replace(' ','_')}.json")
    data = {"topic": topic, "explanation": explanation, "code": code, "links": links}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Update embeddings
    try:
        if os.path.exists(LESSONS_EMBED_PATH):
            with open(LESSONS_EMBED_PATH, "r", encoding="utf-8") as f:
                lessons_with_embeddings = json.load(f)
        else:
            lessons_with_embeddings = []

        existing_topics = [l.get("topic","").lower() for l in lessons_with_embeddings]
        if topic.lower() not in existing_topics:
            embedding_vector = embedder.embed_text(explanation)
            lesson_entry = {
                "topic": topic,
                "explanation": explanation,
                "code": code,
                "links": links,
                "embedding": embedding_vector
            }
            lessons_with_embeddings.append(lesson_entry)
            with open(LESSONS_EMBED_PATH, "w", encoding="utf-8") as f:
                json.dump(lessons_with_embeddings, f, indent=2)
    except Exception as e:
        print(f"Failed to update lesson embeddings: {e}")

    print(f"Lesson saved: {topic}")

def load_saved_lesson(topic: str) -> dict:
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        mask = df["topic"].str.lower() == topic.lower()
        if mask.any():
            row = df[mask].iloc[0]
            return {
                "topic": row["topic"],
                "explanation": row["explanation"],
                "code": row["code"],
                "links": json.loads(row["links"]) if "links" in row else []
            }
    return None

# ------------------- Embedding-based Retrieval -------------------
def retrieve_lesson_from_embeddings(query: str, threshold: float = 0.75):
    if not os.path.exists(LESSONS_EMBED_PATH):
        return None
    with open(LESSONS_EMBED_PATH, "r", encoding="utf-8") as f:
        lessons = json.load(f)

    query_vec = np.array(embedder.embed_text(query), dtype=np.float32).reshape(1, -1)
    best_score = -1
    best_lesson = None
    for lesson in lessons:
        lesson_vec = np.array(lesson["embedding"], dtype=np.float32).reshape(1, -1)
        score = cosine_similarity(query_vec, lesson_vec)[0][0]
        if score > best_score:
            best_score = score
            best_lesson = lesson
    return best_lesson if best_score >= threshold else None

# flaaback run pipeline-

def run_pipeline(user_prompt: str, subject: str = None) -> dict:
    """
    Original working pipeline - manual tool execution
    """
    meta = {"prompts_used": 0, "agent_failures": 0, "timing_sec": {}}
    t0 = time.time()

    try:
        ts = time.time()
        plan = planner_agent(user_prompt)
        meta["timing_sec"]["planner"] = time.time() - ts
        topic = extract_topic(user_prompt)

        # 1. Try embedding retrieval first
        ts = time.time()
        lesson = retrieve_lesson_from_embeddings(topic)
        meta["timing_sec"]["embedding_retrieval"] = time.time() - ts

        if lesson:
            print(f"Retrieved lesson from embeddings: {lesson['topic']}")
        else:
            # 2. Wikipedia
            ts = time.time()
            wiki_data = wikipedia_agent(topic)
            meta["timing_sec"]["wikipedia"] = time.time() - ts
            explanation = wiki_data.get("content", "")

            # 3. OpenAI fallback for explanation
            if not explanation.strip():
                ts = time.time()
                try:
                    api_key = os.getenv("OPENAI_API_KEY", "").strip()
                    if api_key and OpenAI is not None:
                        client = OpenAI(api_key=api_key)
                        system_prompt = "You are a concise tutor. Explain the topic in 3-5 sentences."
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Explain the topic: {topic}"}
                        ]
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.2
                        )
                        explanation = resp.choices[0].message.content.strip()
                        meta["prompts_used"] += 1
                    else:
                        explanation = "No explanation available."
                except Exception as e:
                    explanation = f"Error generating explanation: {e}"
                meta["timing_sec"]["openai_explain"] = time.time() - ts

            # 4. Tutor code only for Computer Science
            ts = time.time()
            if subject == "Computer Science":
                tutor_out = tutor_agent(topic, user_prompt, explain=False)
                code_text = tutor_out.get("code", "")
                meta["prompts_used"] += tutor_out.get("prompts_used", 0)
            else:
                code_text = ""
            meta["timing_sec"]["tutor"] = time.time() - ts

            # 5. Subject-specific resources
            ts = time.time()
            links = web_agent(topic, subject)
            meta["timing_sec"]["web"] = time.time() - ts

            # 6. Validate
            lesson = validator_agent({
                "topic": topic,
                "explanation": explanation,
                "code": code_text,
                "links": links
            })

            # 7. Save lesson + embeddings
            save_lesson(lesson["topic"], lesson["explanation"], lesson["code"], lesson["links"])

        meta["completed"] = True

    except Exception as e:
        meta["agent_failures"] += 1
        meta["completed"] = False
        lesson = {"topic": "unknown", "explanation": f"Pipeline error: {e}", "code": "", "links": []}

    meta["total_time_sec"] = time.time() - t0
    log_entropy("lesson_pipeline", {"prompt": user_prompt, **meta})

    return {"meta": meta, "lesson": lesson}