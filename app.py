import streamlit as st
from langchain_pipeline import pipeline
import re
from collections import defaultdict
import time
import json
from datetime import datetime
import os

LOG_FILE = "entropy_log.json"

# ------------------- Page Config -------------------
st.set_page_config(page_title="LearnMate", page_icon="🎓", layout="centered")
st.title("LearnMate — Interactive Tutor 🎓")
st.markdown(
    "Type a request and select the subject. "
    "The AI will fetch explanations, code, and additional resources."
)

# ------------------- Initialize Session State -------------------
if "lesson" not in st.session_state:
    st.session_state.lesson = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "history" not in st.session_state:
    st.session_state.history = []
if "entropy_metrics" not in st.session_state:
    st.session_state.entropy_metrics = {
        "total_queries": 0,
        "method_counts": defaultdict(int),
        "fallbacks": 0,
        "total_time": 0.0
    }
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = "Explain Data Structures in Python"
if "subject" not in st.session_state:
    st.session_state.subject = "Computer Science"

# Ensure log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ------------------- Sidebar: Session History -------------------
with st.sidebar:
    st.markdown("## 🕘 Session History")
    if st.session_state.history:
        max_height = 400
        st.markdown(f'<div style="overflow-y:auto; max-height:{max_height}px;">', unsafe_allow_html=True)
        for i, entry in enumerate(reversed(st.session_state.history), start=1):
            prompt = entry["prompt"]
            subject_entry = entry["subject"]
            if st.button(f"{i}. {prompt} ({subject_entry})", key=f"history_{i}"):
                st.session_state.user_prompt = entry["prompt"]
                st.session_state.subject = entry["subject"]
                st.session_state.lesson = entry["lesson"]
                st.session_state.meta = entry.get("meta", {})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No session history yet.")

# ------------------- Subject Selection -------------------
st.session_state.subject = st.selectbox(
    "Choose a subject",
    ["Computer Science", "Mathematics", "Biology", "Chemistry", "Physics"],
    index=["Computer Science", "Mathematics", "Biology", "Chemistry", "Physics"].index(st.session_state.subject)
)

# ------------------- User Prompt Input -------------------
st.session_state.user_prompt = st.text_input(
    "Your request:",
    value=st.session_state.user_prompt,
    key="user_prompt_input"
)

# ------------------- Generate Lesson -------------------
if st.button("Generate Lesson"):
    with st.spinner("Thinking..."):
        start = time.time()
        out = pipeline.run(st.session_state.user_prompt, subject=st.session_state.subject)
        elapsed = time.time() - start

        st.session_state.lesson = out["lesson"]
        st.session_state.meta = out["meta"]

        # Save to history
        st.session_state.history.append({
            "prompt": st.session_state.user_prompt,
            "subject": st.session_state.subject,
            "lesson": out["lesson"],
            "meta": out["meta"]
        })

        # ------------------- Update Entropy Metrics -------------------
        method_used = out["meta"].get("method", "unknown")
        st.session_state.entropy_metrics["total_queries"] += 1
        st.session_state.entropy_metrics["method_counts"][method_used] += 1
        st.session_state.entropy_metrics["total_time"] += elapsed

        # Count true fallbacks only
        if method_used == "fallback_pipeline" and out.get("lesson") is None:
            st.session_state.entropy_metrics["fallbacks"] += 1

        # ------------------- Append to Entropy Log -------------------
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": st.session_state.user_prompt,
            "subject": st.session_state.subject,
            "method": method_used,
            "response_time_sec": round(elapsed, 3),
            "fallback": method_used == "fallback_pipeline" and out.get("lesson") is None
        }
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=2)

# ------------------- Display Lesson -------------------
lesson = st.session_state.lesson
meta = st.session_state.meta

if lesson:
    # Explanation
    explanation_text = lesson.get("explanation", "").strip()
    explanation_text = re.sub(r"```python[\s\S]*?```", "", explanation_text).strip()
    if explanation_text:
        st.markdown("### Explanation")
        st.write(explanation_text)

    # Code Example (CS only)
    code_text = lesson.get("code", "").strip()
    if st.session_state.subject == "Computer Science" and code_text:
        st.markdown("### Code Example")
        st.code(code_text, language="python")

    # Additional Resources
    links = lesson.get("links", [])
    if links:
        st.markdown("### Additional Resources")
        for link in links:
            title = link.get("title", "(no title)")
            href = link.get("link", "")
            snippet = link.get("snippet", "")
            if href:
                st.markdown(f"- [{title}]({href}) — {snippet}")
            else:
                st.markdown(f"- **{title}** — {snippet}")

    # Debug Metadata
    with st.expander("Run metadata (for debugging)"):
        st.json(meta)

# ------------------- Entropy Metrics Panel -------------------
with st.sidebar:
    st.markdown("## Entropy Metrics")
    metrics = st.session_state.entropy_metrics
    st.write(f"**Total Queries:** {metrics['total_queries']}")
    st.write(f"**Fallbacks:** {metrics['fallbacks']}")
    fallback_ratio = metrics['fallbacks'] / metrics['total_queries'] if metrics['total_queries'] else 0
    st.write(f"**Fallback Ratio:** {fallback_ratio:.2f}")
    avg_time = metrics['total_time'] / metrics['total_queries'] if metrics['total_queries'] else 0
    st.write(f"**Average Response Time:** {avg_time:.2f}s")

    st.write("**Queries by Method:**")
    for method, count in metrics['method_counts'].items():
        st.write(f"- {method}: {count}")

# ------------------- Footer -------------------
st.caption(
    "This app uses LearnMate agent to fetch explanations, code, and curated resources."
)