import streamlit as st
from langchain_pipeline import run_pipeline_langchain as run_pipeline
from agents import tutor_agent, web_agent

st.set_page_config(page_title="LearnMate", page_icon="🎓", layout="centered")
st.title("LearnMate — Interactive Tutor 🎓")
st.markdown("Type a request like: **'Explain recursion in Python'** and select the subject.")

# ------------------- Subject Selection -------------------
subject = st.selectbox(
    "Choose a subject",
    ["Computer Science", "Mathematics", "Biology", "Chemistry", "Physics"]
)

# ------------------- User Prompt -------------------
user_prompt = st.text_input(
    "Your request:",
    value="Explain recursion in Python"
)

# ------------------- Session State -------------------
if "lesson" not in st.session_state:
    st.session_state.lesson = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "show_code" not in st.session_state:
    st.session_state.show_code = False
if "show_resources" not in st.session_state:
    st.session_state.show_resources = False

# ------------------- Generate Lesson -------------------
if st.button("Generate Lesson"):
    with st.spinner("Thinking..."):
        out = run_pipeline(user_prompt, subject=subject)
        st.session_state.lesson = out["lesson"]
        st.session_state.meta = out["meta"]
        # Reset buttons
        st.session_state.show_code = False
        st.session_state.show_resources = False

# Only display lesson if it exists
lesson = st.session_state.lesson
meta = st.session_state.meta

if lesson:
    # Display topic
    st.subheader(f"Topic: {lesson.get('topic', '(unknown)')}")

    # Explanation
    st.markdown("### Explanation")
    st.write(lesson.get("explanation", ""))

    # ------------------- Code Generation (CS Only) -------------------
    if subject == "Computer Science":
        if st.button("Generate Code Example"):
            with st.spinner("Generating code..."):
                code_out = tutor_agent(lesson["topic"], user_prompt)
                st.session_state.lesson["code"] = code_out.get("code", "")
                st.session_state.show_code = True

        if st.session_state.show_code:
            code_text = st.session_state.lesson.get("code", "")
            if code_text:
                st.markdown("### Code Example")
                st.code(code_text, language="python")
            else:
                st.info("No code example available.")

    # ------------------- Additional Resources -------------------
    if st.button("Generate Additional Resources"):
        with st.spinner("Fetching additional resources..."):
            links = web_agent(lesson.get("topic", ""), subject)
            st.session_state.lesson["links"] = links
            st.session_state.show_resources = True

    if st.session_state.show_resources:
        links = st.session_state.lesson.get("links", [])
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
        else:
            st.info("No additional resources found.")

    # ------------------- Metadata -------------------
    with st.expander("Run metadata (entropy)"):
        st.json(meta)

st.caption(
    "This MVP uses OpenAI if available; otherwise falls back to an offline dataset and subject-specific web resources."
)