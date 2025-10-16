# langchain_tools.py
from langchain.tools import Tool

def get_tools():
    """Create and return all tools with safe wrappers to avoid keyword issues"""
    # Lazy imports
    from agents import (
        planner_agent,
        wikipedia_agent,
        tutor_agent,
        explainer_agent,
        web_agent,
        validator_agent,
        retrieve_lesson_from_embeddings
    )

    # ------------------- Planner Tool -------------------
    planner_tool = Tool(
        name="TopicPlanner",
        description="Analyzes user requests to determine the main topic and plan subtasks",
        func=planner_agent
    )

    # ------------------- Wikipedia Tool -------------------
    wikipedia_tool = Tool(
        name="WikipediaSearch",
        description="Fetches summaries and information from Wikipedia for any topic",
        func=wikipedia_agent
    )

    # ------------------- Tutor Tool (Code Only) -------------------
    def tutor_wrapper(payload):
        # Accept both dict and string
        if isinstance(payload, dict):
            topic = payload.get("topic", "")
            user_prompt = payload.get("input", f"Explain {topic}")
        else:
            topic = payload
            user_prompt = f"Explain {topic}"
        return tutor_agent(topic, user_prompt)

    tutor_tool = Tool(
        name="ProgrammingTutor",
        description="Generates Python code examples for Computer Science topics only.",
        func=tutor_wrapper
    )

    # ------------------- Explainer Tool (Explanation Only) -------------------
    def explainer_wrapper(payload):
        if isinstance(payload, dict):
            topic = payload.get("topic", "")
            user_prompt = payload.get("input", f"Explain {topic}")
        else:
            topic = payload
            user_prompt = f"Explain {topic}"
        return explainer_agent(topic, user_prompt)

    explainer_tool = Tool(
        name="Explainer",
        description="Provides clear explanations for any topic without code.",
        func=explainer_wrapper
    )

    # ------------------- Web Resources Tool -------------------
    def web_wrapper(*args, **kwargs):
        # LangChain may pass multiple args; use 'subject' if available
        payload = kwargs.get("subject") or (args[0] if args else "General")
        return web_agent(payload)

    web_tool = Tool(
        name="WebResources",
        description="Finds curated educational resources based on subject.",
        func=web_wrapper
    )

    # ------------------- Retrieval Tool -------------------
    retrieval_tool = Tool(
        name="LessonRetrieval",
        description="Searches existing lesson database using semantic similarity to find relevant past lessons",
        func=retrieve_lesson_from_embeddings
    )

    # ------------------- Validator Tool -------------------
    def validator_wrapper(payload):
        if isinstance(payload, str):
            return {
                "topic": "unknown",
                "explanation": payload,
                "code": "",
                "links": [],
                "embedding": []
            }
        lesson = validator_agent(payload)
        if "embedding" not in lesson:
            lesson["embedding"] = []
        return lesson

    validator_tool = Tool(
        name="OutputValidator",
        description="Validates and formats the final lesson output to ensure it has explanation, code, links, and embedding",
        func=validator_wrapper
    )

    return [
        planner_tool,
        retrieval_tool,
        wikipedia_tool,
        tutor_tool,
        explainer_tool,
        web_tool,
        validator_tool
    ]