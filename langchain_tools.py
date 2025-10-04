# langchain_tools.py
from langchain.tools import Tool

def get_tools():
    """Create and return all tools with lazy imports to avoid circular imports"""
    # Import inside function to avoid circular imports
    from agents import (
        planner_agent, 
        wikipedia_agent, 
        tutor_agent, 
        web_agent, 
        validator_agent,
        retrieve_lesson_from_embeddings
    )
    
    # Planner tool
    planner_tool = Tool(
        name="TopicPlanner",
        description="Analyzes user requests to determine the main topic and plan subtasks",
        func=planner_agent
    )

    # Wikipedia tool
    wikipedia_tool = Tool(
        name="WikipediaSearch",
        description="Fetches summaries and information from Wikipedia for any topic",
        func=wikipedia_agent
    )

    # Tutor tool - fixed signature for LangChain
    def tutor_wrapper(topic):
        """Wrapper for tutor_agent that works with LangChain tool calling"""
        return tutor_agent(topic, f"Explain {topic}", explain=True, code=True)
    
    tutor_tool = Tool(
        name="ProgrammingTutor",
        description="Explains programming concepts and generates Python code examples. Use for computer science and programming topics.",
        func=tutor_wrapper
    )

    # Web resources tool - fixed signature
    def web_wrapper(topic):
        """Wrapper for web_agent with default subject"""
        return web_agent(topic, "General")  # Default subject
    
    web_resources_tool = Tool(
        name="WebResources",
        description="Finds curated educational resources and tutorials for various subjects",
        func=web_wrapper
    )

    # Retrieval tool
    retrieval_tool = Tool(
        name="LessonRetrieval",
        description="Searches existing lesson database using semantic similarity to find relevant past lessons",
        func=retrieve_lesson_from_embeddings
    )

    # Validator tool - fixed to handle string inputs
    def validator_wrapper(payload):
        """Wrapper for validator_agent that handles both dict and string inputs"""
        # If payload is a string, convert it to a proper dict structure
        if isinstance(payload, str):
            return {
                "topic": "unknown",
                "explanation": payload,
                "code": "",
                "links": []
            }
        else:
            # Ensure the payload has all required fields
            return validator_agent(payload)
    
    validator_tool = Tool(
        name="OutputValidator",
        description="Validates and formats the final lesson output to ensure it has explanation, code, and links",
        func=validator_wrapper
    )
    
    return [
        planner_tool,
        retrieval_tool, 
        wikipedia_tool,
        tutor_tool,
        web_resources_tool,
        validator_tool
    ]

# Optional: Individual tool getters if needed elsewhere
def get_planner_tool():
    from agents import planner_agent
    return Tool(name="TopicPlanner", description="Analyzes user requests", func=planner_agent)

def get_tutor_tool():
    from agents import tutor_agent
    def tutor_wrapper(topic):
        return tutor_agent(topic, f"Explain {topic}", explain=True, code=True)
    return Tool(name="ProgrammingTutor", description="Explains programming concepts", func=tutor_wrapper)

def get_web_tool():
    from agents import web_agent
    def web_wrapper(topic):
        return web_agent(topic, "General")
    return Tool(name="WebResources", description="Finds educational resources", func=web_wrapper)

def get_wikipedia_tool():
    from agents import wikipedia_agent
    return Tool(name="WikipediaSearch", description="Fetches Wikipedia summaries", func=wikipedia_agent)

def get_retrieval_tool():
    from agents import retrieve_lesson_from_embeddings
    return Tool(name="LessonRetrieval", description="Searches lesson database", func=retrieve_lesson_from_embeddings)

def get_validator_tool():
    from agents import validator_agent
    def validator_wrapper(payload):
        if isinstance(payload, str):
            return {
                "topic": "unknown",
                "explanation": payload,
                "code": "",
                "links": []
            }
        return validator_agent(payload)
    return Tool(name="OutputValidator", description="Validates lesson output", func=validator_wrapper)