# langchain_agent.py
import os
from dotenv import load_dotenv

load_dotenv()

def create_learnmate_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key loaded: {bool(api_key)}")
    
    if not api_key:
        print("OPENAI_API_KEY not found")
        return None
    
    try:
        from langchain_community.chat_models import ChatOpenAI
        from langchain.agents import initialize_agent, AgentType
        from langchain_tools import get_tools
        
        # ------------------- LLM Initialization -------------------
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=api_key
        )
        print("LLM initialized")
        
        # ------------------- Tools -------------------
        tools = get_tools()
        print(f"Tools loaded: {len(tools)}")
        
        # ------------------- System Prompt -------------------
        # Tells the agent how to behave and choose tools dynamically
        system_message = """
        You are LearnMate, an intelligent tutor agent.
        - Analyze the user's prompt and subject.
        - Decide which tools to call (Planner, Wikipedia, Tutor, Explainer, Web Resources, Embedding Retrieval).
        - Only generate Python code if the subject is 'Computer Science'.
        - Provide structured output: topic, explanation, code, links.
        - Avoid unnecessary steps; use retrieval first if possible.
        """
        
        # ------------------- Initialize Agent -------------------
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            system_message=system_message
        )
        
        print("LangChain agent created successfully!")
        return agent
        
    except Exception as e:
        print(f"Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ------------------- Create Agent -------------------
learnmate_agent = create_learnmate_agent()

# ------------------- Debug Output -------------------
if learnmate_agent:
    print("LangChain agent is READY!")
else:
    print("LangChain agent creation FAILED")