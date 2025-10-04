# langchain_agent.py
import os
from dotenv import load_dotenv

load_dotenv()

def create_learnmate_agent():
    """
    Simple agent using OpenAIFunctions (most reliable)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"🔑 API Key loaded: {bool(api_key)}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return None
    
    try:
        from langchain_community.chat_models import ChatOpenAI
        from langchain.agents import initialize_agent, AgentType
        from langchain_tools import get_tools
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=api_key
        )
        
        print("✅ LLM initialized")
        
        tools = get_tools()
        print(f"✅ Tools loaded: {len(tools)}")
        
        # Use OpenAIFunctions agent (most stable)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True
        )
        
        print("✅ LangChain agent created successfully!")
        return agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Create agent instance
learnmate_agent = create_learnmate_agent()

# Debug output
if learnmate_agent:
    print("🎉 LangChain agent is READY!")
else:
    print("💥 LangChain agent creation FAILED")