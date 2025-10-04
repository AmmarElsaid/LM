# langchain_pipeline.py
from langchain_agent import learnmate_agent
import time

def extract_topic_from_response(langchain_result):
    """Extract topic from LangChain response"""
    output = langchain_result.get("output", "")
    if output:
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                topic = line.split('.')[0]
                topic = topic.replace('**', '').replace('#', '').strip()
                if topic and len(topic) > 5:
                    return topic
    return "unknown"

def extract_code_from_response(langchain_result):
    """Extract code from LangChain response"""
    output = langchain_result.get("output", "")
    if "```python" in output:
        code_start = output.find("```python") + 9
        code_end = output.find("```", code_start)
        if code_end > code_start:
            return output[code_start:code_end].strip()
    return ""

def format_langchain_output(langchain_result, subject):
    """Convert LangChain output to lesson format"""
    output = langchain_result.get("output", "")
    topic = extract_topic_from_response(langchain_result)
    code = extract_code_from_response(langchain_result) if subject == "Computer Science" else ""
    
    return {
        "topic": topic,
        "explanation": output,
        "code": code or "# No code example generated.",
        "links": [{"title": "See explanation above", "link": "", "snippet": "Resources included in explanation"}]
    }

def run_pipeline_langchain(user_prompt: str, subject: str = None) -> dict:
    """LangChain-based pipeline"""
    if learnmate_agent is None:
        # Fallback to original pipeline or return error
        from agents import run_pipeline as original_pipeline
        print("⚠️ Using original pipeline (LangChain agent not available)")
        return original_pipeline(user_prompt, subject)
    
    try:
        start_time = time.time()
        result = learnmate_agent.invoke({
            "input": f"User request: {user_prompt}. Subject: {subject}",
            "subject": subject
        })
        
        return {
            "meta": {
                "completed": True,
                "prompts_used": 1,
                "agent_failures": 0,
                "total_time_sec": time.time() - start_time,
                "method": "langchain"
            },
            "lesson": format_langchain_output(result, subject)
        }
        
    except Exception as e:
        # Fallback to original pipeline on error
        from agents import run_pipeline as original_pipeline
        print(f"⚠️ LangChain failed, using original pipeline: {e}")
        return original_pipeline(user_prompt, subject)