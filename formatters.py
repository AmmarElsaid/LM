# formatters.py
import re
from embedding_manager import EmbeddingManager
from agents import web_agent

# ------------------- Base Formatter -------------------
class BaseFormatter:
    def __init__(self, embedder: EmbeddingManager, web_agent):
        self.embedder = embedder
        self.web_agent = web_agent

    def format(self, langchain_result: dict) -> dict:
        raise NotImplementedError

# ------------------- Computer Science Formatter -------------------
class CSFormatter(BaseFormatter):
    def format(self, langchain_result: dict) -> dict:
        output = langchain_result.get("output", "").strip()

        # Clean output
        output = output.replace("Explanation\n", "").replace("Explanation", "").strip()
        output_lines = [line for line in output.split("\n") if "Your request:" not in line]
        output = "\n".join(output_lines).strip()

        # Topic extraction
        topic = langchain_result.get("topic")
        if not topic or topic.lower() in ["unknown", "none", ""]:
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            topic = lines[0] if lines else "Unknown"

        # Extract Python code
        code = ""
        if "```python" in output:
            code_start = output.find("```python") + 9
            code_end = output.find("```", code_start)
            if code_end > code_start:
                code = output[code_start:code_end].strip()
                output = output.replace(f"```python\n{code}\n```", "").strip()
                for header in ["Python Code Example", "Implementation in Python", "Example Code"]:
                    if header in output:
                        output = output.replace(header, f"{header}\n\n```python\n{code}\n```")
                        break
                else:
                    output += f"\n\n```python\n{code}\n```"

        # Remove duplicate Additional Resources
        output = re.sub(r'Additional Resources.*', '', output, flags=re.DOTALL)

        # Curated links
        links = self.web_agent(subject="Computer Science")

        # Embedding
        embedding = self.embedder.embed_text(output)

        return {"topic": topic, "explanation": output, "code": code, "links": links, "embedding": embedding}

# ------------------- Non-CS Formatter -------------------
class NonCSFormatter(BaseFormatter):
    def format(self, langchain_result: dict) -> dict:
        output = langchain_result.get("output", "").strip()

        # Clean output
        output = output.replace("Explanation\n", "").replace("Explanation", "").strip()
        output_lines = [line for line in output.split("\n") if "Your request:" not in line]
        output = "\n".join(output_lines).strip()

        # Topic extraction
        topic = langchain_result.get("topic")
        if not topic or topic.lower() in ["unknown", "none", ""]:
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            topic = lines[0] if lines else "Unknown"

        # Remove all code
        output = re.sub(r'```.*?```', '', output, flags=re.DOTALL)

        code_keywords = [
            "python code example", "implementation in python", "example code",
            "python example", "here is a simple example", "code snippet"
        ]
        resource_keywords = [
            "resources", "for more detailed explanations",
            "additional resources", "if you need further details"
        ]

        cleaned_lines = []
        for line in output.split("\n"):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in code_keywords + resource_keywords):
                continue
            line = re.sub(r'`.*?`', '', line)  # Remove inline code
            cleaned_lines.append(line)
        output = "\n".join(cleaned_lines).strip()

        # Curated links
        links = self.web_agent(subject=langchain_result.get("subject", "General"))

        # Embedding
        embedding = self.embedder.embed_text(output)

        return {"topic": topic, "explanation": output, "code": "", "links": links, "embedding": embedding}

# ------------------- Math/Physics Formatter -------------------
class MathPhysicsFormatter(NonCSFormatter):
    def format(self, langchain_result: dict) -> dict:
        output_dict = super().format(langchain_result)
        output = output_dict["explanation"]

        # Remove any leftover code/backticks
        output = re.sub(r'```.*?```', '', output, flags=re.DOTALL)
        output = re.sub(r'`.*?`', '', output)

        # Convert common formulas to LaTeX
        def latexify(match):
            expr = match.group(0)
            return f"\\({expr}\\)"

        output = re.sub(r'\([^\)]+\)\^[0-9]+', latexify, output)
        output = re.sub(r'\b\d+!\b', latexify, output)
        output = re.sub(r'\b[a-zA-Z]+\!\b', latexify, output)
        output = re.sub(
            r'(\b[a-zA-Z0-9!]+\s*)/(\s*[a-zA-Z0-9!]+)',
            lambda m: f"\\frac{{{m.group(1).strip()}}}{{{m.group(2).strip()}}}",
            output
        )

        output_dict["explanation"] = output
        return output_dict

# ------------------- Formatter Factory -------------------
def get_formatter(subject: str, embedder: EmbeddingManager = None, web_agent_fn=None):
    embedder = embedder or EmbeddingManager(use_openai=False)
    web_agent_fn = web_agent_fn or web_agent

    if subject == "Computer Science":
        return CSFormatter(embedder, web_agent_fn)
    elif subject in ["Math", "Physics"]:
        return MathPhysicsFormatter(embedder, web_agent_fn)
    else:
        return NonCSFormatter(embedder, web_agent_fn)