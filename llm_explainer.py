# llm_explainer.py
import ollama
import json

SYSTEM_PROMPT = """
You are a scientific assistant for a histopathology AI system.

Rules:
- Be concise
- Be accurate
- Do not hallucinate
- If no data is available, say so clearly
"""

def explain(plan: dict, structured_answer: str | None):
    """
    Convert structured outputs + query plan into natural language
    """

    prompt = f"""
User intent:
{plan["intent"]}

Entities:
{json.dumps(plan["entities"], indent=2)}

Flags:
{json.dumps(plan["flags"], indent=2)}

Structured result:
{structured_answer}

Explain this to the user in simple, clear language.
"""

    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]