from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

def evaluate_answers(query, graph_answer, vector_answer, ground_truth):

    prompt = f"""
You are an expert evaluator comparing two AI systems.

Question:
{query}

Ground Truth:
{ground_truth}

Graph RAG Answer:
{graph_answer}

Vector RAG Answer:
{vector_answer}

Evaluate BOTH systems on:

1. Correctness (0-10)
2. Reasoning Depth (0-10)
3. Use of Relationships (0-10)
4. Faithfulness (0-10)

Return ONLY JSON:

{{
  "graph": {{
    "correctness": ...,
    "reasoning": ...,
    "relationships": ...,
    "faithfulness": ...
  }},
  "vector": {{
    "correctness": ...,
    "reasoning": ...,
    "relationships": ...,
    "faithfulness": ...
  }},
  "winner": "graph/vector"
}}
"""

    response = llm.invoke(prompt).content

    try:
        return json.loads(response)
    except:
        return {"error": response}