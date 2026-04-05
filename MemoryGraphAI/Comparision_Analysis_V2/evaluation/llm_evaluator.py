# from langchain_groq import ChatGroq
# import os, json
# from dotenv import load_dotenv

# load_dotenv(dotenv_path="../.env") 

# llm = ChatGroq(
#     temperature=0,
#     model_name="llama-3.1-8b-instant",
#     groq_api_key=os.getenv("GROQ_API_KEY"),
# )

# def extract_json(text):
#     """
#     Extract first valid JSON object from text safely
#     """
#     matches = re.findall(r"\{.*?\}", text, re.DOTALL)

#     for m in matches:
#         try:
#             return json.loads(m)
#         except:
#             continue

#     raise ValueError("No valid JSON found in LLM response")

# def judge(query, hybrid_ans, vector_ans, gt):

#     prompt = f"""
# Evaluate two systems.

# Question: {query}
# Ground Truth: {gt}

# Hybrid Answer:
# {hybrid_ans}

# Vector Answer:
# {vector_ans}

# Score each (0-10):

# 1. Correctness
# 2. Reasoning Depth
# 3. Relationship Usage
# 4. Faithfulness
# 5. Context Richness

# Return JSON:
# {{
#  "hybrid": {{
#    "correctness": 0,
#    "reasoning": 0,
#    "relationships": 0,
#    "faithfulness": 0,
#    "richness": 0
#  }},
#  "vector": {{
#    "correctness": 0,
#    "reasoning": 0,
#    "relationships": 0,
#    "faithfulness": 0,
#    "richness": 0
#  }},
#  "winner": "hybrid/vector"
# }}
# """

#     res = llm.invoke(prompt).content
#     return json.loads(res[res.find("{"):res.rfind("}")+1])

from langchain_groq import ChatGroq
import os
import json
from dotenv import load_dotenv

# ─────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────
load_dotenv(dotenv_path="../.env")

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ─────────────────────────────────────────
# STRICT JSON PARSER
# ─────────────────────────────────────────
def safe_parse_json(text):
    try:
        return json.loads(text)
    except:
        print("\n❌ RAW LLM OUTPUT (INVALID JSON):\n", text)
        return None


# ─────────────────────────────────────────
# JUDGE FUNCTION (NO BIAS)
# ─────────────────────────────────────────
def judge(query, hybrid_ans, vector_ans, gt):

    prompt = f"""
You are a strict evaluator.

Return ONLY VALID JSON.
No explanation.
No extra text.

Compare two answers.

Question: {query}
Ground Truth: {gt}

Hybrid Answer:
{hybrid_ans}

Vector Answer:
{vector_ans}

Return EXACT JSON:

{{
 "hybrid": {{
   "correctness": int,
   "reasoning": int,
   "relationships": int,
   "faithfulness": int,
   "richness": int
 }},
 "vector": {{
   "correctness": int,
   "reasoning": int,
   "relationships": int,
   "faithfulness": int,
   "richness": int
 }}
}}
"""

    res = llm.invoke(prompt).content.strip()

    parsed = safe_parse_json(res)

    if parsed is None:
        return {"error": True}

    # ✅ deterministic winner (NO bias)
    def total(d):
        return sum(d.values())

    hybrid_score = total(parsed["hybrid"])
    vector_score = total(parsed["vector"])

    if hybrid_score > vector_score:
        parsed["winner"] = "hybrid"
    elif vector_score > hybrid_score:
        parsed["winner"] = "vector"
    else:
        parsed["winner"] = "tie"

    return parsed