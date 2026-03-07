import os
import json
import requests
import re

GROQ_API_KEY = ""
with open("doc.txt", "r", encoding="utf-8") as f:
    doc_text = f.read()

prompt = f"""
Extract key entities and relationships from the following text.  
Return JSON with two top-level keys:  
1) "nodes": a list of unique entities,  
2) "edges": a list of [entity1, entity2] relationships.

Text:
\"\"\"{doc_text}\"\"\"
"""

data = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "system", "content": "Extract entities and relationships in JSON."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.0,
    "max_tokens": 2048
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers=headers,
    data=json.dumps(data)
)

try:
    result = response.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in LLM output")
    clean_json_str = json_match.group(0)

    extracted = json.loads(clean_json_str)
    
    nodes = extracted.get("nodes", [])
    edges = extracted.get("edges", [])

    print("Extracted Nodes:", nodes)
    print("Extracted Edges:", edges)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)

    print("\noutput.json created successfully!")

except Exception as e:
    print("Failed to extract JSON from LLM output:")
    print(str(e))
    print("Raw response:", response.text)