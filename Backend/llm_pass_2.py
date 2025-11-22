import os
import json
import logging
import re
from langchain_ollama import OllamaLLM

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Directories and files
# -------------------------
TRAITS_JSON_DIR = "Traits_JSON"
PROMPT_FILE = "prompt_templates/traits_extraction.txt"
os.makedirs(TRAITS_JSON_DIR, exist_ok=True)

# -------------------------
# Load prompt template
# -------------------------
def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# Infer traits from resume text
# -------------------------
def infer_traits(clean_text: str, llm_model: str = "qwen2.5:7b-instruct-q5_K_M") -> dict:
    prompt_text = load_prompt_template(PROMPT_FILE)
    prompt_text = prompt_text.replace("{resume}", clean_text)

    try:
        logging.info("Calling LLM model: %s", llm_model)
        llm = OllamaLLM(model=llm_model, temperature=0)
        response = llm.generate([prompt_text])
        response_text = response.generations[0][0].text
        logging.info("LLM response received (first 500 chars): %s", response_text[:500])

        # Strip Markdown code fences if present
        response_text = re.sub(r'^```json|```$', '', response_text.strip(), flags=re.MULTILINE).strip()

        return json.loads(response_text)

    except json.JSONDecodeError:
        logging.warning("Failed to parse LLM output as JSON, returning empty trait structure.")
    except Exception as e:
        logging.error("LLM call failed: %s", str(e))

    return {
        "leadership": 0.0,
        "communication": 0.0,
        "analytical_thinking": 0.0,
        "ownership": 0.0,
        "problem_solving": 0.0,
        "attention_to_detail": 0.0
    }

# -------------------------
# Save traits JSON
# -------------------------
# Dynamically create any subfolders in Traits_JSON if included in filename
def save_traits_json(trait_json: dict, filename: str):
    path = os.path.join(TRAITS_JSON_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trait_json, f, indent=4)