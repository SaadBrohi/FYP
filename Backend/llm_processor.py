import os
import json
import logging
from langchain_ollama import OllamaLLM  # Updated import

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Directories and files
# -------------------------
CLEANED_DIR = "Cleaned"
JSON_DIR = "JSON"
PROMPT_FILE = "prompt_templates/resume_extraction.txt"

os.makedirs(JSON_DIR, exist_ok=True)

# -------------------------
# Load prompt template
# -------------------------
def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# Extract structured JSON from resume text
# -------------------------
def extract_structured_json(cleaned_text: str, llm_model: str = "qwen2.5:7b-instruct-q5_K_M") -> dict:
    prompt_text = load_prompt_template(PROMPT_FILE)
    prompt_text = prompt_text.replace("{resume_text}", cleaned_text)

    try:
        logging.info("Calling LLM model: %s", llm_model)
        llm = OllamaLLM(model=llm_model, temperature=0)
        response = llm.generate([prompt_text])
        response_text = response.generations[0][0].text
        logging.info("LLM response received (first 500 chars): %s", response_text[:500])

        return json.loads(response_text)

    except json.JSONDecodeError:
        logging.warning("Failed to parse LLM output as JSON, returning empty structure.")
    except Exception as e:
        logging.error("LLM call failed: %s", str(e))

    return {
        "name": "",
        "email": "",
        "phone": "",
        "education": [],
        "experience": [],
        "skills": [],
        "projects": [],
        "certifications": []
    }


# -------------------------
# Save JSON output
# -------------------------
def save_json_output(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
