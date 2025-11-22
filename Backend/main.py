import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from resume_cleaner import clean_resume_file, save_cleaned_text
from llm_processor import extract_structured_json, save_json_output
from llm_pass_2 import infer_traits, save_traits_json

from embedding import model, flatten_resume_json, flatten_traits_json
import numpy as np
import faiss
import json

# -------------------------
# Setup directories
# -------------------------
UPLOAD_DIR = "Uploads"
CLEANED_DIR = "Cleaned"
JSON_DIR = "JSON"
TRAITS_DIR = "Traits_JSON"
FAISS_INDEX_DIR = "FAISS_Index"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(TRAITS_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# -------------------------
# FAISS index paths
# -------------------------
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_index.faiss')
CANDIDATE_IDS_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_ids.json')

# -------------------------
# Load or create FAISS index
# -------------------------
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CANDIDATE_IDS_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CANDIDATE_IDS_PATH, 'r', encoding='utf-8') as f:
        candidate_ids = json.load(f)
else:
    index = None
    candidate_ids = []

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="TalentScope Full Resume Pipeline")

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        name = os.path.splitext(file.filename)[0]

        # 1. Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Clean resume
        cleaned_text = clean_resume_file(upload_path)
        cleaned_path = os.path.join(CLEANED_DIR, f"{name}_cleaned.txt")
        save_cleaned_text(cleaned_text, cleaned_path)

        # 3. LLM Pass 1: Structured JSON
        structured_json = extract_structured_json(cleaned_text)
        json_path = os.path.join(JSON_DIR, f"{name}_structured.json")
        save_json_output(structured_json, json_path)

        # 4. LLM Pass 2: Trait Extraction
        trait_json = infer_traits(cleaned_text)
        traits_path = os.path.join(TRAITS_DIR, f"{name}_traits.json")
        save_traits_json(trait_json, traits_path)

        # 5. Embeddings + Add to FAISS
        resume_text = flatten_resume_json(structured_json)
        traits_text = flatten_traits_json(trait_json)

        resume_embedding = model.encode(resume_text)
        traits_embedding = model.encode(traits_text)

        combined_embedding = np.concatenate(
            [resume_embedding, traits_embedding]
        ).astype('float32')

        global index, candidate_ids

        # Initialize FAISS index if first time
        if index is None:
            index = faiss.IndexFlatL2(len(combined_embedding))

        # Add new embedding
        index.add(np.expand_dims(combined_embedding, axis=0))
        candidate_ids.append(name)

        # Save updated index + IDs
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(CANDIDATE_IDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(candidate_ids, f)

        # 6. Final response
        return JSONResponse(
            status_code=200,
            content={
                "message": "Resume fully processed successfully",
                "uploaded_file": upload_path,
                "cleaned_file": cleaned_path,
                "structured_json_file": json_path,
                "traits_json_file": traits_path,
                "structured_output": structured_json,
                "traits_output": trait_json
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
