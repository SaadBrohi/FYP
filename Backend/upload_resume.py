import os
import shutil
import json
import numpy as np
import faiss
import logging
import traceback
import psycopg2  # ← Added for specific exception handling

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from resume_cleaner import clean_resume_file, save_cleaned_text
from llm_processor import extract_structured_json, save_json_output
from llm_pass_2 import infer_traits, save_traits_json
from embedding import model, flatten_resume_json
from db import get_connection, release_connection
from graph_builder import insert_candidate_graph

# -------------------------
# Setup directories
# -------------------------
UPLOAD_DIR = "Uploads"
CLEANED_DIR = "Cleaned"
JSON_DIR = "JSON"
TRAITS_DIR = "Traits_JSON"
FAISS_INDEX_DIR = "FAISS_Index"

for directory in [UPLOAD_DIR, CLEANED_DIR, JSON_DIR, TRAITS_DIR, FAISS_INDEX_DIR]:
    os.makedirs(directory, exist_ok=True)

# -------------------------
# FAISS paths
# -------------------------
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_index.faiss')
CANDIDATE_IDS_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_ids.json')
SNIPPET_METADATA_PATH = os.path.join(FAISS_INDEX_DIR, 'snippet_metadata.json')

# -------------------------
# FastAPI Router
# -------------------------
router = APIRouter()


def load_or_create_faiss_index():
    """Load FAISS index + metadata safely."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CANDIDATE_IDS_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CANDIDATE_IDS_PATH, 'r', encoding='utf-8') as f:
                candidate_ids = json.load(f)

            snippet_metadata = {}
            if os.path.exists(SNIPPET_METADATA_PATH):
                with open(SNIPPET_METADATA_PATH, 'r', encoding='utf-8') as f:
                    snippet_metadata = json.load(f)

            logging.info(f"Loaded FAISS index with {index.ntotal} vectors")
            return index, candidate_ids, snippet_metadata
        except Exception as e:
            logging.warning(f"Failed to load FAISS files: {e}. Creating fresh index.")

    # Create new index
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    return index, [], {}


def save_faiss_data(index, candidate_ids, snippet_metadata):
    """Persist FAISS index and metadata."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CANDIDATE_IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(candidate_ids, f, indent=2)
    with open(SNIPPET_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(snippet_metadata, f, indent=4)
    logging.info(f"FAISS index saved. Total vectors: {index.ntotal}")


@router.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    conn = None
    cur = None
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

        # 3. LLM Pass 1 - Structured JSON
        structured_json = extract_structured_json(cleaned_text)
        json_path = os.path.join(JSON_DIR, f"{name}_structured.json")
        save_json_output(structured_json, json_path)

        # 4. LLM Pass 2 - Traits
        trait_json = infer_traits(cleaned_text)
        traits_path = os.path.join(TRAITS_DIR, f"{name}_traits.json")
        save_traits_json(trait_json, traits_path)

        # 5. Create embedding (using only resume text for search consistency)
        resume_text = flatten_resume_json(structured_json)
        resume_embedding = model.encode(resume_text).astype('float32')

        # ====================== DATABASE OPERATIONS ======================
        conn = get_connection()
        cur = conn.cursor()

        # Insert into resumes table
        cur.execute("""
            INSERT INTO resumes (name, email, phone, raw_text, cleaned_text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            structured_json.get('name'),
            structured_json.get('email'),
            structured_json.get('phone'),
            cleaned_text,
            cleaned_text
        ))
        resume_id = cur.fetchone()[0]

        # Insert structured JSON
        cur.execute("""
            INSERT INTO resume_structured (resume_id, structured_json)
            VALUES (%s, %s)
        """, (resume_id, json.dumps(structured_json)))

        # Insert trait scores
        cur.execute("""
            INSERT INTO resume_traits (resume_id, leadership, communication,
                analytical_thinking, ownership, problem_solving, attention_to_detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            resume_id,
            trait_json.get('leadership', 0.0),
            trait_json.get('communication', 0.0),
            trait_json.get('analytical_thinking', 0.0),
            trait_json.get('ownership', 0.0),
            trait_json.get('problem_solving', 0.0),
            trait_json.get('attention_to_detail', 0.0)
        ))

        # ====================== FAISS OPERATIONS ======================
        index, candidate_ids, snippet_metadata = load_or_create_faiss_index()

        # Add embedding to FAISS
        index.add(np.expand_dims(resume_embedding, axis=0))
        faiss_idx = index.ntotal - 1

        # Update metadata
        candidate_ids.append(int(resume_id))
        snippet_metadata[str(faiss_idx)] = {
            "candidate_id": str(resume_id),
            "name": structured_json.get('name', 'Unknown'),
            "text": resume_text[:600] + "..." if len(resume_text) > 600 else resume_text
        }

        # Save FAISS data to disk
        save_faiss_data(index, candidate_ids, snippet_metadata)

        # ====================== Sync FAISS with PostgreSQL ======================
        cur.execute("""
            INSERT INTO resume_faiss_map (resume_id, faiss_index)
            VALUES (%s, %s)
            ON CONFLICT (resume_id) 
            DO UPDATE SET faiss_index = EXCLUDED.faiss_index
        """, (resume_id, faiss_idx))

        conn.commit()

        # Neo4j Graph Sync (non-blocking)
        try:
            insert_candidate_graph(
                resume_id=resume_id,
                structured_json=structured_json,
                traits=trait_json
            )
        except Exception as neo_err:
            logging.warning(f"Neo4j sync failed: {neo_err}")

        # ====================== SUCCESS RESPONSE ======================
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Resume processed successfully",
                "resume_id": int(resume_id),
                "faiss_index": int(faiss_idx),
                "name": structured_json.get('name', 'Unknown'),
                
                # Added back for frontend compatibility
                "structured_output": structured_json,
                "traits_output": trait_json,
                
                "files": {
                    "uploaded": upload_path,
                    "cleaned": cleaned_path,
                    "structured_json": json_path,
                    "traits_json": traits_path
                }
            }
        )

    except psycopg2.errors.UniqueViolation as db_err:
        logging.error(f"Database unique violation: {db_err}")
        if conn:
            conn.rollback()
        return JSONResponse(
            status_code=500, 
            content={"error": "Duplicate FAISS index detected. Please try again."}
        )

    except Exception as e:
        logging.error("Error in upload_resume: %s", str(e))
        logging.error(traceback.format_exc())

        if conn:
            conn.rollback()
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

    finally:
        if cur:
            cur.close()
        if conn:
            release_connection(conn)