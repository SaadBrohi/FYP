import os
import logging
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from db import get_connection, release_connection

load_dotenv()

router = APIRouter()

# ========================= CONFIG =========================
FAISS_INDEX_DIR = "FAISS_Index"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_index.faiss')
SNIPPET_METADATA_PATH = os.path.join(FAISS_INDEX_DIR, 'snippet_metadata.json')

# HARDCODED NEO4J CREDENTIALS
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "saadbrohi"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Trait weights (higher for analytical & problem-solving as they are critical)
TRAIT_WEIGHTS = {
    "leadership":           0.18,
    "communication":        0.15,
    "analytical_thinking":  0.25,
    "ownership":            0.12,
    "problem_solving":      0.22,
    "attention_to_detail":  0.08,
}

# ========================= MODELS =========================
class CompareRequest(BaseModel):
    candidate_ids: List[int]
    job_id: Optional[int] = None


# ========================= HELPERS =========================
def flatten_structured_json(structured: dict) -> str:
    """Rich text representation for embeddings."""
    parts = []
    if structured.get("name"): parts.append(structured["name"])
    if structured.get("summary"): parts.append(structured["summary"])

    skills = structured.get("skills", [])
    if skills:
        parts.append("Skills: " + ", ".join(skills))

    for exp in structured.get("experience", []):
        title = exp.get("job_title") or exp.get("role", "")
        company = exp.get("company", "")
        desc = exp.get("description", "")
        parts.append(f"{title} at {company}. {desc}")

    for edu in structured.get("education", []):
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        parts.append(f"{degree} from {institution}")

    return " ".join(parts)


def get_candidate_embeddings_batch(rows: list) -> Dict[int, np.ndarray]:
    ids = [row[0] for row in rows]
    texts = [flatten_structured_json(row[1]) for row in rows]
    vectors = embed_model.encode(texts, batch_size=16, normalize_embeddings=True).astype("float32")
    return {cid: vec for cid, vec in zip(ids, vectors)}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def compute_graph_signals(candidate_ids: List[int]) -> Dict[str, dict]:
    """Fetch graph signals from Neo4j."""
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        print(f"\n🛑 NEO4J AUTH/CONNECTION ERROR: {e}\n")
        raise HTTPException(status_code=500, detail=f"Neo4j Connection Error: {e}")

    results = {}

    try:
        with driver.session() as session:
            for cid in candidate_ids:
                cid_str = str(cid)

                # --- FIX: Retrieve record once, then check it ---
                
                # Skills
                skill_res = session.run(
                    "MATCH (c:Candidate {id: $cid})-[:HAS_SKILL]->(s:Skill) "
                    "RETURN collect(DISTINCT s.name) AS skills",
                    cid=cid_str
                )
                skill_record = skill_res.single()
                skills = skill_record["skills"] if skill_record else []

                # Experience
                exp_res = session.run(
                    "MATCH (c:Candidate {id: $cid})-[:WORKED_AT]->() RETURN count(*) AS count",
                    cid=cid_str
                )
                exp_record = exp_res.single()
                exp_count = exp_record["count"] if exp_record else 0

                # Education
                edu_res = session.run(
                    "MATCH (c:Candidate {id: $cid})-[:HAS_EDUCATION]->() RETURN count(*) AS count",
                    cid=cid_str
                )
                edu_record = edu_res.single()
                edu_count = edu_record["count"] if edu_record else 0

                results[cid_str] = {
                    "skills": skills,
                    "skill_count": len(skills),
                    "exp_count": int(exp_count),
                    "edu_count": int(edu_count),
                }

        driver.close()
        return results

    except Exception as e:
        print(f"\n🛑 NEO4J QUERY ERROR: {e}\n")
        raise HTTPException(status_code=500, detail=f"Neo4j Query Error: {e}")


def smart_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Robust normalization: handles equal values gracefully with absolute scaling."""
    if not values:
        return {}

    vals = list(values.values())
    min_v = min(vals)
    max_v = max(vals)

    if max_v - min_v < 0.01:
        max_abs = max(vals) + 1e-8
        return {k: min(0.3 + (v / max_abs) * 0.7, 1.0) for k, v in values.items()}

    return {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}


def compute_trait_score(traits: Dict[str, float]) -> float:
    return sum(traits.get(t, 0.0) * TRAIT_WEIGHTS.get(t, 0.0) for t in TRAIT_WEIGHTS)


# ========================= MAIN ENDPOINT =========================
@router.post("/compare-candidates")
async def compare_candidates(request: CompareRequest):
    if len(request.candidate_ids) < 2 or len(request.candidate_ids) > 5:
        raise HTTPException(status_code=400, detail="Please select between 2 and 5 candidates")

    # 1. PostgreSQL data
    conn = get_connection()
    cur = conn.cursor()
    placeholders = ','.join(['%s'] * len(request.candidate_ids))
    cur.execute(f"""
        SELECT r.id, rs.structured_json,
               COALESCE(rt.leadership, 0.0), COALESCE(rt.communication, 0.0),
               COALESCE(rt.analytical_thinking, 0.0), COALESCE(rt.ownership, 0.0),
               COALESCE(rt.problem_solving, 0.0), COALESCE(rt.attention_to_detail, 0.0)
        FROM resumes r
        JOIN resume_structured rs ON r.id = rs.resume_id
        LEFT JOIN resume_traits rt ON r.id = rt.resume_id
        WHERE r.id IN ({placeholders})
    """, request.candidate_ids)
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)

    if len(rows) != len(request.candidate_ids):
        raise HTTPException(status_code=404, detail="Some candidates not found")

    # 2. Embeddings
    candidate_embeddings = get_candidate_embeddings_batch(rows)

    # 3. Job embedding (if provided)
    job_embedding = None
    job_skills = []
    if request.job_id:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT title, description, skills FROM jobs WHERE job_id = %s", (request.job_id,))
        job_row = cur.fetchone()
        release_connection(conn)
        if job_row:
            job_text = f"{job_row[0] or ''} {job_row[1] or ''} {' '.join(job_row[2] if isinstance(job_row[2], list) else [])}"
            job_embedding = embed_model.encode(job_text, normalize_embeddings=True).astype("float32")
            job_skills = job_row[2] if isinstance(job_row[2], list) else []

    # 4. Graph Signals
    graph_raw = compute_graph_signals(request.candidate_ids)

    # Smart normalization
    norm_skills = smart_normalize({k: float(v["skill_count"]) for k, v in graph_raw.items()})
    norm_exp    = smart_normalize({k: float(v["exp_count"]) for k, v in graph_raw.items()})
    norm_edu    = smart_normalize({k: float(v["edu_count"]) for k, v in graph_raw.items()})

    # 5. Build results
    comparison_results = []

    for row in rows:
        candidate_id = row[0]
        structured = row[1]
        name = structured.get("name", f"Candidate {candidate_id}")
        cid_str = str(candidate_id)

        traits = {
            "leadership": float(row[2]),
            "communication": float(row[3]),
            "analytical_thinking": float(row[4]),
            "ownership": float(row[5]),
            "problem_solving": float(row[6]),
            "attention_to_detail": float(row[7]),
        }

        trait_score = compute_trait_score(traits)

        # === FINAL GRAPH SCORE ===
        skill_norm = norm_skills.get(cid_str, 0.0)
        exp_norm   = norm_exp.get(cid_str, 0.0)
        edu_norm   = norm_edu.get(cid_str, 0.0)

        skill_bonus = min(graph_raw.get(cid_str, {}).get("skill_count", 0) / 25.0, 1.0)

        graph_score = (
            skill_norm * 0.48 +
            exp_norm   * 0.32 +
            edu_norm   * 0.10 +
            skill_bonus * 0.10
        )

        # === EMBEDDING SIMILARITY ===
        cand_vec = candidate_embeddings[candidate_id]
        if job_embedding is not None:
            emb_sim = cosine_similarity(cand_vec, job_embedding)
            emb_type = "vs Job"
        else:
            vectors = list(candidate_embeddings.values())
            centroid = np.mean(vectors, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            emb_sim = cosine_similarity(cand_vec, centroid)
            emb_type = "vs Group Centroid"

        # Shared skills
        candidate_skills_set = set(s.lower() for s in graph_raw.get(cid_str, {}).get("skills", []))
        job_skills_set = set(s.lower() for s in job_skills)
        shared_skills = sorted(candidate_skills_set & job_skills_set) if job_skills_set else sorted(candidate_skills_set)[:12]

        # Overall Score
        overall = (trait_score * 0.40) + (graph_score * 0.32) + (emb_sim * 0.28)

        comparison_results.append({
            "candidate_id": int(candidate_id),
            "name": name,
            "overall_score": round(float(overall), 4),

            "trait_score": round(float(trait_score), 4),
            "trait_scores": {k: round(v, 3) for k, v in traits.items()},

            "graph_score": round(float(graph_score), 4),
            "graph_breakdown": {
                "skill_count": graph_raw.get(cid_str, {}).get("skill_count", 0),
                "exp_count": graph_raw.get(cid_str, {}).get("exp_count", 0),
                "edu_count": graph_raw.get(cid_str, {}).get("edu_count", 0),
            },

            "embedding_similarity": round(float(emb_sim), 4),
            "embedding_type": emb_type,

            "shared_skills": shared_skills,
            "structured_json": structured,

            "provenance": {
                "trait_source": "LLM inference (weighted)",
                "graph_source": "Neo4j + absolute skill bonus + group normalization",
                "embedding_source": "all-MiniLM-L6-v2"
            }
        })

    comparison_results.sort(key=lambda x: x["overall_score"], reverse=True)
    return comparison_results