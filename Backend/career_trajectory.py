import os
import json
import logging
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from db import get_connection, release_connection
from neo4j_client import neo4j_client   # ← use the shared singleton

load_dotenv()

router = APIRouter()

# ========================= MODELS =========================
class CareerTrajectoryRequest(BaseModel):
    candidate_id: int
    num_paths: int = 3

class CareerPath(BaseModel):
    predicted_role: str
    company_type: str
    time_to_promotion_years: float
    probability: float
    key_skills_needed: List[str]
    evidence: List[str]
    rationale: str

class CareerTrajectoryResponse(BaseModel):
    candidate_id: int
    candidate_name: str
    current_role: str
    current_company: str
    predicted_paths: List[CareerPath]
    summary: str


# ========================= LLM SETUP =========================
def get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY is not set in .env file")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        api_key=groq_key,
        max_tokens=3000
    )


# ========================= ENDPOINT =========================
@router.post("/career-trajectory", response_model=CareerTrajectoryResponse)
async def career_trajectory(request: CareerTrajectoryRequest):
    conn = get_connection()
    cur = conn.cursor()

    # Get candidate basic info from PostgreSQL
    cur.execute("""
        SELECT rs.structured_json 
        FROM resumes r
        JOIN resume_structured rs ON r.id = rs.resume_id
        WHERE r.id = %s
    """, (request.candidate_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        release_connection(conn)
        raise HTTPException(status_code=404, detail="Candidate not found")

    structured = row[0]
    candidate_name = structured.get("name", f"Candidate {request.candidate_id}")
    current_role = structured.get("experience", [{}])[-1].get("job_title", "Unknown")
    current_company = structured.get("experience", [{}])[-1].get("company", "Unknown")

    cur.close()
    release_connection(conn)

    # Get career history from Neo4j using the shared client
    history = []
    try:
        results = neo4j_client.run("""
            MATCH (c:Candidate {id: $cid})-[:WORKED_AT|PERFORMED_ROLE]->(r)
            OPTIONAL MATCH (c)-[p:PROMOTED_TO]->(next)
            RETURN 
                r.name AS role,
                r.company AS company,
                p.years AS promotion_years,
                p.date AS date
            ORDER BY p.date ASC
        """, {"cid": str(request.candidate_id)})

        for record in results:
            history.append({
                "role": record["role"],
                "company": record["company"],
                "promotion_years": record["promotion_years"]
            })

    except Exception as e:
        logging.error(f"Neo4j error in career trajectory: {e}")
        history = []   # non-blocking — LLM will still run with what it has

    # Build prompt
    prompt = f"""
You are an expert career advisor and talent analyst.

**Candidate:** {candidate_name}
**Current Role:** {current_role} at {current_company}
**Career History:** {json.dumps(history, indent=2)}

Based on the candidate's skills, traits, and career progression, predict their most likely future career paths.

Generate exactly {request.num_paths} realistic career paths.

For each path provide:
- Predicted next role
- Likely company type (Startup, MNC, Product Company, etc.)
- Estimated years to next promotion
- Probability (0-100%)
- Key skills needed to achieve it
- Clear evidence/rationale from current profile

Return ONLY valid JSON in this format (no markdown fences, no extra text):
{{
  "summary": "One sentence career trajectory summary",
  "paths": [
    {{
      "predicted_role": "...",
      "company_type": "...",
      "time_to_promotion_years": 2.5,
      "probability": 75,
      "key_skills_needed": ["Docker", "Leadership"],
      "evidence": ["Fast promotion history", "Strong analytical thinking trait"],
      "rationale": "Based on current trajectory and skills..."
    }}
  ]
}}
"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        response_text = response.content

        if not response_text or not response_text.strip():
            raise ValueError("LLM returned an empty response")

        response_text = response_text.strip()
        logging.info(f"Career trajectory raw response (first 300 chars): {response_text[:300]}")

        # Strip markdown code fences robustly
        if "```json" in response_text:
            response_text = response_text.split("```json", 1)[1]
        if "```" in response_text:
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()

        data = json.loads(response_text)

        paths = []
        for p in data.get("paths", []):
            paths.append(CareerPath(
                predicted_role=p.get("predicted_role", ""),
                company_type=p.get("company_type", ""),
                time_to_promotion_years=float(p.get("time_to_promotion_years", 0)),
                probability=float(p.get("probability", 0)),
                key_skills_needed=p.get("key_skills_needed", []),
                evidence=p.get("evidence", []),
                rationale=p.get("rationale", "")
            ))

        return CareerTrajectoryResponse(
            candidate_id=request.candidate_id,
            candidate_name=candidate_name,
            current_role=current_role,
            current_company=current_company,
            predicted_paths=paths,
            summary=data.get("summary", "AI-generated career trajectory based on current profile and industry trends.")
        )

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM JSON response: {e}\nRaw text: {response_text}")
        raise HTTPException(status_code=500, detail=f"LLM returned malformed JSON: {str(e)}")

    except Exception as e:
        logging.error(f"Career trajectory generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate career trajectory: {str(e)}")