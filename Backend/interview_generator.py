import os
import re
import json
import logging
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from skill_gap import _normalize_skills
from db import get_connection, release_connection

load_dotenv()

router = APIRouter()

# ========================= MODELS =========================
class InterviewRequest(BaseModel):
    candidate_id: int
    job_id: int
    num_questions: int = 8
    focus: str = "balanced"   # balanced, technical, behavioral, leadership


class InterviewQuestion(BaseModel):
    question: str
    difficulty: str
    category: str
    rubric: str
    evidence: List[str]


class InterviewResponse(BaseModel):
    candidate_id: int
    candidate_name: str
    job_id: int
    job_title: str
    summary: str
    questions: List[InterviewQuestion]


# ========================= LLM SETUP =========================
def get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY is not set in .env file")

    logging.info("Using Groq API with model: llama-3.3-70b-versatile")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=groq_key,
        max_tokens=3000
    )


# ========================= HELPERS =========================
def extract_json(text: str) -> dict:
    """
    Robustly extract the first valid JSON object from a string.
    Handles markdown fences, preamble text, and trailing content.
    """
    # Strip common markdown fences first
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the outermost { ... } block
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        raise ValueError(f"No JSON object found in LLM response. Preview: {text[:300]}")

    return json.loads(json_match.group(0).strip())


# ========================= ENDPOINT =========================
@router.post("/generate-interview", response_model=InterviewResponse)
async def generate_interview(request: InterviewRequest):
    if request.num_questions < 4 or request.num_questions > 15:
        raise HTTPException(
            status_code=400,
            detail="Number of questions must be between 4 and 15"
        )

    conn = get_connection()
    cur = conn.cursor()

    try:
        # Fetch Job
        cur.execute(
            "SELECT title, description, skills FROM jobs WHERE job_id = %s",
            (request.job_id,)
        )
        job_row = cur.fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail="Job not found")
        job_title, job_desc, job_skills_raw = job_row
        job_skills = _normalize_skills(job_skills_raw)

        # Fetch Candidate
        cur.execute("""
            SELECT rs.structured_json,
                   COALESCE(rt.leadership, 0),
                   COALESCE(rt.communication, 0),
                   COALESCE(rt.analytical_thinking, 0),
                   COALESCE(rt.ownership, 0),
                   COALESCE(rt.problem_solving, 0),
                   COALESCE(rt.attention_to_detail, 0)
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            LEFT JOIN resume_traits rt ON r.id = rt.resume_id
            WHERE r.id = %s
        """, (request.candidate_id,))
        cand_row = cur.fetchone()

    finally:
        cur.close()
        release_connection(conn)

    if not cand_row:
        raise HTTPException(status_code=404, detail="Candidate not found")

    structured = cand_row[0]
    candidate_name = structured.get("name", f"Candidate {request.candidate_id}")
    candidate_skills = _normalize_skills(structured.get("skills", []))

    traits = {
        "leadership":           float(cand_row[1]),
        "communication":        float(cand_row[2]),
        "analytical_thinking":  float(cand_row[3]),
        "ownership":            float(cand_row[4]),
        "problem_solving":      float(cand_row[5]),
        "attention_to_detail":  float(cand_row[6]),
    }

    missing_skills = sorted(set(job_skills) - set(candidate_skills))

    prompt = f"""You are an expert technical recruiter designing a personalized interview for the role of {job_title}.

Candidate: {candidate_name}
Candidate Skills: {list(candidate_skills)}
Missing Skills: {missing_skills}
Inferred Traits (0-1 scale): {json.dumps({k: round(v, 2) for k, v in traits.items()})}

Generate exactly {request.num_questions} high-quality, candidate-specific interview questions.
Focus style: {request.focus}

Rules:
- Mix of Technical, Behavioral, Situational, and Trait-based questions.
- Target skill gaps and weak traits specifically.
- Questions should be natural and progressive in difficulty.
- For each question include difficulty (Easy/Medium/Hard), category, rubric, and evidence.

CRITICAL: Your entire response must be ONLY a raw JSON object. No explanation, no markdown, \
no code fences. Start your response with {{ and end with }}.

Required format:
{{
  "summary": "One short sentence describing the interview focus",
  "questions": [
    {{
      "question": "Full question text here",
      "difficulty": "Medium",
      "category": "Technical",
      "rubric": "Strong answer should demonstrate...",
      "evidence": ["Skill gap: Docker", "Low trait: Ownership"]
    }}
  ]
}}"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip()

        logging.info(f"Groq raw response (first 500 chars): {raw[:500]}")

        data = extract_json(raw)

        raw_questions = data.get("questions", [])
        if not raw_questions:
            raise HTTPException(
                status_code=500,
                detail="LLM returned no questions in the response."
            )

        questions = [
            InterviewQuestion(
                question=q.get("question", ""),
                difficulty=q.get("difficulty", "Medium"),
                category=q.get("category", "General"),
                rubric=q.get("rubric", "No rubric provided."),
                evidence=q.get("evidence", [])
            )
            for q in raw_questions
        ]

        return InterviewResponse(
            candidate_id=request.candidate_id,
            candidate_name=candidate_name,
            job_id=request.job_id,
            job_title=job_title,
            summary=data.get(
                "summary",
                f"Custom interview for {job_title} focusing on skill gaps and traits."
            ),
            questions=questions
        )

    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"JSON parsing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="LLM returned invalid JSON. Please try again."
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Interview generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate interview: {str(e)}"
        )