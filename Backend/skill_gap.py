# skill_gap.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from db import get_connection, release_connection

router = APIRouter()


class JobCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    skills: List[str]
    required_traits: Optional[Dict[str, float]] = None


def _normalize_skills(skills) -> set:
    if not skills:
        return set()

    # Handle PostgreSQL array string format: {flutter,dart,kotlin}
    if isinstance(skills, str):
        s = skills.strip()
        if s.startswith('{') and s.endswith('}'):
            skills = [x.strip().strip('"\'') for x in s[1:-1].split(',')]
        else:
            try:
                skills = json.loads(s)
            except:
                skills = [s]

    result = set()
    for s in skills:
        s = str(s).strip()
        if not s:
            continue
        # Handle "Category: skill1, skill2, skill3" format
        if ':' in s:
            s = s.split(':', 1)[1]
        # Split by comma to extract individual skills
        for part in s.split(','):
            part = part.strip().lower()
            if part:
                result.add(part)
    return result


def _calculate_trait_match(candidate_traits: dict, required_traits: dict) -> float:
    if not required_traits:
        return 0.0
    total_weight = sum(required_traits.values())
    if total_weight == 0:
        return 0.0

    score = 0.0
    for trait, required_score in required_traits.items():
        candidate_score = float(candidate_traits.get(trait, 0.0))
        score += min(candidate_score / required_score, 1.0) * required_score

    return round((score / total_weight) * 100, 1)


# GET /jobs
@router.get("/jobs")
async def get_all_jobs():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT job_id, title, description, skills, required_traits FROM jobs ORDER BY job_id")
        rows = cur.fetchall()
        cur.close()
        release_connection(conn)

        jobs = [
            {
                "job_id": row[0],
                "title": row[1],
                "description": row[2] or "",
                "skills": row[3] if row[3] else [],
                "required_traits": row[4] if row[4] else {}
            }
            for row in rows
        ]
        return JSONResponse(status_code=200, content=jobs)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# POST /jobs
@router.post("/jobs")
async def create_job(job: JobCreate):
    try:
        conn = get_connection()
        cur = conn.cursor()

        title = job.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="Job title cannot be empty")

        required_traits = job.required_traits or {}

        cur.execute(
            "INSERT INTO jobs (title, description, skills, required_traits) VALUES (%s, %s, %s, %s) RETURNING job_id",
            (title, job.description.strip() if job.description else "", job.skills, json.dumps(required_traits))
        )
        job_id = cur.fetchone()[0]
        conn.commit()

        cur.close()
        release_connection(conn)

        return JSONResponse(
            status_code=201,
            content={
                "message": "Job created successfully",
                "job_id": job_id,
                "title": title,
                "skills": job.skills,
                "required_traits": required_traits
            }
        )

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            release_connection(conn)

        error_str = str(e).lower()
        if "unique_job_title" in error_str or "duplicate key" in error_str:
            return JSONResponse(status_code=409, content={"error": "A job with this title already exists"})

        return JSONResponse(status_code=500, content={"error": str(e)})


# GET /skill-gap/{job_id}/{candidate_id}
@router.get("/skill-gap/{job_id}/{candidate_id}")
async def get_skill_gap(job_id: int, candidate_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT title, skills, required_traits FROM jobs WHERE job_id = %s", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_title, job_skills_raw, required_traits = job_row
        job_skills = _normalize_skills(job_skills_raw)
        required_traits = required_traits or {}

        cur.execute("""
            SELECT rs.structured_json,
                   COALESCE(rt.leadership, 0.0), COALESCE(rt.communication, 0.0),
                   COALESCE(rt.analytical_thinking, 0.0), COALESCE(rt.ownership, 0.0),
                   COALESCE(rt.problem_solving, 0.0), COALESCE(rt.attention_to_detail, 0.0)
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            LEFT JOIN resume_traits rt ON r.id = rt.resume_id
            WHERE r.id = %s
        """, (candidate_id,))

        candidate_row = cur.fetchone()
        if not candidate_row:
            raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")

        structured_json = candidate_row[0]
        candidate_name = structured_json.get("name", f"Candidate {candidate_id}")
        candidate_skills = _normalize_skills(structured_json.get("skills", []))

        candidate_traits = {
            "leadership": candidate_row[1],
            "communication": candidate_row[2],
            "analytical_thinking": candidate_row[3],
            "ownership": candidate_row[4],
            "problem_solving": candidate_row[5],
            "attention_to_detail": candidate_row[6]
        }

        matched_skills = sorted(job_skills & candidate_skills)
        missing_skills = sorted(job_skills - candidate_skills)
        skill_match_pct = round((len(matched_skills) / len(job_skills) * 100) if job_skills else 0.0, 1)
        trait_match_pct = _calculate_trait_match(candidate_traits, required_traits)
        final_score = round((skill_match_pct * 0.70) + (trait_match_pct * 0.30), 1)

        cur.close()
        release_connection(conn)

        return JSONResponse(status_code=200, content={
            "job_id": job_id,
            "job_title": job_title,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            # Frontend-expected field names
            "match_percentage": final_score,
            "skill_match_percentage": skill_match_pct,
            "trait_match_percentage": trait_match_pct,
            "final_score": final_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_count": len(matched_skills),
            "missing_count": len(missing_skills),
            "required_traits": required_traits,
            "candidate_traits": candidate_traits
        })

    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})


# GET /rank-candidates/{job_id}
@router.get("/rank-candidates/{job_id}")
async def rank_candidates(job_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT title, skills, required_traits FROM jobs WHERE job_id = %s", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_title, job_skills_raw, required_traits = job_row
        job_skills = _normalize_skills(job_skills_raw)
        required_traits = required_traits or {}
        total_required = len(job_skills)

        cur.execute("""
            SELECT r.id, rs.structured_json,
                   COALESCE(rt.leadership, 0.0), COALESCE(rt.communication, 0.0),
                   COALESCE(rt.analytical_thinking, 0.0), COALESCE(rt.ownership, 0.0),
                   COALESCE(rt.problem_solving, 0.0), COALESCE(rt.attention_to_detail, 0.0)
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            LEFT JOIN resume_traits rt ON r.id = rt.resume_id
            ORDER BY r.id
        """)
        rows = cur.fetchall()
        cur.close()
        release_connection(conn)

        ranked = []
        for row in rows:
            candidate_id = row[0]
            structured_json = row[1]
            candidate_name = structured_json.get("name", f"Candidate {candidate_id}")
            raw_skills = structured_json.get("skills", [])
            candidate_skills = _normalize_skills(raw_skills)

            candidate_traits = {
                "leadership": row[2], "communication": row[3], "analytical_thinking": row[4],
                "ownership": row[5], "problem_solving": row[6], "attention_to_detail": row[7]
            }

            matched = sorted(job_skills & candidate_skills)
            missing = sorted(job_skills - candidate_skills)
            skill_pct = round((len(matched) / total_required * 100) if total_required > 0 else 0.0, 1)
            trait_pct = _calculate_trait_match(candidate_traits, required_traits)
            final_score = round((skill_pct * 0.70) + (trait_pct * 0.30), 1)

            ranked.append({
                "rank": 0,
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "skill_match": skill_pct,
                "trait_match": trait_pct,
                "final_score": final_score,
                "match_percentage": final_score,       # Frontend expects this
                "matched_skills": matched,
                "missing_skills": missing,
                "matched_count": len(matched),         # Frontend expects this
                "missing_count": len(missing),         # Frontend expects this
                "candidate_traits": candidate_traits
            })

        ranked.sort(key=lambda x: -x["final_score"])
        for i, item in enumerate(ranked):
            item["rank"] = i + 1

        return JSONResponse(status_code=200, content={
            "job_id": job_id,
            "job_title": job_title,
            "total_required_skills": total_required,
            "candidates": ranked
        })

    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})


# GET /recommend-jobs/{candidate_id}
@router.get("/recommend-jobs/{candidate_id}")
async def recommend_jobs(candidate_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Fetch candidate
        cur.execute("""
            SELECT rs.structured_json,
                   COALESCE(rt.leadership, 0.0), COALESCE(rt.communication, 0.0),
                   COALESCE(rt.analytical_thinking, 0.0), COALESCE(rt.ownership, 0.0),
                   COALESCE(rt.problem_solving, 0.0), COALESCE(rt.attention_to_detail, 0.0)
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            LEFT JOIN resume_traits rt ON r.id = rt.resume_id
            WHERE r.id = %s
        """, (candidate_id,))

        candidate_row = cur.fetchone()
        if not candidate_row:
            raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")

        structured_json = candidate_row[0]
        candidate_name = structured_json.get("name", f"Candidate {candidate_id}")
        candidate_skills = _normalize_skills(structured_json.get("skills", []))

        candidate_traits = {
            "leadership": candidate_row[1],
            "communication": candidate_row[2],
            "analytical_thinking": candidate_row[3],
            "ownership": candidate_row[4],
            "problem_solving": candidate_row[5],
            "attention_to_detail": candidate_row[6]
        }

        # Fetch all jobs
        cur.execute("SELECT job_id, title, skills, required_traits FROM jobs ORDER BY job_id")
        job_rows = cur.fetchall()
        cur.close()
        release_connection(conn)

        recommended = []
        for job_row in job_rows:
            job_id, job_title, job_skills_raw, required_traits = job_row
            job_skills = _normalize_skills(job_skills_raw)
            required_traits = required_traits or {}
            total_required = len(job_skills)

            matched = sorted(job_skills & candidate_skills)
            missing = sorted(job_skills - candidate_skills)
            skill_pct = round((len(matched) / total_required * 100) if total_required > 0 else 0.0, 1)
            trait_pct = _calculate_trait_match(candidate_traits, required_traits)
            final_score = round((skill_pct * 0.70) + (trait_pct * 0.30), 1)

            recommended.append({
                "rank": 0,
                "job_id": job_id,
                "job_title": job_title,
                "match_percentage": final_score,
                "skill_match_percentage": skill_pct,
                "trait_match_percentage": trait_pct,
                "total_required_skills": total_required,
                "matched_skills": matched,
                "missing_skills": missing,
                "matched_count": len(matched),
                "missing_count": len(missing)
            })

        recommended.sort(key=lambda x: -x["match_percentage"])
        for i, item in enumerate(recommended):
            item["rank"] = i + 1

        return JSONResponse(status_code=200, content={
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "candidate_skills": sorted(candidate_skills),
            "recommended_jobs": recommended
        })

    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})