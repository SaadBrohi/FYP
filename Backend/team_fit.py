import os
import json
import logging
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq

from db import get_connection, release_connection
from neo4j_client import neo4j_client

load_dotenv()

router = APIRouter()

TRAIT_KEYS = [
    "leadership", "communication", "analytical_thinking",
    "ownership", "problem_solving", "attention_to_detail"
]


# ========================= MODELS =========================

class TeamFitRequest(BaseModel):
    candidate_id: int
    team_member_ids: List[int]


class MemberFitScore(BaseModel):
    member_id: int
    member_name: str
    trait_complementarity_score: float
    skill_proximity_score: float
    graph_shared_skills: int
    overall_pairwise_fit: float


class TeamFitResponse(BaseModel):
    candidate_id: int
    candidate_name: str
    team_size: int
    overall_team_fit_score: float
    candidate_traits: Dict[str, float]
    team_avg_traits: Dict[str, float]
    skill_overlap: List[str]
    unique_skills_brought: List[str]
    member_scores: List[MemberFitScore]
    counterfactual_suggestions: List[str]
    llm_analysis: str
    summary: str


# ========================= LLM SETUP =========================

def get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY is not set in .env file")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=groq_key,
        max_tokens=2000
    )


# ========================= HELPERS =========================

def extract_skills_list(raw_skills) -> List[str]:
    """
    Mirrors graph_builder.py skill extraction so results stay consistent
    with what was actually stored in Neo4j.
    Handles: plain strings, "Category: s1, s2" strings, and nested lists.
    """
    skills_list = []
    for entry in raw_skills:
        if isinstance(entry, str):
            if ":" in entry:
                entry = entry.split(":", 1)[1].strip()
            skills_list.extend([s.strip() for s in entry.split(",") if s.strip()])
        elif isinstance(entry, (list, tuple)):
            skills_list.extend([str(s).strip() for s in entry if s])
        else:
            skills_list.append(str(entry).strip())

    seen: set = set()
    return [s for s in skills_list if s and not (s in seen or seen.add(s))]


def fetch_candidate_data(candidate_id: int) -> Optional[dict]:
    """
    Pull structured JSON + trait scores from PostgreSQL for one candidate.
    Returns None when the ID does not exist.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT rs.structured_json,
                   rt.leadership, rt.communication, rt.analytical_thinking,
                   rt.ownership, rt.problem_solving, rt.attention_to_detail
            FROM resumes r
            JOIN  resume_structured rs ON r.id = rs.resume_id
            LEFT JOIN resume_traits  rt ON r.id = rt.resume_id
            WHERE r.id = %s
        """, (candidate_id,))
        row = cur.fetchone()
    finally:
        cur.close()
        release_connection(conn)

    if not row:
        return None

    structured = row[0]
    traits = {
        "leadership":          float(row[1] or 0.0),
        "communication":       float(row[2] or 0.0),
        "analytical_thinking": float(row[3] or 0.0),
        "ownership":           float(row[4] or 0.0),
        "problem_solving":     float(row[5] or 0.0),
        "attention_to_detail": float(row[6] or 0.0),
    }

    return {
        "name":       structured.get("name", f"Candidate {candidate_id}"),
        "traits":     traits,
        "skills":     extract_skills_list(structured.get("skills", [])),
        "experience": structured.get("experience", []),
    }


# ========================= SCORING =========================

def compute_trait_complementarity(
    candidate_traits: dict,
    team_traits: List[dict]
) -> float:
    """
    Scores how well the candidate fills the team's trait gaps.

    Per trait:
      gap  = how far the team average falls below the 0.70 threshold
      fill = candidate_value * gap   — rewards plugging weak spots
      base = candidate_value * 0.30  — unconditional strength bonus so a
             well-rounded candidate still scores well on a well-rounded team

    Returns float in [0.0, 1.0].
    """
    if not team_traits:
        return 0.75  # no team yet — neutral score

    team_avg = {t: float(np.mean([m[t] for m in team_traits])) for t in TRAIT_KEYS}

    per_trait = []
    for trait in TRAIT_KEYS:
        cand_val = candidate_traits.get(trait, 0.0)
        gap      = max(0.0, 0.70 - team_avg[trait])
        fill     = cand_val * gap
        base     = cand_val * 0.30
        per_trait.append(fill + base)

    # Theoretical max per trait: 1.0*0.70 + 1.0*0.30 = 1.0
    return float(np.clip(np.mean(per_trait), 0.0, 1.0))


def compute_skill_proximity(skills_a: List[str], skills_b: List[str]) -> float:
    """Jaccard similarity between two skill sets (case-insensitive)."""
    set_a = {s.lower() for s in skills_a}
    set_b = {s.lower() for s in skills_b}
    if not set_a and not set_b:
        return 0.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def fetch_shared_skills_neo4j(candidate_id: int, member_id: int) -> int:
    """
    Count Skill nodes shared between two Candidate nodes via HAS_SKILL edges.
    Returns 0 on any graph error so the rest of scoring is unaffected.
    """
    try:
        result = neo4j_client.run("""
            MATCH (c1:Candidate {id: $cid1})-[:HAS_SKILL]->(s:Skill)
                  <-[:HAS_SKILL]-(c2:Candidate {id: $cid2})
            RETURN count(s) AS shared
        """, {"cid1": str(candidate_id), "cid2": str(member_id)})
        for record in result:
            return int(record["shared"] or 0)
    except Exception as e:
        logging.error(
            f"Neo4j shared-skills error "
            f"(candidate={candidate_id}, member={member_id}): {e}"
        )
    return 0


# ========================= ENDPOINT =========================

@router.post("/team-fit", response_model=TeamFitResponse)
async def team_fit_analysis(request: TeamFitRequest):

    # ── Validation ──────────────────────────────────────────────────────────
    if not request.team_member_ids:
        raise HTTPException(
            status_code=400,
            detail="team_member_ids must contain at least one member"
        )
    if request.candidate_id in request.team_member_ids:
        raise HTTPException(
            status_code=400,
            detail="candidate_id cannot also appear in team_member_ids"
        )

    # ── Fetch candidate ──────────────────────────────────────────────────────
    candidate = fetch_candidate_data(request.candidate_id)
    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {request.candidate_id} not found"
        )

    # ── Fetch team members (silently skip unresolvable IDs) ─────────────────
    team_members = []
    for mid in request.team_member_ids:
        member = fetch_candidate_data(mid)
        if member:
            member["id"] = mid
            team_members.append(member)
        else:
            logging.warning(f"Team member ID {mid} not found — skipping")

    if not team_members:
        raise HTTPException(
            status_code=404,
            detail="None of the provided team_member_ids resolved to valid candidates"
        )

    # ── Trait analysis ────────────────────────────────────────────────────────
    team_traits_list = [m["traits"] for m in team_members]
    team_avg_traits  = {
        t: float(np.mean([m[t] for m in team_traits_list]))
        for t in TRAIT_KEYS
    }
    trait_comp_score = compute_trait_complementarity(candidate["traits"], team_traits_list)

    # ── Skill analysis ────────────────────────────────────────────────────────
    all_team_skills_lower = set()
    for m in team_members:
        all_team_skills_lower.update(s.lower() for s in m["skills"])

    skill_overlap    = [s for s in candidate["skills"] if s.lower() in all_team_skills_lower]
    unique_skills    = [s for s in candidate["skills"] if s.lower() not in all_team_skills_lower]
    uniqueness_ratio = len(unique_skills) / max(len(candidate["skills"]), 1)

    # ── Pairwise member scores ────────────────────────────────────────────────
    member_scores = []
    for member in team_members:
        t_comp       = compute_trait_complementarity(candidate["traits"], [member["traits"]])
        s_prox       = compute_skill_proximity(candidate["skills"], member["skills"])
        shared_count = fetch_shared_skills_neo4j(request.candidate_id, member["id"])
        graph_bonus  = min(0.10, shared_count * 0.02)  # capped at 0.10

        pairwise = float(np.clip(
            t_comp * 0.50 + s_prox * 0.40 + graph_bonus,
            0.0, 1.0
        ))

        member_scores.append(MemberFitScore(
            member_id=member["id"],
            member_name=member["name"],
            trait_complementarity_score=round(t_comp, 3),
            skill_proximity_score=round(s_prox, 3),
            graph_shared_skills=shared_count,
            overall_pairwise_fit=round(pairwise, 3),
        ))

    # ── Overall team fit score ────────────────────────────────────────────────
    avg_pairwise  = float(np.mean([m.overall_pairwise_fit for m in member_scores]))
    overall_score = float(np.clip(
        trait_comp_score * 0.45 +   # fills team's trait gaps
        avg_pairwise     * 0.35 +   # pairwise harmony across all members
        uniqueness_ratio * 0.20,    # fresh skills only this candidate introduces
        0.0, 1.0
    ))

    # ── LLM analysis ─────────────────────────────────────────────────────────
    team_summary_for_llm = [
        {
            "name":   m["name"],
            "traits": {k: round(v, 2) for k, v in m["traits"].items()},
            "skills": m["skills"][:12],
        }
        for m in team_members
    ]

    prompt = f"""
You are an expert talent analyst specialising in team dynamics and organisational psychology.

**Candidate Being Evaluated:** {candidate["name"]}
**Candidate Trait Scores (0-1 scale):**
{json.dumps({k: round(v, 2) for k, v in candidate["traits"].items()}, indent=2)}
**Candidate Skills (top 20):** {json.dumps(candidate["skills"][:20])}

**Existing Team ({len(team_members)} member(s)):**
{json.dumps(team_summary_for_llm, indent=2)}

**Computed Metrics:**
- Trait Complementarity Score : {trait_comp_score:.2f}
- Average Pairwise Fit Score  : {avg_pairwise:.2f}
- Overall Team Fit Score      : {overall_score:.2f}
- Team Trait Averages         : {json.dumps({k: round(v, 2) for k, v in team_avg_traits.items()})}
- Skills Shared with Team     : {skill_overlap[:10]}
- Unique Skills Candidate Adds: {unique_skills[:10]}

Provide:
1. A thorough team-fit analysis covering trait complementarity, collaboration
   potential, and skill contribution (2-3 paragraphs).
2. Exactly 3-4 specific counterfactual pairing suggestions, e.g.
   "This candidate pairs especially well with [member] because..."
   or "If the team is weak in [trait], pairing this candidate with [member] would..."
3. A single-sentence executive summary.

Return ONLY valid JSON — no markdown fences, no text outside the JSON object:
{{
  "llm_analysis": "Detailed multi-paragraph analysis...",
  "counterfactual_suggestions": [
    "Suggestion 1...",
    "Suggestion 2...",
    "Suggestion 3..."
  ],
  "summary": "One-sentence executive summary."
}}
"""

    # Sensible defaults — endpoint always returns computed scores even if LLM fails
    llm_analysis               = "LLM analysis is currently unavailable."
    counterfactual_suggestions = []
    summary = (
        f"{candidate['name']} has an overall team fit score of "
        f"{overall_score:.0%} based on trait complementarity and skill proximity."
    )

    try:
        llm  = get_llm()
        resp = llm.invoke(prompt)
        text = resp.content

        if not text or not text.strip():
            raise ValueError("LLM returned an empty response")

        text = text.strip()
        logging.info(f"Team fit LLM raw response (first 300 chars): {text[:300]}")

        # Robust fence stripping
        if "```json" in text:
            text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed                     = json.loads(text)
        llm_analysis               = parsed.get("llm_analysis",              llm_analysis)
        counterfactual_suggestions = parsed.get("counterfactual_suggestions", counterfactual_suggestions)
        summary                    = parsed.get("summary",                    summary)

    except json.JSONDecodeError as e:
        logging.error(f"Team fit LLM JSON parse error: {e}\nRaw snippet: {text[:500]}")
    except Exception as e:
        logging.error(f"Team fit LLM call failed: {e}")

    # ── Response ──────────────────────────────────────────────────────────────
    return TeamFitResponse(
        candidate_id=request.candidate_id,
        candidate_name=candidate["name"],
        team_size=len(team_members),
        overall_team_fit_score=round(overall_score, 3),
        candidate_traits=candidate["traits"],
        team_avg_traits={k: round(v, 3) for k, v in team_avg_traits.items()},
        skill_overlap=skill_overlap,
        unique_skills_brought=unique_skills,
        member_scores=member_scores,
        counterfactual_suggestions=counterfactual_suggestions,
        llm_analysis=llm_analysis,
        summary=summary,
    )