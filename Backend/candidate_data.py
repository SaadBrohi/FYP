from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from db import get_connection, release_connection
import json

router = APIRouter()

@router.get("/candidate/{candidate_id}")
async def get_candidate_data(candidate_id: int):
    """
    Fetch structured JSON + trait scores for a specific candidate.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Structured JSON
        cur.execute("""
            SELECT structured_json
            FROM resume_structured
            WHERE resume_id = %s
        """, (candidate_id,))
        structured_row = cur.fetchone()
        if not structured_row:
            raise HTTPException(status_code=404, detail="Candidate not found")
        structured_json = structured_row[0]

        # Trait scores
        cur.execute("""
            SELECT leadership, communication, analytical_thinking,
                   ownership, problem_solving, attention_to_detail
            FROM resume_traits
            WHERE resume_id = %s
        """, (candidate_id,))
        traits_row = cur.fetchone()
        traits_json = {
            "leadership": traits_row[0],
            "communication": traits_row[1],
            "analytical_thinking": traits_row[2],
            "ownership": traits_row[3],
            "problem_solving": traits_row[4],
            "attention_to_detail": traits_row[5]
        } if traits_row else {}

        cur.close()
        release_connection(conn)

        return JSONResponse(
            status_code=200,
            content={
                "candidate_id": candidate_id,
                "structured_json": structured_json,
                "traits": traits_json
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})
