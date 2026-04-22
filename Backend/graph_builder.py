# graph_builder.py
from neo4j_client import neo4j_client
import logging

def insert_candidate_graph(resume_id: int, structured_json: dict, traits: dict):
    """
    Insert candidate data into Neo4j graph.
    Uses MERGE to avoid duplicates.
    """
    try:
        candidate_id = str(resume_id)
        name = structured_json.get("name", "Unknown")

        # --- Candidate Node ---
        neo4j_client.run("""
            MERGE (c:Candidate {id: $id})
            SET c.name = $name,
                c.updated_at = datetime()
        """, {"id": candidate_id, "name": name})

        # --- Skills ---
        raw_skills = structured_json.get("skills", [])
        skills_list = []

        for entry in raw_skills:
            if isinstance(entry, str):
                # Handle "Category: skill1, skill2"
                if ":" in entry:
                    entry = entry.split(":", 1)[1].strip()
                # Split by comma
                split_skills = [s.strip() for s in entry.split(",") if s.strip()]
                skills_list.extend(split_skills)
            elif isinstance(entry, (list, tuple)):
                skills_list.extend([str(s).strip() for s in entry if s])
            else:
                # Single item
                skills_list.append(str(entry).strip())

        # Remove duplicates while preserving order
        seen = set()
        unique_skills = [skill for skill in skills_list if skill and not (skill in seen or seen.add(skill))]

        for skill in unique_skills:
            neo4j_client.run("""
                MERGE (s:Skill {name: $skill})
                MERGE (c:Candidate {id: $id})-[:HAS_SKILL]->(s)
            """, {"skill": skill, "id": candidate_id})

        # --- Experience ---
        for exp in structured_json.get("experience", []):
            company = exp.get("company")
            job_title = exp.get("job_title") or exp.get("role", "")

            if not company:
                continue

            neo4j_client.run("""
                MERGE (cmp:Company {name: $company})
                MERGE (r:Role {name: $role})
                MERGE (c:Candidate {id: $id})-[:WORKED_AT]->(cmp)
                MERGE (c)-[:PERFORMED_ROLE]->(r)
            """, {
                "company": company,
                "role": job_title,
                "id": candidate_id
            })

        # --- Education ---
        for edu in structured_json.get("education", []):
            institution = edu.get("institution")
            degree = edu.get("degree", "")

            if not institution:
                continue

            neo4j_client.run("""
                MERGE (e:Education {institution: $inst, degree: $degree})
                MERGE (c:Candidate {id: $id})-[:HAS_EDUCATION]->(e)
            """, {
                "inst": institution,
                "degree": degree,
                "id": candidate_id
            })

        # --- Traits ---
        for trait_name, score in traits.items():
            neo4j_client.run("""
                MERGE (t:Trait {name: $trait})
                MERGE (c:Candidate {id: $id})-[:EXHIBITS {score: $score}]->(t)
            """, {
                "trait": trait_name,
                "score": float(score),
                "id": candidate_id
            })

        logging.info(f"Successfully inserted candidate {resume_id} into Neo4j graph")

    except Exception as e:
        logging.error(f"Error inserting candidate {resume_id} into Neo4j: {e}")
        raise
