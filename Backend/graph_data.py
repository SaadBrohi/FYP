# graph_data.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neo4j import GraphDatabase
import logging

router = APIRouter()

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "saadbrohi"

# Create driver (singleton - best practice)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_graph(tx):
    """Fetch nodes and relationships for visualization."""
    # Nodes query
    nodes_query = """
    MATCH (n)
    RETURN 
        id(n) AS internal_id,
        labels(n) AS labels,
        n.name AS name,
        n.institution AS institution,
        n.degree AS degree,
        n.id AS candidate_id
    """
    nodes_result = tx.run(nodes_query)
    
    nodes = []
    for record in nodes_result:
        labels = record["labels"]
        node_type = labels[0] if labels else "Unknown"
        
        # Determine display label
        if node_type == "Candidate":
            display_label = record["candidate_id"] or "Candidate"
        elif node_type == "Education":
            display_label = f"{record['degree']} @ {record['institution']}" if record['degree'] else record["institution"] or "Education"
        else:
            display_label = record["name"] or node_type

        nodes.append({
            "id": str(record["internal_id"]),
            "label": display_label,
            "type": node_type
        })

    # Relationships query
    rels_query = """
    MATCH (a)-[r]->(b)
    RETURN 
        id(a) AS source,
        id(b) AS target,
        type(r) AS relation,
        r.score AS score
    """
    rels_result = tx.run(rels_query)
    
    edges = []
    for record in rels_result:
        edge = {
            "source": str(record["source"]),
            "target": str(record["target"]),
            "label": record["relation"]
        }
        if record["score"] is not None:
            edge["score"] = float(record["score"])
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


@router.get("/graph-data")
async def get_graph_data():
    try:
        with driver.session() as session:
            graph_data = session.execute_read(fetch_graph)
        
        return JSONResponse(
            status_code=200, 
            content=graph_data
        )
        
    except Exception as e:
        logging.error(f"Error fetching graph data: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e), "message": "Failed to fetch graph data from Neo4j"}
        )


# Proper shutdown (recommended)
@router.on_event("shutdown")
def shutdown_event():
    try:
        driver.close()
        logging.info("Neo4j driver closed successfully")
    except Exception as e:
        logging.warning(f"Error closing Neo4j driver: {e}")