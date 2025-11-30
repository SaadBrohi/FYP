from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neo4j import GraphDatabase

router = APIRouter()

# -------------------------
# Neo4j Connection Setup
# -------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "saadbrohi"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------
# Helper Function
# -------------------------
def fetch_graph(tx):
    nodes_query = "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, n.name AS name, n.type AS type"
    nodes_result = tx.run(nodes_query)
    nodes = [
        {
            "data": {
                "id": str(record["id"]),
                "label": record["name"] or record["labels"][0],
                "type": record["labels"][0]
            }
        }
        for record in nodes_result
    ]

    rels_query = "MATCH (a)-[r]->(b) RETURN id(a) AS source, id(b) AS target, type(r) AS relation"
    rels_result = tx.run(rels_query)
    edges = [
        {
            "data": {
                "source": str(record["source"]),
                "target": str(record["target"]),
                "label": record["relation"]
            }
        }
        for record in rels_result
    ]

    return {"nodes": nodes, "edges": edges}

# -------------------------
# Endpoint
# -------------------------
@router.get("/graph-data")
async def get_graph_data():
    try:
        with driver.session() as session:
            graph_data = session.read_transaction(fetch_graph)
        return JSONResponse(status_code=200, content=graph_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------------
# Shutdown event
# -------------------------
@router.on_event("shutdown")
def shutdown_event():
    driver.close()
