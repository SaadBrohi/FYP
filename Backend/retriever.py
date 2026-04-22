from sentence_transformers import SentenceTransformer
import faiss
from neo4j import GraphDatabase
import numpy as np
import json

class HybridRetriever:
    def __init__(self, faiss_index_path, snippet_metadata_path, neo4j_uri, neo4j_user, neo4j_pass):
        # Load FAISS index
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Load snippet metadata
        with open(snippet_metadata_path, "r", encoding='utf-8') as f:
            self.snippet_metadata = json.load(f)
        
        # Load embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        
        # Scoring weights
        self.alpha = 0.6
        self.beta = 0.4

    def embed_query(self, query_text):
        return self.embed_model.encode(query_text).astype('float32')

    def search_faiss(self, query_vector, top_k=20):
        D, I = self.faiss_index.search(np.array([query_vector]), top_k)
        
        top_snippets = []
        top_sim = []
        for idx, d in zip(I[0], D[0]):
            if idx == -1:
                continue
            str_idx = str(idx)
            if str_idx in self.snippet_metadata:
                top_snippets.append(self.snippet_metadata[str_idx])
                top_sim.append(float(1 / (1 + d)))   # ← Convert to native float
                
        return top_snippets, top_sim

    def get_graph_score(self, candidate_id, query_entities):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Candidate {id: $cid})-[r]-(n)
                    RETURN type(r) AS rel_type, n.name AS neighbor
                """, cid=candidate_id)
                
                score = 0
                paths = []
                for record in result:
                    paths.append(f"{candidate_id} -[{record['rel_type']}]-> {record['neighbor']}")
                    if record['neighbor'] and record['neighbor'].lower() in query_entities:
                        score += 1
                return float(score), paths                  # ← Convert to native float
        except Exception as e:
            print(f"Graph query failed for {candidate_id}: {e}")
            return 0.0, []

    def retrieve(self, query_text, top_k=5):
        query_entities = [word.lower() for word in query_text.split()]
        query_vector = self.embed_query(query_text)
        
        fetch_k = max(20, top_k * 2)
        total_in_index = self.faiss_index.ntotal
        fetch_k = min(fetch_k, total_in_index) if total_in_index > 0 else fetch_k
        
        top_snippets, dense_sim = self.search_faiss(query_vector, top_k=fetch_k)
        
        results = []
        seen_candidates = set()
        
        for snippet, sim in zip(top_snippets, dense_sim):
            candidate_id = snippet.get("candidate_id")
            if not candidate_id or candidate_id in seen_candidates:
                continue
            
            graph_score, graph_paths = self.get_graph_score(candidate_id, query_entities)
            
            composite = self.alpha * sim + self.beta * graph_score
            
            results.append({
                "candidate_id": str(candidate_id),
                "snippet": snippet.get("text", "")[:200],
                "graph_path": graph_paths[0] if graph_paths else None,
                "dense_sim": float(sim),           # ← Force native float
                "graph_score": float(graph_score), # ← Force native float
                "composite_score": float(composite) # ← Force native float
            })
            seen_candidates.add(candidate_id)
        
        # Rank and return top_k
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results[:top_k]