from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn  # add this import

# Import routers
from upload_resume import router as upload_resume_router
from graph_data import router as graph_data_router
from candidate_data import router as candidate_router

app = FastAPI(
    title="TalentScope AI Backend",
    description="Backend for TalentScope AI: Resume Upload, Candidate Graph, and Candidate Details",
    version="1.0.0"
)

# -------------------------
# CORS Middleware
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Include routers
# -------------------------
app.include_router(upload_resume_router, tags=["Resume Upload"])
app.include_router(graph_data_router, tags=["Graph Data"])
app.include_router(candidate_router, tags=["Candidate Data"])

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/", tags=["Root"])
async def root():
    return {"message": "TalentScope AI Backend is running!"}

# -------------------------
# Run server automatically
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
