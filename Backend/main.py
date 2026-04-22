from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import routers
from upload_resume import router as upload_resume_router
from graph_data import router as graph_data_router
from candidate_data import router as candidate_router
from retriever_api import router as retriever_router
from skill_gap import router as skill_gap_router
from compare_api import router as compare_router
from career_trajectory import router as career_trajectory_router
from team_fit import router as team_fit_router 

app = FastAPI(
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
app.include_router(retriever_router, tags=["Retriever Search"])
app.include_router(skill_gap_router, tags=["Skill Gap Analysis"])
app.include_router(compare_router, tags=["Candidate Comparison"])
app.include_router(interview_router, tags=["Interview Generator"])
app.include_router(career_trajectory_router, tags=["Career Trajectory"])
app.include_router(team_fit_router, tags=["Team Fit Analysis"])
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
