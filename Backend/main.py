import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from resume_cleaner import clean_resume_file, save_cleaned_text
from llm_processor import extract_structured_json, save_json_output

# -------------------------
# Setup directories
# -------------------------
UPLOAD_DIR = "Uploads"
CLEANED_DIR = "Cleaned"
JSON_DIR = "JSON"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="TalentScope Full Resume Pipeline")

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # 1. Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Clean resume
        cleaned_text = clean_resume_file(upload_path)

        cleaned_path = os.path.join(
            CLEANED_DIR, f"{os.path.splitext(file.filename)[0]}_cleaned.txt"
        )
        save_cleaned_text(cleaned_text, cleaned_path)

        # 3. Run LLM structured extraction
        structured_json = extract_structured_json(cleaned_text)

        # 4. Save JSON output
        json_path = os.path.join(
            JSON_DIR, f"{os.path.splitext(file.filename)[0]}_structured.json"
        )
        save_json_output(structured_json, json_path)

        # 5. Return output
        return JSONResponse(
            status_code=200,
            content={
                "message": "Resume fully processed successfully",
                "uploaded_file": upload_path,
                "cleaned_file": cleaned_path,
                "json_file": json_path,
                "structured_output": structured_json
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
