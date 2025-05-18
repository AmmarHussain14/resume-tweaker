from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ResumeRequest(BaseModel):
    resume: str
    job_description: str

# Root route
@app.get("/")
async def root():
    return {"message": "Resume Tweaker is running!"}

# Resume tweak endpoint
@app.post("/tweak_resume")
async def tweak_resume(data: ResumeRequest):
    try:
        prompt = f"""
You are a resume optimization assistant. Given a candidate's resume and a job description, modify the resume to align it at least 75% with the job description while keeping all claims truthful.

ONLY RETURN THE MODIFIED RESUME TEXT.

Resume:
{data.resume}

Job Description:
{data.job_description}
"""

        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)

        return {"modified_resume": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
