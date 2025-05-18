from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import pdfplumber
from docx import Document
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # To load COHERE_API_KEY or GEMINI_API_KEY from .env

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def extract_text(file: UploadFile) -> str:
    suffix = file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if suffix == "pdf":
            with pdfplumber.open(tmp_path) as pdf:
                return "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )
        elif suffix == "docx":
            doc = Document(tmp_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif suffix == "txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "Unsupported file type"
    finally:
        os.unlink(tmp_path)


def build_prompt(resume: str, jd: str) -> str:
    return f"""
You are a resume optimization expert.

Here is the candidate's resume:
{resume}

Here is a job description:
{jd}

Modify the resume to align up to 75% with the job description. Highlight relevant skills and experiences.
Keep formatting professional. Do not fabricate information.
"""


@app.post("/tweak_resume")
async def tweak_resume(
    resume_file: UploadFile = File(...), jd_file: UploadFile = File(...)
):
    try:
        resume_text = extract_text(resume_file)
        jd_text = extract_text(jd_file)
        prompt = build_prompt(resume_text, jd_text)

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        return JSONResponse(content={"modified_resume": response.text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
