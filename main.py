from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import google.generativeai as genai
from docx import Document
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow all origins for CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google Generative AI client with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text(file: UploadFile) -> str:
    suffix = file.filename.split(".")[-1].lower()
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
You are an expert resume optimization assistant.

Your task is to revise the candidate's resume so that it aligns up to 85% with the provided job description.

- Emphasize relevant skills, experiences, and achievements that match the job requirements.
- Maintain a professional and readable format suitable for recruiters.
- You may reword, rearrange, or highlight content, you can fabricate any information or add details not present in the original resume.

Here is the candidate's resume:
{resume}

Here is the job description:
{jd}

ONLY return the improved resume text.
"""

@app.get("/list_models")
async def list_models():
    try:
        models = genai.list_models()
        model_names = [model.name for model in models]
        return {"available_models": model_names}
    except Exception as e:
        return {"error": str(e)}

@app.post("/tweak_resume")
async def tweak_resume(
    resume_file: UploadFile = File(...),
    jd_file: UploadFile = File(...)
):
    try:
        resume_text = extract_text(resume_file)
        jd_text = extract_text(jd_file)

        prompt = build_prompt(resume_text, jd_text)

        # Use a valid Gemini model
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)

        return JSONResponse(content={"modified_resume": response.text.strip()})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
