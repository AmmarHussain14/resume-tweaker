from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import google.generativeai as genai
from docx import Document
import pdfplumber

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
You are a resume optimization expert.

Here is the candidate's resume:
{resume}

Here is a job description:
{jd}

Modify the resume to align up to 75% with the job description. Highlight relevant skills and experiences. Keep formatting professional. Do not fabricate information.
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

        # Change this model to one you get from /list_models, example: "models/chat-bison-001"
        model_name = "models/chat-bison-001"

        response = genai.generate_text(
            model=model_name,
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=2000,
        )

        modified_resume = response.text.strip()
        return JSONResponse(content={"modified_resume": modified_resume})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
