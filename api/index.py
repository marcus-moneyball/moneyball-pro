import os
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google import genai
from google.genai import types
from groq import Groq

app = FastAPI(title="MoneyballPro Engine", version="1.0.0")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada na Vercel.")
    return genai.Client(api_key=GEMINI_API_KEY)

def get_groq_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada na Vercel.")
    return Groq(api_key=GROQ_API_KEY)

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "engine": "MoneyballPro FastAPI",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "groq_key_set": bool(GROQ_API_KEY)
    }

@app.post("/api/v1/analyze")
async def analyze_tickets(
    sport: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    # 1. OCR COM GEMINI 3.5 FLASH
    gemini_client = get_gemini_client()
    contents = []

    for file in files:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents.append(part)

    prompt_ocr = f"Extraia o texto, times, confrontos, odds e seleções destas imagens para {sport}. Retorne apenas a transcrição direta."
    contents.append(prompt_ocr)

    try:
        res_ocr = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=contents
        )
        texto_extraido_ocr = res_ocr.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no OCR (Gemini): {str(e)}")

    # 2. ANÁLISE QUANTITATIVA VIA GROQ (GPT-OSS 120B)
    groq_client = get_groq_client()

    system_instruction_mie1 = f"""
Você é o Moneyball Intelligence Engine (MIE), analista quantitativo de {sport.upper()}.
Gere um JSON estrito conforme o schema abaixo com base nos dados fornecidos:

{{
  "match_info": {{ "sport": "{sport.upper()}", "teams": "Time A vs Time B", "date": "Data" }},
  "expected_projections": {{
    "macro_total_projected": 2.85,
    "team_a_projected": 1.75,
    "team_b_projected": 1.10
  }},
  "key_asymmetries": [
    {{ "clash": "Confronto", "statistical_evidence": "Evidência", "betting_angle": "Direcionamento" }}
  ]
}}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_instruction_mie1},
                {"role": "user", "content": f"OCR DOS PRINTS:\n{texto_extraido_ocr}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        return json.loads(completion.choices[0].message.content.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na Groq: {str(e)}")
