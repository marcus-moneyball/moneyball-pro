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

    prompt_ocr = f"Extraia em português o texto, nomes dos times/jogadores, confrontos, odds e seleções destas imagens para {sport}. Retorne apenas a transcrição direta e em português."
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
Você é o Moneyball Intelligence Engine (MIE), analista quantitativo de apostas para {sport.upper()}.
MUITO IMPORTANTE: RESPONDA TUDO EM PORTUGUÊS DO BRASIL (PT-BR).
Use os nomes tradicionais de mercados das casas de apostas brasileiras (ex: "Acima de 2.5 Gols", "Dupla Chance X2", "Handicap Asiático +0.5", "Resultado Final", "Ambas Marcam: Sim").

Gere um JSON estrito em português conforme o schema abaixo:

{{
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Nome do Time Casa vs Nome do Time Fora",
    "date": "Data ou Horário do Jogo"
  }},
  "expected_projections": {{
    "macro_total_projected": 2.85,
    "team_a_projected": 1.75,
    "team_b_projected": 1.10
  }},
  "recommended_bets": {{
    "ancora_macro": "Mercado Seguro/Conservador em Português (Ex: Acima de 1.5 Gols)",
    "coringa_tatico": "Mercado de Maior Valor/Assimetria em Português (Ex: Dupla Chance Time Casa ou Empate)"
  }},
  "key_asymmetries": [
    {{
      "clash": "Nome do Duelo/Confronto em Português",
      "statistical_evidence": "Explicação e estatística em Português",
      "betting_angle": "Direcionamento da Aposta em Português"
    }}
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
