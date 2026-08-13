import os
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google import genai
from google.genai import types
from duckduckgo_search import DDGS

app = FastAPI(title="MoneyballPro Engine", version="1.0.0")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada no servidor.")
    return genai.Client(api_key=GEMINI_API_KEY)

def buscar_noticias_recentes(time: str, esporte: str) -> List[str]:
    """Busca notícias recentes de desfalques/lesões no DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            query = f"noticias lesoes desfalques {time} {esporte}"
            results = list(ddgs.text(query, max_results=3))
            return [r.get("body", "") for r in results if "body" in r]
    except Exception:
        return []

@app.get("/api/health")
def health_check():
    return {"status": "ok", "engine": "MoneyballPro FastAPI", "gemini_key_set": bool(GEMINI_API_KEY)}

@app.post("/api/v1/analyze")
async def analyze_tickets(
    sport: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    client = get_gemini_client()
    contents = []

    for file in files:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents.append(part)

    prompt_ocr = f"""
    Você é um extrator de dados de apostas esportivas da Engine MoneyballPro.
    Analise a(s) imagem(ns) fornecida(s) para o esporte: '{sport}'.

    Extraia estritamente em formato JSON com as seguintes chaves:
    {{
        "esporte": "{sport}",
        "jogos": [
            {{
                "time_casa": "Nome do Time Casa",
                "time_fora": "Nome do Time Fora",
                "mercado": "Mercado Selecionado (ex: Handicap, Over/Under, Ambas Marcam, Moneyline)",
                "selecao": "Sua aposta específica",
                "odd": "Valor da odd"
            }}
        ]
    }}
    Responda APENAS com o JSON válido.
    """
    contents.append(prompt_ocr)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        
        texto_resposta = response.text.strip()
        if texto_resposta.startswith("```"):
            texto_resposta = texto_resposta.split("```")[1]
            if texto_resposta.startswith("json"):
                texto_resposta = texto_resposta[4:]
        
        dados_extraidos = json.loads(texto_resposta)

        jogos_enriquecidos = []
        for jogo in dados_extraidos.get("jogos", []):
            time_casa = jogo.get("time_casa", "")
            time_fora = jogo.get("time_fora", "")

            noticias_casa = buscar_noticias_recentes(time_casa, sport) if time_casa else []
            noticias_fora = buscar_noticias_recentes(time_fora, sport) if time_fora else []

            jogo["contexto_web"] = {
                "noticias_time_casa": noticias_casa,
                "noticias_time_fora": noticias_fora
            }
            jogos_enriquecidos.append(jogo)

        return {
            "status": "sucesso",
            "esporte": sport,
            "total_jogos": len(jogos_enriquecidos),
            "jogos": jogos_enriquecidos
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao formatar resposta do OCR em JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento: {str(e)}")
