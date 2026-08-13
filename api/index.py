import os
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google import genai
from google.genai import types
from groq import Groq

app = FastAPI(title="MoneyballPro Engine", version="2.0.0")

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
        "engine": "MoneyballPro Engine v2.0 (Full Matrix & Safety Locks)",
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

    # 1. OCR VIA GEMINI 3.5 FLASH
    gemini_client = get_gemini_client()
    contents = []

    for file in files:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents.append(part)

    prompt_ocr = f"Extraia em PORTUGUÊS todo o texto, nomes de times, jogadores, confrontos, odds/cotações e linhas destas imagens para a modalidade: {sport}. Retorne apenas a transcrição direta e limpa."
    contents.append(prompt_ocr)

    try:
        res_ocr = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=contents
        )
        texto_extraido_ocr = res_ocr.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no OCR (Gemini): {str(e)}")

    # 2. ANÁLISE QUANTITATIVA COM MODELO DE RIGOR E TRAVAS DE SEGURANÇA VIA GROQ
    groq_client = get_groq_client()

    system_instruction_mie2 = f"""
Você é o Moneyball Intelligence Engine (MIE - Versão 2.0), analista quantitativo de elite especialista em apostas esportivas.
Sua missão é analisar o texto OCR dos bilhetes/partidas para a modalidade: {sport.upper()}.

MATRIZ OFICIAL DE MERCADOS POR ESPORTE:
- FUTEBOL:
  * MACRO (Entrada 1): Gols (Over/Under), BTTS (Ambas Marcam), Chance Dupla, Handicap (Asiático/Europeu - acionado se Chance Dupla tiver Odd < 1.60).
  * MICRO (Entrada 2): Escanteios (Jogo ou Time), Chutes a Gol (Jogo ou Time), Cartões (Jogo ou Time).
- BEISEBOL (MLB):
  * MACRO (Entrada 1): Moneyline (ML), Run Line (Spread), Total de Runs.
  * MICRO (Entrada 2): Strikeouts (Pitcher), Outs / Eliminações, Hits Permitidos (Pitcher), Hits (Rebatedor).
- BASQUETE (NBA):
  * MACRO (Entrada 1): Vencedor (Moneyline), Spread (Handicap), Total de Pontos, Team Total.
  * MICRO (Entrada 2): Pontos do Jogador, Rebotes do Jogador, Assistências do Jogador.
- FUTEBOL AMERICANO (NFL):
  * Selecionar as 2 maiores assimetrias (Delta >= 5.0%) em jogos/mercados distintos cobrindo ML, Spread, Total de Pontos ou Props Individuais (Jardas, Anytime TD).

REGRAS ESTRITAS E TRAVAS DE SEGURANÇA (OBRIGATÓRIO):
1. IDIOMA E PADRÃO: 100% em PORTUGUÊS DO BRASIL (PT-BR). Nomes de times, jogadores e mercados EXPLICITAMENTE indicados na seleção (Ex: "Mirassol ou Empate (1X)", "Michael Estrada - Over 1.5 Chutes").
2. JANELA DE ODDS OPERÁVEL: 1.60 <= Odd <= 2.80.
   - Rejeite cotações < 1.60 ou > 2.80 (Exceção MLB/NFL ML: Odd até 3.00 se Delta >= 10%).
3. EDGE MÍNIMO (DELTA >= 5.0%):
   - Calcule a Assimetria / Delta = Probabilidade_Modelo - Probabilidade_Odd.
   - Se Delta < 5.0%, a aposta DEVE SER REJEITADA.
4. CALCULE O MSC SCORE (MONEYBALL SCORE): Integer de 0 a 100 baseado no Delta e EV+.
5. CONTEXTO TÁTICO E COPEIRO (LIVRE DE HEDGE):
   - Jogos amarrados/mata-mata (ex: Libertadores, playoffs): Ajuste margem de segurança para Under em mercados inflacionados.
   - Fator Desfalques/Blowout: Penalize em 20% o Delta se houver desfalque crítico ou falta de motivação.

Retorne EXCLUSIVAMENTE um objeto JSON válido (sem textos antes ou depois, sem markdown extra):
{{
  "perfil_geral": "Síntese quantitativa da partida e contexto tático...",
  "status_geral": "processado_com_sucesso",
  "stake_medio_partida": "1.0u",
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Time A vs Time B",
    "date": "Hoje"
  }},
  "expected_projections": {{
    "macro_projected": "Projeção Macro",
    "micro_projected": "Projeção Micro"
  }},
  "dupla_de_elite": {{
    "entrada_1_macro": {{
      "mercado": "Nome do Mercado Macro",
      "selecao": "Seleção Explícita com Nome",
      "odd": "1.85",
      "delta_edge": "6.8%",
      "msc_score": 88,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa quantitativa detalhando o Delta e valor esperado."
    }},
    "entrada_2_micro": {{
      "mercado": "Nome do Mercado Micro",
      "selecao": "Seleção Explícita com Nome",
      "odd": "1.95",
      "delta_edge": "8.2%",
      "msc_score": 84,
      "stake_recomendada": "1.5u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa quantitativa detalhando o Delta e volume."
    }}
  }},
  "key_asymmetries": [
    {{
      "clash": "Ponto Tático Analisado",
      "statistical_evidence": "Evidência estatística em PT-BR com margem aplicada",
      "betting_angle": "Direcionamento final"
    }}
  ]
}}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_instruction_mie2},
                {"role": "user", "content": f"TEXTO EXTRAÍDO DOS PRINTS:\n{texto_extraido_ocr}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        return json.loads(completion.choices[0].message.content.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento quântico (Groq): {str(e)}")
