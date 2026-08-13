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

    prompt_ocr = f"Extraia em PORTUGUÊS todo o texto, nomes de times, jogadores, confrontos, odds/cotações e linhas destas imagens para {sport}. Retorne apenas a transcrição direta."
    contents.append(prompt_ocr)

    try:
        res_ocr = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=contents
        )
        texto_extraido_ocr = res_ocr.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no OCR (Gemini): {str(e)}")

    # 2. ANÁLISE QUANTITATIVA RIGOROSA VIA GROQ (GPT-OSS 120B)
    groq_client = get_groq_client()

    system_instruction_mie1 = f"""
Você é o Moneyball Intelligence Engine (MIE), analista quantitativo de apostas de elite para {sport.upper()}.

REGRAS ESTRITAS DE ANÁLISE:
1. RESPONDA TUDO 100% EM PORTUGUÊS DO BRASIL (PT-BR). Nomes de mercados padronizados nas casas brasileiras.
2. Entrada 1 (Macro) e Entrada 2 (Micro/Time Isolado) DEVEM pertencer a categorias distintas dentro do mesmo confronto.
3. É TERMINANTEMENTE PROIBIDO fazer "Hedge" (indicar lados opostos do mesmo mercado na partida).
4. Em Props de Jogador ou Ações de Time, O ALVO DEVE CONSTAR EXPLICITAMENTE (Ex: "Atleta — Over 4.5" ou "Time — Over 5.5 Escanteios").
5. As odds DEVEM ESTAR estritamente em formato DECIMAL (ex: "1.85", "2.10").

[REGRA MATEMÁTICA UNIVERSAL DE MARGEM DE SEGURANÇA - ANTI-DRIBLE]
- Mercados de escala pequena (Gols, Corridas, Cartões, Escanteios, Chutes):
  * Sugerir OVER X: A projeção quantitativa deve ser >= (X + 0.20).
  * Sugerir UNDER X: A projeção quantitativa deve ser <= (X - 0.20).
- Mercados de escala alta (Pontos NBA, Jardas NFL, Rebotes/Assistências):
  * Sugerir OVER X: Projeção deve ser pelo menos 5% maior que a linha X.
  * Sugerir UNDER X: Projeção deve ser pelo menos 5% menor que a linha X.

[ALINHAMENTO NARRATIVO]
- Em partidas com alta assimetria ofensiva, dê preferência a mercados proativos (Over). É proibido utilizar Under passivo apenas para cumprir margem se houver opção de Over com maior EV+.

Retorne EXCLUSIVAMENTE um objeto JSON válido (sem blocos markdown adicionais, apenas o JSON puro) seguindo rigorosamente este schema:
{{
  "perfil_geral": "Síntese quantitativa da partida destacando as 2 maiores assimetrias puras (Δ EV+) encontradas no confronto...",
  "status_geral": "processado_com_sucesso",
  "stake_medio_partida": "1.0u",
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Time Casa vs Time Fora",
    "date": "Hoje"
  }},
  "expected_projections": {{
    "macro_total_projected": 2.5,
    "team_a_projected": 1.5,
    "team_b_projected": 1.0
  }},
  "dupla_de_elite": {{
    "entrada_1_macro": {{
      "mercado": "Nome do Mercado em PT-BR",
      "selecao": "Seleção Exata",
      "odd": "1.85",
      "msc_score": 89,
      "stake_recomendada": "1.5u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa embasada na assimetria quantitativa do jogo todo."
    }},
    "entrada_2_micro": {{
      "mercado": "Nome do Mercado em PT-BR",
      "selecao": "Seleção Exata (incluindo nome do jogador ou do time isolado)",
      "odd": "1.90",
      "msc_score": 84,
      "stake_recomendada": "1.0u",
      "confiabilidade": "MÉDIA-ALTA",
      "motivo": "Justificativa embasada na assimetria quantitativa individual ou do time isolado."
    }}
  }},
  "key_asymmetries": [
    {{
      "clash": "Mercado / Confronto Analisado",
      "statistical_evidence": "Explicação quantitativa direta",
      "betting_angle": "Conclusão prática para a aposta"
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
