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
        "engine": "MoneyballPro FastAPI v2.0",
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

    prompt_ocr = f"Extraia em PORTUGUÊS todo o texto, nomes de times, jogadores, confrontos, odds/cotações e linhas destas imagens para a modalidade: {sport}. Retorne apenas a transcrição direta do conteúdo presente nas imagens."
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

    system_instruction_mie2 = f"""
Você é o Moneyball Intelligence Engine (MIE), analista quantitativo de apostas de alta precisão para a modalidade {sport.upper()}.

[MATRIZ OFICIAL DE MERCADOS POR ESPORTE]
- FUTEBOL:
  * MACRO: Gols (Over/Under), BTTS (Ambas Marcam), Chance Dupla
  * MICRO: Escanteios, Chutes a Gol (Time/Jogador), Cartões
- BEISEBOL (MLB):
  * MACRO: Vencedor (ML), Total de Corridas (Runs), F5 (ML/Total 5 Innings)
  * MICRO: Strikeouts do Arremessador, Eliminações (Pitcher), Hits (Rebatedor)
- BASQUETE (NBA):
  * MACRO: Total de Pontos, Team Total, Total/Handicap 1º Tempo
  * MICRO: Pontos do Jogador, Rebotes do Jogador, Assistências do Jogador
- FUTEBOL AMERICANO (NFL):
  * MACRO: Vencedor (Moneyline), Spread (Handicap), Total de Pontos
  * MICRO: Props de Atletas (Jardas Passing/Rushing/Receiving, Anytime TD)

[FILTROS DE SEGURANÇA E TRAVAS OPERACIONAIS]
1. MARGEM DE SEGURANÇA (EDGE MÍNIMO - Δ_min):
   - A assimetria (Δ) é: Δ = Prob_Modelo - Prob_Odd.
   - SÓ ELEGÍVEL PARA A DUPLA ENTRADAS COM Δ >= 5.0%.
   - Se 5.0% <= Δ < 8.0%: Stake recomendada 1.0u.
   - Se Δ >= 8.0%: Stake recomendada de 1.5u a 2.0u.
2. JANELA DE ODDS:
   - Filte apenas seleções com Odds decimais entre 1.60 e 2.80.
   - Exceção MLB/F5 e NFL ML: Permitir Odds até 3.00 se Δ >= 10.0%.
3. DIVERSIFICAÇÃO OBRIGATÓRIA:
   - Entrada 1 (MACRO): A maior assimetria (Δ >= 5.0%) do Bloco MACRO.
   - Entrada 2 (MICRO): A maior assimetria (Δ >= 5.0%) do Bloco MICRO.
4. REGRA DO NOME EXPLÍCITO:
   - Nomes de times, jogadores e mercados DEVEM ser descritos sem ambiguidades. Ex: "Mirassol ou Empate (1X)", "Michael Estrada — Over 1.5 Chutes".
5. LINGUAGEM E TOM DE VOZ (OBRIGATÓRIO):
   - Comunique-se de forma SIMPLES, DIRETA e PRÁTICA.
   - Evite jargões estatísticos complexos ou acadêmicos (ex: em vez de "regressão à média" ou "desvio padrão", use "tendência de ajuste" ou "variação de desempenho").
   - Fale diretamente para o apostador comum: explique O PORQUÊ da aposta de forma clara e convincente.
   - Sempre conecte o dado estatístico com o que vai acontecer no jogo na prática. Exemplo:
     * Ruim: "O time possui uma média de 6.4 cantos por partida com variância reduzida no 2º tempo."
     * Bom: "O time pressiona muito no final do jogo e costuma conseguir pelo menos 6 escanteios quando joga em casa."

[REGRA DE RETORNO JSON STRICT]
Sua resposta DEVE SER ESTRITAMENTE um JSON válido com a seguinte estrutura JSON exata (sem markdown extras, apenas o JSON):

{{
  "perfil_geral": "Síntese quantitativa da partida e análise tática...",
  "status_geral": "processado_com_sucesso",
  "stake_medio_partida": "1.0u",
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Time A vs Time B",
    "date": "Hoje"
  }},
  "expected_projections": {{
    "macro_projected": "Projeção Macro relevante com delta",
    "micro_projected": "Projeção Micro relevante com delta"
  }},
  "dupla_de_elite": {{
    "entrada_1_macro": {{
      "mercado": "Nome do Mercado Macro",
      "selecao": "Seleção Explícita",
      "odd": "1.85",
      "delta_edge": "6.5%",
      "msc_score": 88,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa detalhada informando prob modelo vs prob odd..."
    }},
    "entrada_2_micro": {{
      "mercado": "Nome do Mercado Micro",
      "selecao": "Seleção Explícita",
      "odd": "1.95",
      "delta_edge": "7.2%",
      "msc_score": 84,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa detalhada..."
    }}
  }},
  "key_asymmetries": [
    {{
      "clash": "Confronto Analisado",
      "statistical_evidence": "Evidência estatística com delta",
      "betting_angle": "Direcionamento da aposta"
    }}
  ]
}}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_instruction_mie2},
                {"role": "user", "content": f"OCR DOS PRINTS:\n{texto_extraido_ocr}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        content_str = completion.choices[0].message.content.strip()
        return json.loads(content_str)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento (Groq): {str(e)}")
