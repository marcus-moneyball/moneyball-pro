import os
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google import genai
from google.genai import types
from duckduckgo_search import DDGS
from groq import Groq

app = FastAPI(title="MoneyballPro Engine", version="1.0.0")

# Chaves das variáveis de ambiente na Vercel
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

def buscar_noticias_recentes(termo: str, esporte: str) -> List[str]:
    """Busca notícias e desfalques recentes via DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            query = f"noticias lesoes desfalques {termo} {esporte}"
            results = list(ddgs.text(query, max_results=3))
            return [r.get("body", "") for r in results if "body" in r]
    except Exception:
        return []

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

    # ---------------------------------------------------------
    # PASSO 1: OCR dos Prints com Gemini
    # ---------------------------------------------------------
    gemini_client = get_gemini_client()
    contents = []

    for file in files:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents.append(part)

    prompt_ocr = f"""
    Extraia todo o texto, times, confrontos, cotações/odds e seleções contidas nestas imagens de apostas esportivas para o esporte '{sport}'.
    Retorne uma transcrição limpa e direta de tudo o que encontrar na imagem.
    """
    contents.append(prompt_ocr)

    try:
        res_ocr = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        texto_extraido_ocr = res_ocr.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na extração OCR: {str(e)}")

    # ---------------------------------------------------------
    # PASSO 2: Enriquecimento via Busca Web (DuckDuckGo)
    # ---------------------------------------------------------
    contexto_web = buscar_noticias_recentes(texto_extraido_ocr[:100], sport)
    texto_contexto = "\n".join(contexto_web) if contexto_web else "Sem notícias adicionais de desfalques encontradas."

    # ---------------------------------------------------------
    # PASSO 3: MIE 1 via Groq (Llama 3.3 70B Versatile)
    # ---------------------------------------------------------
    groq_client = get_groq_client()

    system_instruction_mie1 = f"""
Você é o Moneyball Intelligence Engine (MIE), um Investigador e Analista Quantitativo Esportivo de elite especializado em: {sport.upper()}.

OBJETIVO:
Investigar o evento esportivo solicitado, extrair dados estatísticos brutos, mapear assimetrias matemáticas e gerar PROJEÇÕES NUMÉRICAS DIRETAS (Expected Values / λ) para alimentar a Matriz de Carteira do Engine 2 (Âncora Macro, Âncora Tática e Coringa Flexível).

------------------------------------------------

DIRETRIZES DE EXECUÇÃO (MULTIMODALIDADE & ANTI-ALUCINAÇÃO):
1. PROCESSAMENTO DE PRINTS E OCR:
   - Ignore ruídos de interface (menus, propagandas, botões).
   - Extraia com fidelidade nomes, estatísticas e odds.
   - Se um print estiver cortado ou uma métrica crítica incompleta, REGISTRE O CORTE no campo 'limitations'.

2. FOCO NUMÉRICO ABSOLUTO & PROJEÇÕES OBRIGATÓRIAS:
   - NUNCA use adjetivos ("ataque bom", "jogo fraco"). Use dados exatos (ex: "xG 1.85", "Pace 102.4").
   - Você DEVE calcular e incluir projeções numéricas diretas (Valores Esperados / λ) para:
     a) ÂNCORA MACRO: Projeção de Total de Gols/Corridas/Pontos da Partida.
     b) ÂNCORA TÁTICA: Projeção de Total do Time A/B, % de BTTS ou Projeção de Ks do Arremessador.
     c) CORINGA FLEXÍVEL: Projeções de Posse/Dominância (para Dupla Chance/Handicap) OU Projeções Individuais (Chutes no Alvo, Hits, PTS/REB/AST).

3. CONTEXTO FACTUAL & CETICISMO ESTATÍSTICO:
   - Mapeie desfalques confirmados, clima e fadiga de calendário (back-to-back/repouso).
   - Identifique "red flags" (times superestimados por vitórias com baixo xG ou alta sorte estatística).

------------------------------------------------

REGRAS DE SAÍDA E FORMATAÇÃO (CRÍTICO - HARD CONSTRAINT):
- Sua resposta DEVE SER EXCLUSIVAMENTE um objeto JSON válido.
- NUNCA inclua texto explicativo, saudações, ou blocos de código markdown (como ```json) antes ou depois do objeto.
- O primeiro caractere da sua resposta deve ser {{ e o último deve ser }}.

------------------------------------------------

ESTRUTURA OBRIGATÓRIA DE SAÍDA (JSON):
{{
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Time A vs Time B",
    "date": "Data do evento"
  }},
  "contextual_factors": [
    {{
      "factor_type": "injury | weather | schedule_fatigue | motivation",
      "description": "Descrição factual baseada em dados",
      "impact_level": "high | medium | low",
      "affected_team": "Nome do time ou 'ambos'"
    }}
  ],
  "expected_projections": {{
    "macro_total_projected": 2.85,
    "team_a_projected": 1.75,
    "team_b_projected": 1.10,
    "btts_probability_pct": 65,
    "player_props_projected": [
      {{
        "player_name": "Nome do Atleta",
        "prop_type": "Strikeouts / Chutes no Alvo / Hits / PTS",
        "projected_value": 6.2,
        "line_offered": 5.5
      }}
    ]
  }},
  "quantitative_metrics": [
    {{
      "entity": "Nome do Time ou Jogador",
      "metric_name": "xG, Pace, xERA, K%, Usage Rate, Shots/90",
      "value": "Valor numérico exato",
      "sample_size": "Últimos 5 jogos / Temporada"
    }}
  ],
  "key_asymmetries": [
    {{
      "clash": "Onde a força de um time encontra a fraqueza do outro",
      "statistical_evidence": "Evidência numérica da assimetria",
      "betting_angle": "Indicação de direcionamento para o Engine 2"
    }}
  ],
  "limitations": [
    "Métricas ausentes, incertezas de escalação ou cortes no print."
  ]
}}
"""

    prompt_usuario = f"""
    DADOS EXTRAÍDOS DOS PRINTS (OCR):
    {texto_extraido_ocr}

    CONTEXTO DE NOTÍCIAS E DESFALQUES RECENTES DA WEB:
    {texto_contexto}
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction_mie1},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        resposta_json_str = completion.choices[0].message.content.strip()
        dossie_json = json.loads(resposta_json_str)
        
        return dossie_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento do MIE1 via Groq: {str(e)}")
