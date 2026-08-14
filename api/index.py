import os
import json
import re
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from groq import Groq

app = FastAPI(title="MoneyballPro Engine", version="2.0.0")

# Habilita CORS para liberar requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# DICIONÁRIO NECESSÁRIO PARA A FUNÇÃO montar_system_prompt_mie2
REGRAS_ESPORTES = {
    "futebol": "Mercados: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor.",
    "nfl": "Mercados: Vencedor (Moneyline), Spread (Handicap), Total de Pontos, Yardas de Passe/Corrida/Recepção, Touchdown Qualquer Momento."
}

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada na Vercel.")
    return genai.Client(api_key=GEMINI_API_KEY)

def get_groq_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada na Vercel.")
    return Groq(api_key=GROQ_API_KEY)

def montar_system_prompt_mie2(sport: str, foco: str = "misto") -> str:
    esporte_key = sport.lower()
    catalogo_esporte = REGRAS_ESPORTES.get(esporte_key, REGRAS_ESPORTES["futebol"])

    return f"""Você é o Moneyball Intelligence Engine (MIE v2.5), analista quantitativo de alta precisão para a modalidade {sport.upper()}.

Você recebe a transcrição OCR de 2 a 5 prints contendo dados táticos, probabilísticos e odds reais capturadas. Sua missão é aplicar o funil quantitativo sobre TODOS os mercados presentes nos prints e devolver as 2 melhores tomadas de decisão gerais no formato JSON padronizado.

[1. MATRIZ OFICIAL DE MERCADOS]
Analise livremente os mercados permitidos dentro da modalidade {sport.upper()}:

{catalogo_esporte}

------------------------------------------------

[2. CLASSIFICAÇÃO DA HIPÓTESE DA PARTIDA]
Classifique a partida em EXCLUSIVAMENTE UMA das 3 hipóteses táticas:
1. TIPO A — PRODUÇÃO (Volume e Fluidez Distribuída): Ambas as partes contribuem.
2. TIPO B — DOMÍNIO (Superioridade e Controle): Um lado domina o resultado.
3. TIPO C — PRODUÇÃO ASSIMÉTRICA (Concentração Unilateral): Performance concentrada em um lado ou atleta.

------------------------------------------------

[3. FILTROS DE SEGURANÇA E REGRA DOS DOIS MELHORES EDGES]
1. MARGEM DE SEGURANÇA (EDGE MÍNIMO - Δ_min):
   - A assimetria (Δ) é: Δ = Prob_Modelo - Prob_Odd.
   - SÓ É ELEGÍVEL PARA A DUPLA QUALQUER SELEÇÃO COM Δ >= 5.0%.
   - Se 5.0% <= Δ < 8.0%: Stake recomendada 1.0u.
   - Se Δ >= 8.0%: Stake recomendada de 1.5u a 2.0u.

2. SELEÇÃO DA DUPLA DE ELITE (LIVRE DE CATEGORIA):
   - Avalie TODOS os mercados extraídos de todos os prints fornecidos.
   - Entrada 1: A maior assimetria validada (Δ >= 3.0%) entre os mercados permitidos.
   - Entrada 2: A segunda maior assimetria validada (Δ >= 2.5%) entre os mercados permitidos.
   - NUNCA invente seleções, linhas, atletas ou odds que não estejam explicitamente presentes nos prints.
   - Se houver apenas 1 mercado elegível com Δ >= 5.0% em todas as fotos, retorne "entrada_2" como null.

3. JANELA DE ODDS E NORMALIZAÇÃO AMERICANA:
   - Cotações entre 1.60 e 2.80 (Exceção MLB/F5 e NFL ML: até 3.00 se Δ >= 10.0%).
   - Se as odds no JSON estiverem em formato americano (+120, -150), CONVERTA para decimal na saída.

4. REGRA DO NOME EXPLÍCITO:
   - Identifique nominalmente o atleta/equipe e a linha exata. Ex: "Las Vegas Aces — Margem", "A'ja Wilson — Over 20.5 Pontos".

------------------------------------------------

[4. REGRAS DE BLOQUEIO E PROTEÇÃO - MONEYBALL 2.0]
1. BLOQUEIO TOTAL DE MONEYLINE (ML):
   - O mercado de Vencedor (Moneyline / 1x2 / ML) está ESTRITAMENTE PROIBIDO. Sob nenhuma circunstância o sistema deve sugerir vitória seca de equipes. Foque apenas em mercados de volume, totais, handicaps ou estatísticas.
2. FILTRO ANTI-ESTRELA (ODDS ESMAGADAS):
   - Proibido sugerir apostas em favoritos com cotações abaixo de @1.50, a menos que venham acompanhadas de uma linha de segurança robusta (como Handicap Asiático ou Escanteios) validada pelo Delta.
3. PROTEÇÃO CONTRA JOGOS TRUNCADOS (GOLS E PROPS DE FUTEBOL):
   - No Futebol, NUNCA sugira "Mais de 2.5 Gols" (Over 2.5) se houver indício de jogo travado ou média baixa; priorize linhas de segurança (ex: Mais de 1.5 ou Escanteios).
   - No Futebol, é ESTRITAMENTE PROIBIDO sugerir props individuais de atletas (chutes, passes, desarmes), devido à alta variância. Props individuais são permitidas apenas em esportes americanos (NBA/NFL/MLB).
4. ISOLAMENTO DE CONTEXTO E VALIDAÇÃO CRUZADA DE ESPORTE:
   - Trate cada print de forma totalmente independente. NÃO misture dados de jogos diferentes.
   - Valide se o esporte real identificado no print bate com o esporte selecionado na interface ({sport.upper()}). Se houver divergência (ex: usuário selecionou Futebol mas enviou Basquete), INTERROMPA o fluxo imediatamente e retorne status de erro por incompatibilidade de esporte, sem tentar adivinhar resultados.

------------------------------------------------

[5. LINGUAGEM E TOM DE VOZ]
- Comunique-se de forma SIMPLES, DIRETA e PRÁTICA.
- Conecte o dado numérico à realidade tática do jogo de forma convincente.

------------------------------------------------

[6. REGRA DE RETORNO JSON STRICT]
Sua resposta DEVE SER ESTRITAMENTE um JSON válido na estrutura exata abaixo (sem marcações markdown antes ou depois, apenas o JSON bruto). Se houver erro de validação de esporte (Regra 4), preencha o `status_geral` com "erro_divergencia_esporte" e detalhe no `perfil_geral`:

{{
  "perfil_geral": "Síntese quantitativa da partida e leitura tática...",
  "status_geral": "processado_com_sucesso",
  "hipotese_partida": "TIPO A | TIPO B | TIPO C",
  "stake_medio_partida": "1.0u",
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Time A vs Time B",
    "date": "Hoje"
  }},
  "expected_projections": {{
    "macro_projected": "Projeção relevante com delta",
    "micro_projected": "Projeção relevante com delta"
  }},
  "dupla_de_elite": {{
    "entrada_1": {{
      "categoria": "MACRO ou MICRO",
      "mercado": "Nome do Mercado",
      "selecao": "Seleção Explícita com Linha",
      "odd": "1.85",
      "delta_edge": "7.6%",
      "msc_score": 90,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa da entrada..."
    }},
    "entrada_2": {{
      "categoria": "MACRO ou MICRO",
      "mercado": "Nome do Mercado",
      "selecao": "Seleção Explícita com Linha",
      "odd": "1.90",
      "delta_edge": "6.2%",
      "msc_score": 85,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa da entrada..."
    }}
  }},
  "key_asymmetries": [
    {{
      "clash": "Confronto ou Atleta Analisado",
      "statistical_evidence": "Evidência estatística extraída com delta",
      "betting_angle": "Direcionamento da aposta"
    }}
  ]
}}
"""

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
    foco: str = Form("misto"),
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
            model="gemini-3.5-flash-lite",
            contents=contents
        )
        texto_extraido_ocr = res_ocr.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no OCR (Gemini): {str(e)}")

    # 2. ANÁLISE QUANTITATIVA VIA GROQ (GPT-OSS 120B)
    groq_client = get_groq_client()
    system_instruction_mie2 = montar_system_prompt_mie2(sport=sport, foco=foco)

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

        # Limpeza defensiva de blocos Markdown (evita crash no json.loads)
        if "```" in content_str:
            content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
            content_str = re.sub(r"\s*```$", "", content_str)
            content_str = content_str.strip()

        return json.loads(content_str)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Erro de formatação JSON retornado pela IA: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento (Groq): {str(e)}")
