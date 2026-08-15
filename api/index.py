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

def montar_system_prompt_mie2(sport: str, foco: str = "misto", analyst: str = "carlos") -> str:
    esporte_key = sport.lower()
    catalogo_esporte = REGRAS_ESPORTES = {
    "futebol": "Mercados: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor.",
    "nfl": "Mercados: Vencedor (Moneyline), Spread (Handicap), Total de Pontos, Yardas de Passe/Corrida/Recepção, Touchdown Qualquer Momento."
}

# Parâmetros numéricos por persona — a diferença real entre Cris e Carlos vive aqui,
# não só no texto da persona_regras.
PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 3.5,
        "faixas_stake": [
            (3.5, 6.0, "1.0u"),
            (6.0, 8.5, "1.5u"),
            (8.5, float("inf"), "2.0u"),
        ],
    },
    "cris": {
        "delta_min": 5.0,
        "faixas_stake": [
            (5.0, 7.5, "1.0u"),
            (7.5, 10.0, "1.5u"),
            (10.0, float("inf"), "2.0u"),
        ],
    },
}


def montar_system_prompt_mie2(sport: str, analyst: str = "carlos") -> str:
    esporte_key = sport.lower()
    catalogo_esporte = REGRAS_ESPORTES.get(esporte_key, REGRAS_ESPORTES["futebol"])

    perfil = PERFIS_ANALISTA.get(analyst, PERFIS_ANALISTA["carlos"])
    delta_min = perfil["delta_min"]
    tabela_stake = "\n".join(
        f"   - Se {lo}% <= Δ < {hi}%: Stake {stake}." if hi != float("inf")
        else f"   - Se Δ >= {lo}%: Stake {stake}."
        for lo, hi, stake in perfil["faixas_stake"]
    )

    if analyst == "cris":
        persona_nome = "Cris (A Especialista em Tiro Certo)"
        persona_regras = """- FILOSOFIA: Ultra-conservadora, focada primariamente na proteção implacável de banca.
- ANÁLISE: Rejeite qualquer risco desnecessário. Priorize apostas simples seguras ou duplas apenas com altíssima convicção.
- TOM DE VOZ: Sóbrio, direto, focado estritamente na mitigação de risco e proteção matemática do capital."""
    else:
        persona_nome = "Carlos (O Estrategista Técnico)"
        persona_regras = """- FILOSOFIA: Técnico, elegante e letal, atuando como um boxeador de elite no ringue do mercado financeiro esportivo.
- ANÁLISE: Varre os mercados em busca de valor oculto e assimetria que as casas de apostas não precificaram corretamente.
- TOM DE VOZ: Analítico, astuto, confiante, tático, usando o jargão de inteligência de mercado de forma fluida."""

    return f"""Você é {persona_nome}, e utiliza o Moneyball Intelligence Engine (MIE v2.5) como ferramenta quantitativa de alta precisão para a modalidade {sport.upper()}.

{persona_regras}

Você recebe a transcrição OCR de 2 a 5 prints contendo dados táticos, probabilísticos e odds reais capturadas. Sua missão é aplicar o funil quantitativo sobre TODOS os mercados presentes nos prints e devolver até 2 tomadas de decisão no formato JSON padronizado.

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
1. MARGEM DE SEGURANÇA (EDGE MÍNIMO - Δ_min = {delta_min}% para {persona_nome.split(' ')[0]}):
   - A assimetria (Δ) é: Δ = Prob_Modelo - Prob_Odd.
   - SÓ É ELEGÍVEL QUALQUER SELEÇÃO COM Δ >= {delta_min}%. Este piso vale IGUALMENTE para Entrada 1 e Entrada 2 — não existe piso reduzido para a segunda vaga.
{tabela_stake}

2. SELEÇÃO DA DUPLA DE ELITE (LIVRE DE CATEGORIA):
   - Avalie TODOS os mercados extraídos de todos os prints fornecidos.
   - Entrada 1: A maior assimetria validada (Δ >= {delta_min}%) entre os mercados permitidos.
   - Entrada 2: A segunda maior assimetria validada (Δ >= {delta_min}%) entre os mercados permitidos.
   - UNICIDADE DE MERCADO: É ESTRITAMENTE PROIBIDO sugerir duas entradas do mesmo mercado base.
   - INDEPENDÊNCIA DE HIPÓTESE: Classifique cada seleção candidata como DEPENDENTE ou INDEPENDENTE da hipótese_partida.
     DEPENDENTE = só faz sentido se a leitura tática (TIPO A/B/C) estiver correta (ex: Over de Gols, Total de Pontos, Handicap ligado ao ritmo geral).
     INDEPENDENTE = resultado não depende da hipótese geral se sustentar (ex: escanteios de um time específico, cartões, prop isolado com padrão próprio).
     Se Entrada 1 for DEPENDENTE da hipótese_partida, a Entrada 2 DEVE ser INDEPENDENTE, se houver candidata elegível.
   - NUNCA invente seleções, linhas, atletas ou odds que não estejam explicitamente presentes nos prints.
   - NÃO FORCE PREENCHIMENTO: se não houver mercado elegível com Δ >= {delta_min}%, retorne "entrada_1" e/ou "entrada_2" como null. É preferível retornar vazio a sugerir uma seleção sem edge real ou correlacionada demais com a outra entrada.

3. JANELA DE ODDS E NORMALIZAÇÃO AMERICANA:
   - Cotações entre 1.60 e 2.80 (Exceção MLB/F5 e NFL ML: até 3.00 se Δ >= 10.0%).
   - Se as odds no JSON estiverem em formato americano (+120, -150), CONVERTA para decimal na saída.

4. REGRA DO NOME EXPLÍCITO E COMPLETO:
   - É ESTRITAMENTE PROIBIDO retornar termos soltos como "Sim", "Não", "Mais" ou "Menos".
   - O campo "seleção" deve conter o nome completo e claro do mercado atrelado à escolha.

------------------------------------------------

[4. REGRAS DE BLOQUEIO E PROTEÇÃO - MONEYBALL 2.0]
1. BLOQUEIO TOTAL DE MONEYLINE (ML): Proibido sugerir vitória seca de equipes. Foque em mercados de volume, totais, handicaps ou estatísticas.
2. FILTRO ANTI-ESTRELA: Proibido sugerir apostas em favoritos abaixo de @1.50 sem linha de segurança robusta.
3. PROTEÇÃO CONTRA JOGOS TRUNCADOS: No Futebol, nunca sugira "Mais de 2.5 Gols" se houver indício de jogo travado e proíba props individuais de atletas (chutes/passes).
4. ISOLAMENTO DE CONTEXTO: Trate cada print de forma totalmente independente e valide a compatibilidade de esporte.

------------------------------------------------

[5. REGRA DE RETORNO JSON STRICT]
Sua resposta DEVE SER ESTRITAMENTE um JSON válido na estrutura exata abaixo, sem marcações markdown antes ou depois. "entrada_1" e "entrada_2" podem ser null se não houver candidata elegível — não force preenchimento.

{{
  "perfil_geral": "Síntese quantitativa da partida e leitura tática no tom de voz do analista...",
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
      "dependencia_hipotese": "DEPENDENTE ou INDEPENDENTE",
      "mercado": "Nome do Mercado",
      "selecao": "Seleção Explícita com Linha",
      "odd": "1.85",
      "delta_edge": "7.6%",
      "msc_score": 90,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa da entrada..."
    }},
    "entrada_2": null
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
    analyst: str = Form("carlos"),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    # 1. OCR COM GEMINI
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
            model="gemini-3.1-flash-lite", # Ou o modelo flash atual que você usa no seu ambiente
            contents=contents
        )
        texto_extraido_ocr = res_ocr.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no OCR (Gemini): {str(e)}")

    # 2. ANÁLISE QUANTITATIVA VIA GROQ COM GPT-OSS-120B
    groq_client = get_groq_client()
    system_instruction_mie2 = montar_system_prompt_mie2(sport=sport, foco=foco, analyst=analyst)

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_instruction_mie2},
                {"role": "user", "content": f"OCR DOS PRINTS:\n{texto_extraido_ocr}\n\nResponda APENAS com o JSON bruto, sem nenhuma explicação ou saudação."}
            ],
            temperature=0.1,
            max_tokens=4096
        )

        content_str = completion.choices[0].message.content or ""
        content_str = content_str.strip()

        if not content_str:
            raise HTTPException(status_code=500, detail="A Groq retornou uma resposta vazia com o modelo gpt-oss-120b.")

        # LIMPEZA UNIVERSAL (Isola perfeitamente o JSON entre chaves)
        match = re.search(r'\{.*\}', content_str, re.DOTALL)
        if match:
            content_str = match.group(0)
        
        if "```" in content_str:
            content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
            content_str = re.sub(r"\s*```$", "", content_str)
            content_str = content_str.strip()

        return json.loads(content_str)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Erro de formatação JSON retornado pela IA: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento (Groq): {str(e)}")
