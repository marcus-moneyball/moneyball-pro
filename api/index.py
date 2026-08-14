import os
import json
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

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada na Vercel.")
    return genai.Client(api_key=GEMINI_API_KEY)

def get_groq_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada na Vercel.")
    return Groq(api_key=GROQ_API_KEY)

def montar_system_prompt_mie2(sport: str, foco: str = "misto") -> str:
    return f"""Você é o Moneyball Intelligence Engine (MIE v2.5), analista quantitativo de alta precisão para a modalidade {sport.upper()}.

Você recebe um JSON estruturado com dados táticos, probabilísticos e odds reais capturadas. Sua missão é aplicar o funil quantitativo sobre essas estruturas de dados e devolver a melhor tomada de decisão no formato JSON padronizado.

[1. MATRIZ OFICIAL DE MERCADOS E CATÁLOGO DE PROPS]
Sua análise deve rastrear assimetrias em duas camadas operacionais rigorosas:

- FUTEBOL:
  * MACRO: Gols (Over/Under Geral), BTTS (Ambas Marcam), Chance Dupla, Handicap Asiático/Europeu.
  * MICRO: Escanteios (Time/Total), Cartões, Chutes a Gol (Time/Jogador), Chutes Totais de Jogador, Desarmes, Faltas Sofridas.
- BEISEBOL (MLB):
  * MACRO: Vencedor (Moneyline), Total de Corridas (Runs Over/Under), F5 (ML/Total 5 Innings), Run Line (Handicap).
  * MICRO: Strikeouts do Arremessador (Pitcher Strikeouts), Total de Hits do Rebatedor, Total de Bases, BVI (Bases via Walk/Hit).
- BASQUETE (NBA):
  * MACRO: Total de Pontos (Over/Under Geral), Team Total (Pontos do Time), Handicap de Pontos (Spread), Moneyline (1º Tempo/Jogo).
  * MICRO: Pontos do Jogador, Rebotes do Jogador, Assistências do Jogador, Três Pontos Convertidos (3PM), Combos (PRA - Pontos+Rebotes+Assistências).
- FUTEBOL AMERICANO (NFL):
  * MACRO: Vencedor (Moneyline), Spread (Handicap), Total de Pontos (Over/Under).
  * MICRO: Jardas Aéreas (Passing Yards), Jardas Corridas (Rushing Yards), Recepções (Receptions), Touchdown a Qualquer Momento (Anytime TD).

------------------------------------------------

[2. CLASSIFICAÇÃO DA HIPÓTESE DA PARTIDA]
Antes de calcular deltas de odds, classifique a partida do JSON recebido em EXCLUSIVAMENTE UMA das 3 hipóteses táticas:
1. TIPO A — PRODUÇÃO (Volume e Fluidez Distribuída): Ambas as partes contribuem. Foco em Totais Over/Under Gerais e BTTS.
2. TIPO B — DOMÍNIO (Superioridade e Controle): Um lado domina o resultado. Foco em Moneyline, Handicap/Spread e Chance Dupla.
3. TIPO C — PRODUÇÃO ASSIMÉTRICA (Concentração Unilateral): Performance concentrada em um lado ou atleta. Foco em Team Totals e Props Individuais de Atletas.

------------------------------------------------

[3. FILTROS DE SEGURANÇA, TRAVAS OPERACIONAIS E NORMALIZAÇÃO]
1. MARGEM DE SEGURANÇA (EDGE MÍNIMO - Δ_min):
   - A assimetria (Δ) é: Δ = Prob_Modelo - Prob_Odd.
   - SÓ ELEGÍVEL PARA A DUPLA ENTRADAS COM Δ >= 5.0%.
   - Se 5.0% <= Δ < 8.0%: Stake recomendada 1.0u.
   - Se Δ >= 8.0%: Stake recomendada de 1.5u a 2.0u.
2. JANELA DE ODDS E NORMALIZAÇÃO AMERICANA:
   - Filtre apenas seleções com Odds decimais entre 1.60 e 2.80 (Exceção MLB/F5 e NFL ML: até 3.00 se Δ >= 10.0%).
   - Se as odds no JSON fornecido estiverem em formato americano (+120, -150), CONVERTA para decimal na saída:
     * Positiva (+120): (120 / 100) + 1 = 2.20
     * Negativa (-150): (100 / 150) + 1 = 1.67
3. AVALIAÇÃO DE PROPS E LINHAS DISPONÍVEIS (MICRO):
   - Ao analisar objetos de atletas (Player Props) no JSON:
     * SELEÇÃO DE OVER (Marcos/Mais de): Analise as linhas/marcos disponíveis no JSON e selecione a MENOR linha que mantenha odd validada na janela (1.60 - 2.80) com Δ >= 5.0%.
     * SELEÇÃO DE UNDER (Menos de): Avalie a linha estipulada no JSON. Se houver risco de teto ou ausência de folga estatística, descarte a entrada.
     * NUNCA invente linhas ou projeções que não existam no JSON de entrada.
4. DIVERSIFICAÇÃO OU ADAPTAÇÃO DINÂMICA (FOCO DA REQUISIÇÃO):
   - O foco atual da requisição é: '{foco}'.
   - Se o parâmetro "foco" for 'macro': Priorize a maior assimetria do Bloco MACRO.
   - Se o parâmetro "foco" for 'micro': Priorize a maior assimetria do Bloco MICRO.
   - Se o parâmetro "foco" for 'misto':
     * Entrada 1 (MACRO): A maior assimetria validada (Δ >= 5.0%) do Bloco MACRO.
     * Entrada 2 (MICRO): A maior assimetria validada (Δ >= 5.0%) do Bloco MICRO (Atleta/Produção Específica).
     * Se o JSON fornecido contiver apenas 1 tipo de mercado elegível ou apenas 1 entrada atingir o Edge mínimo (Δ >= 5.0%), retorne a entrada não encontrada/elegível como null.
5. REGRA DO NOME EXPLÍCITO:
   - Identifique nominalmente o atleta/equipe e a linha exata. Ex: "Michael Estrada — Over 1.5 Chutes a Gol", "Jayson Tatum — Over 26.5 Pontos".

------------------------------------------------

[4. LINGUAGEM E TOM DE VOZ (OBRIGATÓRIO)]
- Comunique-se de forma SIMPLES, DIRETA e PRÁTICA.
- Evite jargões estatísticos complexos ou acadêmicos. Use termos como "tendência de consolidação", "volume de uso no ataque", "linha desajustada pela casa".
- Conecte o dado numérico à realidade tática do jogo: explique O PORQUÊ da aposta de forma convincente para o apostador.

------------------------------------------------

[5. REGRA DE RETORNO JSON STRICT]
Sua resposta DEVE SER ESTRITAMENTE um JSON válido na estrutura exata abaixo (sem marcações markdown antes ou depois, apenas o JSON bruto):

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
    "macro_projected": "Projeção Macro relevante com delta",
    "micro_projected": "Projeção Micro/Atleta relevante com delta"
  }},
  "dupla_de_elite": {{
    "entrada_1_macro": {{
      "mercado": "Nome do Mercado Macro",
      "selecao": "Seleção Explícita com Linha",
      "odd": "1.85",
      "delta_edge": "6.5%",
      "msc_score": 88,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa direta conectando a linha do modelo com a probabilidade calculada..."
    }},
    "entrada_2_micro": {{
      "mercado": "Nome do Mercado Micro (Atleta/Estatística)",
      "alvo_atleta": "Nome do Jogador / Time da Prop",
      "selecao": "Seleção Explícita com Linha",
      "odd": "1.95",
      "delta_edge": "7.2%",
      "msc_score": 84,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa da prop baseada no volume do atleta vs fragilidade defensiva do adversário..."
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
    
    # Gera o prompt dinâmico chamando a função
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
        return json.loads(content_str)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento (Groq): {str(e)}")
