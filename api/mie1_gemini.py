"""
Integrações com Gemini: OCR/extração estruturada dos prints, e MIE1
(pesquisa com grounding na web pra pegar estatísticas reais e atuais
dos times). Tudo que fala com a API do Gemini mora aqui.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

"""
Integrações com Gemini: OCR/extração estruturada dos prints, e MIE1
(pesquisa com grounding na web pra pegar estatísticas reais e atuais
dos times). Tudo que fala com a API do Gemini mora aqui.
"""

import os
from typing import Optional
from fastapi import HTTPException
from google import genai
from google.genai import types

from catalogos import CONFIG_MERCADO_PRINCIPAL, FONTES_AUTORIZADAS_POR_ESPORTE
from utils import _extrair_json_de_texto

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_gemini_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada na Vercel.")
    return genai.Client(api_key=GEMINI_API_KEY)


def extrair_mercados_estruturados(gemini_client, contents: list, sport: str) -> Optional[dict]:
    cfg = CONFIG_MERCADO_PRINCIPAL.get(sport.lower(), CONFIG_MERCADO_PRINCIPAL["futebol"])

    bloco_futebol_extra = ""
    if sport.lower() == "futebol":
        bloco_futebol_extra = """,
  "mercados_escanteios": [
    {"linha": 9.5, "odd": 1.90, "lado": "over"}
  ],
  "mercados_cartoes": [
    {"linha": 3.5, "odd": 1.80, "lado": "over"}
  ],
  "mercado_btts": {"odd_sim": 1.75, "odd_nao": 2.00}"""
    elif sport.lower() in ("basquete", "nfl"):
        bloco_futebol_extra = """,
  "mercados_player_props": [
    {"jogador": "Nome do Atleta", "prop": "Pontos/Jardas/Rebotes", "linha": 24.5, "odd": 1.85, "lado": "over"}
  ]"""

    prompt = f"""Extraia dos prints, em JSON estrito (sem markdown, sem texto fora do JSON):
{{
  "time_a": "Nome do primeiro time mencionado",
  "time_b": "Nome do segundo time mencionado",
  "mercados_total_principal": [
    {{"linha": 215.5, "odd": 1.85, "lado": "under"}}
  ]{bloco_futebol_extra}
}}
Regras:
- "mercados_total_principal" deve conter APENAS linhas de {cfg['nome_mercado']} (Over/Under de {cfg['nome_stat']}) que estejam explicitamente visíveis nos prints, com odd real.
- Cada lista deve conter APENAS linhas Over/Under desse mercado que estejam explicitamente visíveis nos prints, com odd real.
- "lado" deve ser exatamente "over" ou "under".
- Se não houver nenhuma linha de um mercado, retorne a lista vazia [] para ele.
- NUNCA invente time, linha ou odd que não esteja no print."""

    try:
        res = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents + [prompt],
            config=types.GenerateContentConfig(temperature=0)
        )
        dados = _extrair_json_de_texto(res.text)
        print(f"[EXTRACAO ESTRUTURADA DEBUG] {dados}")
        return dados
    except Exception as e:
        print(f"[EXTRACAO ESTRUTURADA ERRO] {str(e)}")
        return None


def executar_mie1(gemini_client, time_a: str, time_b: str, sport: str) -> Optional[dict]:
    esporte_key = sport.lower()
    cfg = CONFIG_MERCADO_PRINCIPAL.get(esporte_key, CONFIG_MERCADO_PRINCIPAL["futebol"])
    fontes = FONTES_AUTORIZADAS_POR_ESPORTE.get(esporte_key, FONTES_AUTORIZADAS_POR_ESPORTE["futebol"])
    nome_stat = cfg["nome_stat"]

    bloco_futebol_campos = ""
    bloco_futebol_regra = ""
    if esporte_key == "futebol":
        bloco_futebol_campos = """,
  "team_a_escanteios_projected": 5.2,
  "team_b_escanteios_projected": 4.8,
  "team_a_cartoes_projected": 2.1,
  "team_b_cartoes_projected": 1.9"""
        bloco_futebol_regra = "\n- Inclua também os campos adicionais para escanteios e cartões se disponíveis."

    prompt = f"""Você é um Investigador Quantitativo Esportivo. Busque na internet, OBRIGATORIAMENTE
usando o operador de busca {fontes}, as estatísticas mais recentes e confiáveis dos
times "{time_a}" e "{time_b}" para {sport.upper()}.

Investigue também fatores contextuais atuais: desfalques confirmados, lesões, clima
no horário do jogo, fadiga de calendário.

Retorne ESTRITAMENTE este JSON, sem markdown, sem texto fora do JSON:
{{
  "team_a_projected": 112.4,
  "team_b_projected": 108.1{bloco_futebol_campos},
  "fonte": "nome do site usado",
  "contextual_factors": [
    {{"factor_type": "injury", "description": "...", "impact_level": "high|medium|low", "affected_team": "..."}}
  ],
  "key_asymmetries": [
    {{"clash": "...", "statistical_evidence": "...", "betting_angle": "..."}}
  ]
}}

Regras:
- "team_a_projected"/"team_b_projected" são a expectativa de {nome_stat.upper()} de cada
  time NESTE confronto — já cruzando o ataque/ofensiva de um time com a defesa do
  outro, baseado em dados reais e atuais encontrados na busca.{bloco_futebol_regra}
- Se não encontrar dado confiável para {nome_stat.upper()} de ambos os times, retorne
  null no lugar do JSON inteiro — sem inventar números."""

    try:
        res = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        dados = _extrair_json_de_texto(res.text)
        print(f"[MIE1 DEBUG] time_a={time_a} time_b={time_b} sport={sport} resultado={dados}")

        if not dados or dados.get("team_a_projected") is None or dados.get("team_b_projected") is None:
            return None

        return dados
    except Exception as e:
        print(f"[MIE1 ERRO] {str(e)}")
        return None
