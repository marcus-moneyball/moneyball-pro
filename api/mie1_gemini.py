"""
Módulo MIE1 - Integrações com a API do Gemini.
Responsável por OCR/extração estruturada dos prints e pesquisa MIE1 
com grounding na web para estatísticas reais e atuais dos times.
"""

import sys
import os
from typing import Optional
from fastapi import HTTPException
from google import genai
from google.genai import types

try:
    from api.catalogos import (
        CONFIG_MERCADO_PRINCIPAL,
        FONTES_AUTORIZADAS_POR_ESPORTE,
        CAMPOS_ROTEIRO_POR_ESPORTE,
    )
    from api.utils import _extrair_json_de_texto
except ImportError:
    from catalogos import (
        CONFIG_MERCADO_PRINCIPAL,
        FONTES_AUTORIZADAS_POR_ESPORTE,
        CAMPOS_ROTEIRO_POR_ESPORTE,
    )
    from utils import _extrair_json_de_texto

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_gemini_client() -> genai.Client:
    """Inicializa e retorna o cliente oficial do Google GenAI."""
    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY não configurada no ambiente."
        )
    return genai.Client(api_key=api_key)


def extrair_mercados_estruturados(
    gemini_client: genai.Client, contents: list, sport: str
) -> Optional[dict]:
    """Realiza OCR e extração estruturada de dados dos prints enviados via Gemini."""
    cfg = CONFIG_MERCADO_PRINCIPAL.get(
        sport.lower(), CONFIG_MERCADO_PRINCIPAL["futebol"]
    )

    bloco_futebol_extra = ""
    if sport.lower() == "futebol":
        bloco_futebol_extra = """,
  "mercados_escanteios": [
    {"linha": 9.5, "odd": "1.90", "lado": "over"}
  ],
  "mercados_cartoes": [
    {"linha": 3.5, "odd": "1.80", "lado": "over"}
  ],
  "mercado_btts": {"odd_sim": "1.75", "odd_nao": "2.00"},
  "mercado_chance_dupla": {"odd_1x": "1.30", "odd_x2": "1.40", "odd_12": "1.20"},
  "mercados_handicap_asiatico": [
    {"linha": -1.0, "time_referencia": "A", "odd_real_decimal": "1.90", "selecao_texto": "Time A (-1.0)"}
  ],
  "mercados_player_props": [
    {"jogador": "Nome do Atleta", "prop": "Chutes no Gol/Gols/Cartão/Assistências", "linha": 2.5, "odd": "1.90", "lado": "over"}
  ]"""
    elif sport.lower() == "basquete":
        bloco_futebol_extra = """,
  "mercados_player_props": [
    {"jogador": "Nome do Atleta", "prop": "Pontos/Jardas/Rebotes", "linha": 24.5, "odd": "1.85", "lado": "over"}
  ],
  "mercado_moneyline": {"odd_time_a": "-150", "odd_time_b": "+130"}"""
    elif sport.lower() == "beisebol":
        bloco_futebol_extra = """,
  "mercados_player_props": [
    {"jogador": "Nome do Atleta", "prop": "Strikeouts/Hits/Total Bases/RBIs/Home Runs", "linha": 5.5, "odd": "1.85", "lado": "over"}
  ],
  "mercado_moneyline": {"odd_time_a": "-150", "odd_time_b": "+130"}"""

    prompt = f"""Extraia dos prints, em JSON estrito (sem markdown, sem texto fora do JSON):
{{
  "casa_apostas": "Nome da casa de apostas, se visível no print (ex: cabeçalho do app, barra de navegação, marca d'água) -- null se não for possível identificar com confiança",
  "time_a": "Nome do primeiro time mencionado",
  "time_b": "Nome do segundo time mencionado",
  "mercados_total_principal": [
    {{"linha": 215.5, "odd": "1.85", "lado": "under"}}
  ]
  {bloco_futebol_extra}
}}
Regras:
- "mercados_total_principal" deve conter APENAS linhas de {cfg['nome_mercado']} (Over/Under de {cfg['nome_stat']}) que estejam explicitamente visíveis nos prints, com odd real.
- Cada lista deve conter APENAS linhas Over/Under desse mercado que estejam explicitamente visíveis nos prints, com odd real.
- "lado" deve ser exatamente "over" ou "under".
- Se não houver nenhuma linha de um mercado, retorne a lista vazia [] para ele (ou omita o campo, se for um objeto único como "mercado_moneyline"/"mercado_chance_dupla").
- "mercados_player_props" deve conter APENAS props de jogador explicitamente visíveis no print, com odd real. "jogador" é o nome completo do atleta como aparece no print. "prop" é o nome do mercado exatamente como aparece (ex: "Chutes no Gol", "Total Strikeouts", "Pontos"). "linha" e "odd" seguem as mesmas regras dos demais mercados.
- "mercado_moneyline" (beisebol/basquete): "odd_time_a" é a odd de vitória do
  primeiro time mencionado ("time_a"), "odd_time_b" a odd de vitória do
  segundo ("time_b"). Preencha só as que estiverem visíveis no print.
- "mercado_chance_dupla" (futebol): "odd_1x" = casa-ou-empate, "odd_x2" =
  empate-ou-fora, "odd_12" = casa-ou-fora (dupla chance sem empate). Preencha
  só as que estiverem visíveis.
- "mercados_handicap_asiatico" (futebol): "linha" é o valor do handicap
  exatamente como aparece no print (ex: -1.0, -0.5, -0.25, +0.75 -- mantenha o
  sinal correto). "time_referencia" é "A" se o handicap for do primeiro time
  mencionado, "B" se for do segundo. "selecao_texto" é o texto da seleção como
  aparece no print (ex: "Real Madrid (-1.5)").
- REGRA CRÍTICA DE FORMATO DE ODD -- todo campo de odd ("odd", "odd_sim",
  "odd_nao", "odd_1x", "odd_x2", "odd_12", "odd_real_decimal", "odd_time_a",
  "odd_time_b") deve ser extraído como TEXTO (string), EXATAMENTE como aparece
  no print -- nunca convertido ou reinterpretado por você. Casas de apostas
  diferentes mostram odds em formatos diferentes: decimal (ex: "1.85") ou
  americana, com sinal (ex: "-150", "+130"). Se a odd tiver sinal de "+" ou
  "-" no print, PRESERVE ESSE SINAL EXATO no texto extraído -- nunca omita o
  "-" de uma odd negativa, isso inverteria o significado (o favorito passaria
  a parecer zebra). Não faça nenhuma conta ou conversão você mesmo -- apenas
  copie o texto exato que está visível, a conversão pra decimal é feita depois
  por outro sistema.
- NUNCA invente time, linha ou odd que não esteja no print."""

    try:
        res = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents + [prompt],
            config=types.GenerateContentConfig(temperature=0),
        )
        dados = _extrair_json_de_texto(res.text)
        print(f"[EXTRACAO ESTRUTURADA DEBUG] {dados}")
        return dados
    except Exception as e:
        print(f"[EXTRACAO ESTRUTURADA ERRO] {str(e)}")
        return None


def executar_mie1(
    gemini_client: genai.Client, time_a: str, time_b: str, sport: str
) -> Optional[dict]:
    """Executa a pesquisa com Google Search Grounding (MIE1) para projetar estatísticas e investigar assimetrias do confronto."""
    esporte_key = sport.lower()
    cfg = CONFIG_MERCADO_PRINCIPAL.get(
        esporte_key, CONFIG_MERCADO_PRINCIPAL["futebol"]
    )
    fontes = FONTES_AUTORIZADAS_POR_ESPORTE.get(
        esporte_key, FONTES_AUTORIZADAS_POR_ESPORTE["futebol"]
    )
    nome_stat = cfg["nome_stat"]

    bloco_futebol_campos = ""
    bloco_futebol_regra = ""
    if esporte_key == "futebol":
        bloco_futebol_campos = """,
  "team_a_escanteios_projected": 5.2,
  "team_b_escanteios_projected": 4.8,
  "team_a_cartoes_projected": 2.1,
  "team_b_cartoes_projected": 1.9"""
        bloco_futebol_regra = (
            "\n- Inclua também os campos adicionais para escanteios e cartões se disponíveis."
        )

    # Antes isso era um dict fixo com só 3 campos por esporte -- desalinhado com o
    # catálogo real (CAMPOS_ROTEIRO_POR_ESPORTE), que já declarava campos como PPDA,
    # EFG%, WHIP, wRC+ etc. sem que o MIE1 nunca os pedisse de fato. Agora o exemplo
    # é gerado a partir do catálogo inteiro -- toda vez que alguém adicionar um campo
    # em catalogos.py, o MIE1 passa a pedir esse campo automaticamente, sem precisar
    # editar este arquivo de novo.
    def _gerar_exemplo_roteiro(esporte_key: str) -> str:
        campos = CAMPOS_ROTEIRO_POR_ESPORTE.get(esporte_key, [])
        if not campos:
            return ""
        linhas = ", ".join(f'"{campo}": null' for campo in campos)
        return "{" + linhas + "}"

    bloco_roteiro_campos = ""
    bloco_roteiro_regra = ""
    if esporte_key in CAMPOS_ROTEIRO_POR_ESPORTE:
        exemplo = _gerar_exemplo_roteiro(esporte_key)
        campos_lista = ", ".join(CAMPOS_ROTEIRO_POR_ESPORTE[esporte_key])
        bloco_roteiro_campos = f""",
  "team_a_roteiro": {exemplo},
  "team_b_roteiro": {exemplo}"""
        bloco_roteiro_regra = (
            f'\n- "team_a_roteiro"/"team_b_roteiro" trazem os campos de roteiro de jogo '
            f"para {sport.upper()} ({campos_lista}), buscados nas mesmas fontes. Preencha "
            f"CADA campo individualmente com o valor real encontrado, ou null se não achar "
            f"dado confiável para aquele campo específico -- NÃO deixe de retornar o restante "
            f"do JSON por causa de um campo de roteiro faltando. O exemplo acima mostra null "
            f"em todos os campos só pra ilustrar o formato -- substitua por números reais "
            f"sempre que encontrar."
        )
        if esporte_key == "beisebol":
            bloco_roteiro_regra += (
                '\n- "pitcher_mao" (dentro de team_a_roteiro/team_b_roteiro) deve ser '
                'exatamente "R" ou "L" (destro ou canhoto) -- nunca outro formato.'
                '\n- "lineup_ops_vs_mao_adversaria" deve refletir o OPS do lineup DAQUELE '
                'time especificamente contra a MÃO do arremessador ADVERSÁRIO (ex: se o '
                'arremessador do time B é destro, o "lineup_ops_vs_mao_adversaria" dentro de '
                '"team_a_roteiro" é o OPS do time A contra arremessadores destros -- não a '
                'média geral do lineup contra qualquer arremessador).'
            )

    prompt = f"""Você é um Investigador Quantitativo Esportivo. Busque na internet, OBRIGATORIAMENTE
usando o operador de busca {fontes}, as estatísticas mais recentes e confiáveis dos
times "{time_a}" e "{time_b}" para {sport.upper()}.

Investigue também fatores contextuais atuais: desfalques confirmados, lesões, clima
no horário do jogo, fadiga de calendário.

Retorne ESTRITAMENTE este JSON, sem markdown, sem texto fora do JSON:
{{
  "team_a_projected": 112.4,
  "team_b_projected": 108.1{bloco_futebol_campos}{bloco_roteiro_campos},
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
  outro, baseado em dados reais e atuais encontrados na busca.{bloco_futebol_regra}{bloco_roteiro_regra}
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
        print(
            f"[MIE1 DEBUG] time_a={time_a} time_b={time_b} sport={sport} resultado={dados}"
        )

        if (
            not dados
            or dados.get("team_a_projected") is None
            or dados.get("team_b_projected") is None
        ):
            return None

        return dados
    except Exception as e:
        print(f"[MIE1 ERRO] {str(e)}")
        return None
