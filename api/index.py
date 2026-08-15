import os
import json
import re
import math
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from groq import Groq
import scipy.stats as stats

app = FastAPI(title="MoneyballPro Engine", version="2.5.0")

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

# ============================================================
# CAMADA DE CÁLCULO DETERMINÍSTICO MULTI-ESPORTE
# (Poisson para Futebol/Gols/Escanteios/Cartões/TDs/Strikeouts)
# (Distribuição Normal para Basquete/Pontos/NFL Jardas)
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def prob_over_under_poisson(linha: float, lam: float):
    """Para contagens discretas (futebol, cartões, escanteios, strikeouts, TDs)."""
    piso = math.floor(linha)
    p_under = poisson_cdf(piso, lam)
    p_over = 1.0 - p_under
    return round(p_over, 4), round(p_under, 4)


def prob_over_under_normal(linha: float, media: float, desvio_padrao: float = 11.5):
    """
    Para pontuações altas/variáveis contínuas (Basquete Pontos, NFL Jardas).
    Usa Distribuição Normal Acumulada (CDF).
    """
    if desvio_padrao <= 0:
        desvio_padrao = 10.0
    p_under = stats.norm.cdf(linha, loc=media, scale=desvio_padrao)
    p_over = 1.0 - p_under
    return round(float(p_over), 4), round(float(p_under), 4)


def calcular_delta_mercado(lam: float, linha: float):
    delta_abs = round(lam - linha, 3)
    delta_pct = round((delta_abs / linha) * 100, 2) if linha else None
    return delta_abs, delta_pct


def calcular_ev(prob_real: float, odd_decimal: float):
    if prob_real is None or odd_decimal is None:
        return None
    return round((prob_real * odd_decimal) - 1, 4)


def kelly_fracionado(prob_real: float, odd_decimal: float, fracao=0.25, teto_unidades=2.5):
    if prob_real is None or odd_decimal is None or odd_decimal <= 1:
        return None
    b = odd_decimal - 1
    p = prob_real
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0:
        return None
    unidades = round(f_star * fracao * 10, 2)
    return min(unidades, teto_unidades)


def estimar_lambda(mercado: dict):
    tipo = mercado.get("tipo", "total_jogo")
    marcada_a = mercado.get("media_marcada_time_a")
    sofrida_a = mercado.get("media_sofrida_time_a")
    marcada_b = mercado.get("media_marcada_time_b")
    sofrida_b = mercado.get("media_sofrida_time_b")

    if None in (marcada_a, sofrida_a, marcada_b, sofrida_b):
        return None

    esperado_a = (marcada_a + sofrida_b) / 2
    esperado_b = (marcada_b + sofrida_a) / 2

    if tipo == "total_time_a":
        return round(esperado_a, 3)
    if tipo == "total_time_b":
        return round(esperado_b, 3)
    return round(esperado_a + esperado_b, 3)


def calcular_mercado(mercado: dict, esporte: str = "futebol") -> dict:
    linha = mercado.get("linha")
    if linha is None:
        return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}

    esporte_key = esporte.lower()

    # Seleção de Modelo Matemático por Esporte/Mercado
    if esporte_key in ("basquete", "nfl") and mercado.get("modelo") != "poisson":
        # Usamos Normal para Pontos/Jardas
        media_esperada = mercado.get("media_esperada") or estimar_lambda(mercado)
        if media_esperada is None:
            return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}
        
        std_dev = mercado.get("desvio_padrao", 12.0 if esporte_key == "basquete" else 18.5)
        p_over, p_under = prob_over_under_normal(linha, media_esperada, std_dev)
        lam_ref = media_esperada
    else:
        # Usamos Poisson para Futebol, Beisebol e Props discretos (Cartões, Cantos, Strikeouts, TDs)
        lam_ref = estimar_lambda(mercado) if mercado.get("media_esperada") is None else mercado.get("media_esperada")
        if lam_ref is None:
            return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}
        p_over, p_under = prob_over_under_poisson(linha, lam_ref)

    odd = mercado.get("odd_real_decimal")
    delta_abs, delta_pct = calcular_delta_mercado(lam_ref, linha)

    resultado = {
        "id": mercado.get("id"),
        "status": "calculado",
        "esperado_estimado": lam_ref,
        "probabilidade_over": p_over,
        "probabilidade_under": p_under,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "ev": None,
        "kelly_unidades": None,
    }

    if odd is not None:
        lado = mercado.get("lado_odd", "over")
        prob_desse_lado = p_over if lado == "over" else p_under
        ev = calcular_ev(prob_desse_lado, odd)
        resultado["ev"] = ev
        if ev is not None and ev > 0:
            resultado["kelly_unidades"] = kelly_fracionado(prob_desse_lado, odd)

    return resultado


def calcular_dossie(mercados: list, esporte: str = "futebol") -> list:
    resultados = []
    for m in mercados:
        try:
            resultados.append(calcular_mercado(m, esporte=esporte))
        except Exception as e:
            resultados.append({"id": m.get("id"), "status": "erro_calculo", "detalhe": str(e)})
    return resultados

# ============================================================
# FIM DA CAMADA DE CÁLCULO
# ============================================================

REGRAS_ESPORTES = {
    "futebol": "Mercados: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor.",
    "nfl": "Mercados: Vencedor (Moneyline), Spread (Handicap), Total de Pontos, Yardas de Passe/Corrida/Recepção, Touchdown Qualquer Momento."
}

PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 3.5,
        "odd_min": 1.50,
        "odd_max": 3.50,
        "faixas_stake": [
            (3.5, 6.0, "1.0u"),
            (6.0, 8.5, "1.5u"),
            (8.5, float("inf"), "2.0u"),
        ],
    },
    "cris": {
        "delta_min": 4.0,
        "odd_min": 1.60,
        "odd_max": 2.80,
        "faixas_stake": [
            (4.0, 6.5, "1.0u"),
            (6.5, 9.0, "1.5u"),
            (9.0, float("inf"), "2.0u"),
        ],
    },
}

DELTA_MAX_PLAUSIVEL = 15.0


def get_gemini_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada na Vercel.")
    return genai.Client(api_key=GEMINI_API_KEY)


def get_groq_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada na Vercel.")
    return Groq(api_key=GROQ_API_KEY)


def _extrair_json_de_texto(texto: str) -> Optional[dict]:
    if not texto:
        return None
    texto = texto.strip()
    if "```" in texto:
        texto = re.sub(r"^```(?:json)?\s*", "", texto)
        texto = re.sub(r"\s*```$", "", texto)
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


CONFIG_MERCADO_PRINCIPAL = {
    "futebol": {"nome_stat": "gols", "nome_mercado": "Total de Gols da Partida", "unidade_selecao": "Gols"},
    "beisebol": {"nome_stat": "runs", "nome_mercado": "Total de Runs da Partida", "unidade_selecao": "Runs"},
    "basquete": {"nome_stat": "pontos", "nome_mercado": "Total de Pontos da Partida", "unidade_selecao": "Pontos"},
    "nfl": {"nome_stat": "pontos", "nome_mercado": "Total de Pontos da Partida", "unidade_selecao": "Pontos"},
}


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


FONTES_AUTORIZADAS_POR_ESPORTE = {
    "futebol": "site:fbref.com OR site:sofascore.com",
    "basquete": "site:basketball-reference.com OR site:nba.com",
    "beisebol": "site:baseballsavant.com OR site:baseball-reference.com",
    "nfl": "site:pro-football-reference.com",
}


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


def montar_candidatos_over_under_calculados(
    mercados: list, lam_total: Optional[float], nome_mercado: str, unidade_selecao: str, esporte: str = "futebol"
) -> list:
    if lam_total is None:
        return []

    candidatos = []
    esporte_key = esporte.lower()

    for m in mercados:
        linha = m.get("linha")
        odd = m.get("odd")
        lado = m.get("lado")
        if linha is None or odd is None or lado not in ("over", "under"):
            continue

        # Cálculo conforme o esporte
        if esporte_key in ("basquete", "nfl") and "escanteios" not in nome_mercado.lower() and "cartoes" not in nome_mercado.lower():
            std_dev = 12.0 if esporte_key == "basquete" else 18.5
            p_over, p_under = prob_over_under_normal(linha, lam_total, std_dev)
        else:
            p_over, p_under = prob_over_under_poisson(linha, lam_total)

        prob_real = p_under if lado == "under" else p_over
        prob_implicita_odd = round(1 / odd, 4) if odd else None
        edge_pct = round((prob_real - prob_implicita_odd) * 100, 2) if prob_implicita_odd is not None else None
        ev = calcular_ev(prob_real, odd)
        kelly = kelly_fracionado(prob_real, odd) if ev is not None and ev > 0 else None

        candidatos.append({
            "mercado": nome_mercado,
            "selecao": f"{'Mais' if lado == 'over' else 'Menos'} de {linha} {unidade_selecao}",
            "odd": odd,
            "esperado_partida": lam_total,
            "probabilidade_real_calculada": prob_real,
            "probabilidade_implicita_odd": prob_implicita_odd,
            "delta_edge_pct_calculado": edge_pct,
            "ev": ev,
            "kelly_unidades_sugerido": kelly,
        })

    return candidatos


def montar_candidato_btts(mercado_btts: Optional[dict], lam_a: Optional[float], lam_b: Optional[float]) -> list:
    if not mercado_btts or lam_a is None or lam_b is None:
        return []

    p_a_marca = 1 - poisson_pmf(0, lam_a)
    p_b_marca = 1 - poisson_pmf(0, lam_b)
    p_sim = round(p_a_marca * p_b_marca, 4)
    p_nao = round(1 - p_sim, 4)

    candidatos = []
    odd_sim = mercado_btts.get("odd_sim")
    odd_nao = mercado_btts.get("odd_nao")

    if odd_sim:
        prob_implicita = round(1 / odd_sim, 4)
        edge_pct = round((p_sim - prob_implicita) * 100, 2)
        ev = calcular_ev(p_sim, odd_sim)
        candidatos.append({
            "mercado": "Ambos Marcam (BTTS)",
            "selecao": "Sim",
            "odd": odd_sim,
            "probabilidade_real_calculada": p_sim,
            "probabilidade_implicita_odd": prob_implicita,
            "delta_edge_pct_calculado": edge_pct,
            "ev": ev,
            "kelly_unidades_sugerido": kelly_fracionado(p_sim, odd_sim) if ev is not None and ev > 0 else None,
        })

    if odd_nao:
        prob_implicita = round(1 / odd_nao, 4)
        edge_pct = round((p_nao - prob_implicita) * 100, 2)
        ev = calcular_ev(p_nao, odd_nao)
        candidatos.append({
            "mercado": "Ambos Marcam (BTTS)",
            "selecao": "Não",
            "odd": odd_nao,
            "probabilidade_real_calculada": p_nao,
            "probabilidade_implicita_odd": prob_implicita,
            "delta_edge_pct_calculado": edge_pct,
            "ev": ev,
            "kelly_unidades_sugerido": kelly_fracionado(p_nao, odd_nao) if ev is not None and ev > 0 else None,
        })

    return candidatos


def montar_system_prompt_mie2(sport: str, analyst: str = "carlos") -> str:
    esporte_key = sport.lower()
    catalogo_esporte = REGRAS_ESPORTES.get(esporte_key, REGRAS_ESPORTES["futebol"])

    perfil = PERFIS_ANALISTA.get(analyst, PERFIS_ANALISTA["carlos"])
    delta_min = perfil["delta_min"]
    odd_min = perfil["odd_min"]
    odd_max = perfil["odd_max"]

    tabela_stake = "\n".join(
        f"   - Se {lo}% <= Δ < {hi}%: Stake {stake}."
        if hi != float("inf")
        else f"   - Se Δ >= {lo}%: Stake {stake}."
        for lo, hi, stake in perfil["faixas_stake"]
    )

    if analyst == "cris":
        persona_nome = "Cris (A Especialista em Tiro Certo)"
        persona_curto = "Cris"
        persona_regras = """- FILOSOFIA: Ultra-conservadora, focada primariamente na proteção implacável de banca.
- ANÁLISE: Rejeite qualquer risco desnecessário. Priorize apostas simples seguras ou duplas apenas com altíssima convicção.
- QUANDO HOUVER MAIS DE UMA LINHA POSSÍVEL NO MESMO MERCADO-BASE, PREFIRA SEMPRE A LINHA ESTATISTICAMENTE MAIS PROVÁVEL DE BATER, mesmo que signifique um delta_edge menor.
- TOM DE VOZ: Sóbrio, direto, focado na mitigação de risco."""
    else:
        persona_nome = "Carlos (O Estrategista Técnico)"
        persona_curto = "Carlos"
        persona_regras = """- FILOSOFIA: Técnico, elegante e letal, atuando como um boxeador de elite no ringue do mercado financeiro esportivo.
- ANÁLISE: Varre os mercados em busca de valor oculto e assimetria que as casas de apostas não precificaram corretamente.
- TOM DE VOZ: Analítico, astuto, confiante, tático, usando o jargão de inteligência de mercado."""

    return f"""Você é {persona_nome}, e utiliza o Moneyball Intelligence Engine (MIE v2.5) como ferramenta quantitativa para a modalidade {sport.upper()}.

{persona_regras}

Você recebe a transcrição OCR de 2 a 5 prints contendo dados táticos, probabilísticos e odds reais capturadas. Sua missão é aplicar o funil quantitativo sobre TODOS os mercados e devolver até 2 tomadas de decisão no formato JSON padronizado.

[1. MATRIZ OFICIAL DE MERCADOS]
Analise livremente os mercados permitidos para {sport.upper()}:
{catalogo_esporte}

------------------------------------------------

[2. CLASSIFICAÇÃO DA HIPÓTESE DA PARTIDA]
Classifique a partida em EXCLUSIVAMENTE UMA das 3 hipóteses táticas:
1. TIPO A — PRODUÇÃO (Volume e Fluidez Distribuída): Ambas as partes contribuem.
2. TIPO B — DOMÍNIO (Superioridade e Controle): Um lado domina o resultado.
3. TIPO C — PRODUÇÃO ASSIMÉTRICA (Concentração Unilateral): Performance concentrada em um lado ou atleta.

------------------------------------------------

[3. FILTROS DE SEGURANÇA E REGRA DOS DOIS MELHORES EDGES]
0. DADOS JÁ CALCULADOS (se fornecidos no cabeçalho "CANDIDATOS JÁ CALCULADOS"):
   - Para os mercados listados nesse bloco, o valor de Δ NÃO deve ser estimado por você — ele já foi calculado deterministicamente (Poisson/Normal Gaussiana em Python) a partir de estatísticas reais da web.
   - Use EXATAMENTE o "delta_edge_pct_calculado", "odd" e "selecao" fornecidos ali, sem alterar nenhum número.
   - Você pode e deve continuar estimando Δ normalmente para QUALQUER mercado que NÃO apareça nesse bloco (ex: props de jogadores).

1. MARGEM DE SEGURANÇA (EDGE MÍNIMO - Δ_min = {delta_min}% para {persona_curto}):
   - Δ = Prob_Modelo - Prob_Odd.
   - SÓ É ELEGÍVEL QUALQUER SELEÇÃO COM Δ >= {delta_min}%.
{tabela_stake}

2. SELEÇÃO DA DUPLA DE ELITE:
   - Entrada 1: Maior assimetria validada (Δ >= {delta_min}%).
   - Entrada 2: Segunda maior assimetria validada (Δ >= {delta_min}%).
   - UNICIDADE DE MERCADO: Proibido sugerir duas entradas do mesmo mercado base.
   - Se Entrada 1 for DEPENDENTE da hipótese_partida, a Entrada 2 DEVE ser INDEPENDENTE, se houver elegível.

3. JANELA DE ODDS ({persona_curto}): Cotações entre {odd_min} e {odd_max}.

4. REGRA DO NOME EXPLÍCITO:
   - Proibido retornar "Sim", "Não", "Mais" ou "Menos" solto. O campo "seleção" deve conter a descrição completa.

------------------------------------------------

[4. REGRAS DE BLOQUEIO]
1. BLOQUEIO TOTAL DE MONEYLINE (ML): Proibido sugerir vitória seca. Foque em volume, handicaps ou estatísticas.
2. FILTRO ANTI-ESTRELA: Proibido favoritos abaixo de @1.50 sem linha de segurança.
3. PROTEÇÃO CONTRA JOGOS TRUNCADOS: No Futebol, proibido "Mais de 2.5 Gols" em jogos travados.

------------------------------------------------

[5. REGRA DE RETORNO JSON STRICT]
Retorne ESTRITAMENTE o JSON estruturado do MIE2, sem marcações markdown fora da estrutura.

{{
  "perfil_geral": "Síntese quantitativa...",
  "status_geral": "processado_com_sucesso",
  "hipotese_partida": "TIPO A | TIPO B | TIPO C",
  "stake_medio_partida": "1.0u",
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Time A vs Time B",
    "date": "Hoje"
  }},
  "expected_projections": {{
    "macro_projected": "Projeção macro",
    "micro_projected": "Projeção micro"
  }},
  "dupla_de_elite": {{
    "entrada_1": {{
      "categoria": "MACRO ou MICRO",
      "dependencia_hipotese": "DEPENDENTE ou INDEPENDENTE",
      "mercado": "Nome do Mercado",
      "selecao": "Seleção Explícita",
      "odd": "1.85",
      "delta_edge": "7.6%",
      "msc_score": 90,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa..."
    }},
    "entrada_2": null
  }},
  "key_asymmetries": []
}}
"""


def _parse_float_seguro(valor) -> Optional[float]:
    if valor is None:
        return None
    try:
        texto = str(valor).strip().replace("%", "").replace(",", ".")
        return float(texto)
    except (ValueError, TypeError):
        return None


def validar_e_sanear_entrada(entrada: Optional[dict], perfil: dict) -> Optional[dict]:
    if not entrada or not isinstance(entrada, dict) or not entrada.get("mercado"):
        return None

    odd = _parse_float_seguro(entrada.get("odd"))
    delta = _parse_float_seguro(entrada.get("delta_edge"))

    if odd is None or delta is None:
        return None

    if not (perfil["odd_min"] <= odd <= perfil["odd_max"]):
        return None

    if delta < perfil["delta_min"]:
        return None

    if delta > DELTA_MAX_PLAUSIVEL:
        return None

    return entrada


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "engine": "MoneyballPro FastAPI v2.5 (Cálculo Determinístico: Poisson + Normal Gaussiana)",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "groq_key_set": bool(GROQ_API_KEY),
        "perfis_analista": {
            nome: {"delta_min": p["delta_min"], "odd_min": p["odd_min"], "odd_max": p["odd_max"]}
            for nome, p in PERFIS_ANALISTA.items()
        },
    }


@app.post("/api/v1/calc")
async def calcular_mercados(payload: dict):
    mercados = payload.get("mercados")
    esporte = payload.get("esporte", "futebol")
    if not mercados or not isinstance(mercados, list):
        raise HTTPException(
            status_code=400,
            detail='Corpo inválido. Esperado: { "esporte": "basquete", "mercados": [ {...} ] }'
        )
    return {"resultados": calcular_dossie(mercados, esporte=esporte)}


@app.post("/api/v1/analyze")
async def analyze_tickets(
    sport: str = Form(...),
    analyst: str = Form("carlos"),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    perfil = PERFIS_ANALISTA.get(analyst, PERFIS_ANALISTA["carlos"])

    # 1. OCR + Leitura de Imagem via Gemini Flash Lite
    gemini_client = get_gemini_client()
    contents = []

    for file in files:
        file_bytes = await file.read()
        contents.append(
            types.Part.from_bytes(
                data=file_bytes,
                mime_type=file.content_type or "image/jpeg",
            )
        )

    # 2. Extração Estruturada dos Prints
    dados_estruturados = extrair_mercados_estruturados(gemini_client, contents, sport)
    
    candidatos_calculados = []
    
    if dados_estruturados and dados_estruturados.get("time_a") and dados_estruturados.get("time_b"):
        time_a = dados_estruturados["time_a"]
        time_b = dados_estruturados["time_b"]
        
        # 3. MIE1: Pesquisa Grounding na Web
        mie1_data = executar_mie1(gemini_client, time_a, time_b, sport)
        
        if mie1_data:
            lam_a = mie1_data.get("team_a_projected")
            lam_b = mie1_data.get("team_b_projected")
            
            if lam_a is not None and lam_b is not None:
                lam_total = lam_a + lam_b
                cfg = CONFIG_MERCADO_PRINCIPAL.get(sport.lower(), CONFIG_MERCADO_PRINCIPAL["futebol"])
                
                # Cálculo da linha principal
                candidatos_calculados.extend(
                    montar_candidatos_over_under_calculados(
                        dados_estruturados.get("mercados_total_principal", []),
                        lam_total,
                        cfg["nome_mercado"],
                        cfg["unidade_selecao"],
                        esporte=sport
                    )
                )

                # Se for futebol, calcula Cantos, Cartões e BTTS
                if sport.lower() == "futebol":
                    cantos_a = mie1_data.get("team_a_escanteios_projected")
                    cantos_b = mie1_data.get("team_b_escanteios_projected")
                    if cantos_a and cantos_b:
                        candidatos_calculados.extend(
                            montar_candidatos_over_under_calculados(
                                dados_estruturados.get("mercados_escanteios", []),
                                cantos_a + cantos_b,
                                "Total de Escanteios da Partida",
                                "Escanteios",
                                esporte=sport
                            )
                        )

                    cartoes_a = mie1_data.get("team_a_cartoes_projected")
                    cartoes_b = mie1_data.get("team_b_cartoes_projected")
                    if cartoes_a and cartoes_b:
                        candidatos_calculados.extend(
                            montar_candidatos_over_under_calculados(
                                dados_estruturados.get("mercados_cartoes", []),
                                cartoes_a + cartoes_b,
                                "Total de Cartões da Partida",
                                "Cartões",
                                esporte=sport
                            )
                        )

                    candidatos_calculados.extend(
                        montar_candidato_btts(dados_estruturados.get("mercado_btts"), lam_a, lam_b)
                    )

    # 4. MIE2: Execução com Groq / Llama
    groq_client = get_groq_client()
    system_prompt = montar_system_prompt_mie2(sport, analyst)

    user_prompt_content = "Analise os seguintes dados visuais dos tickets de apostas fornecidos e extraia o valor."
    
    if candidatos_calculados:
        user_prompt_content += f"\n\n[CANDIDATOS JÁ CALCULADOS PELO PYTHON]\n" + json.dumps(candidatos_calculados, indent=2, ensure_ascii=False)

    # OCR Primário pelo Gemini para entregar texto formatado ao Groq
    ocr_res = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents + ["Transcreva de forma limpa e estruturada todo o texto e números visíveis nestes prints."],
        config=types.GenerateContentConfig(temperature=0)
    )
    texto_ocr = ocr_res.text or ""

    groq_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt_content}\n\n[TRANSCRIÇÃO DOS PRINTS]\n{texto_ocr}"}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    resultado_final = json.loads(groq_response.choices[0].message.content)

    # 5. Validação e Saneamento das Entradas
    if resultado_final.get("dupla_de_elite"):
        e1 = resultado_final["dupla_de_elite"].get("entrada_1")
        e2 = resultado_final["dupla_de_elite"].get("entrada_2")

        resultado_final["dupla_de_elite"]["entrada_1"] = validar_e_sanear_entrada(e1, perfil)
        resultado_final["dupla_de_elite"]["entrada_2"] = validar_e_sanear_entrada(e2, perfil)

    return resultado_final
