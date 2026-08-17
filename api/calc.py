"""
Camada de cálculo determinístico multi-esporte (Delta + Poisson + Normal + Kelly).
Integrado com Análise de Assimetrias (Cris) e Matriz de Correlações (Carlos).
Sem chamadas de rede — 100% testável isoladamente.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
from typing import Optional
import scipy.stats as stats


# ============================================================
# PROBABILIDADE (Poisson / Normal)
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
    piso = math.floor(linha)
    p_under = poisson_cdf(piso, lam)
    p_over = 1.0 - p_under
    return round(p_over, 4), round(p_under, 4)


def prob_over_under_normal(linha: float, media: float, desvio_padrao: float = 11.5):
    if desvio_padrao <= 0:
        desvio_padrao = 10.0
    p_under = stats.norm.cdf(linha, loc=media, scale=desvio_padrao)
    p_over = 1.0 - p_under
    return round(float(p_over), 4), round(float(p_under), 4)


def calcular_delta_mercado(lam: float, linha: float):
    delta_abs = round(lam - linha, 3)
    delta_pct = round((delta_abs / linha) * 100, 2) if linha else None
    return delta_abs, delta_pct


# ============================================================
# ROBUSTEZ (confiança nos dados de entrada)
# ============================================================

AMOSTRA_MINIMA_JOGOS = 10
PENALIDADE_FATOR_ALTO = 0.25
PENALIDADE_FATOR_MEDIO = 0.10


def calcular_nivel_confianca_dados(tamanho_amostra: Optional[int] = None,
                                   fatores_incerteza: Optional[list] = None) -> float:
    """
    Nível de confiança (0 a 1) nos dados que sustentam a projeção:
    - confiança pela amostra: quanto mais jogos usados pra tirar a média, mais confiança.
      Se não informado, fica neutro (0.5) -- nem penaliza total nem assume confiança plena.
    - confiança pelo contexto: cada fator de incerteza (lesão, desfalque) de impacto alto/médio
      reduz a confiança -- fatores de impacto "low" não penalizam.
    """
    if tamanho_amostra is None:
        confianca_amostra = 0.5
    else:
        confianca_amostra = max(0.0, min(1.0, tamanho_amostra / AMOSTRA_MINIMA_JOGOS))

    confianca_contexto = 1.0
    for fator in (fatores_incerteza or []):
        impacto = (fator.get("impact_level") if isinstance(fator, dict) else None) or "low"
        if impacto == "high":
            confianca_contexto -= PENALIDADE_FATOR_ALTO
        elif impacto == "medium":
            confianca_contexto -= PENALIDADE_FATOR_MEDIO
    confianca_contexto = max(0.0, confianca_contexto)

    return round((confianca_amostra + confianca_contexto) / 2, 3)


def calcular_fator_robustez(nivel_confianca: float) -> float:
    """Robustez = min(1.0, 0.85 + 0.15 * nivel_confianca). Piso 0.85, teto 1.0."""
    nivel_confianca = max(0.0, min(1.0, nivel_confianca))
    return round(min(1.0, 0.85 + 0.15 * nivel_confianca), 4)


def calcular_probabilidade_real_ajustada(p_modelo: float, robustez: float) -> float:
    """
    Probabilidade real ajustada = probabilidade do modelo (Poisson/Normal) x Robustez.
    É um desconto de segurança sobre a confiança do modelo -- só se aplica ao lado
    da aposta que está sendo avaliado (não força p_over+p_under a somar 1, de propósito:
    é margem de segurança, não uma probabilidade "recalibrada").
    """
    if p_modelo is None:
        return None
    return round(max(0.0, min(1.0, p_modelo * robustez)), 4)


# ============================================================
# EV + KELLY FRACIONADO
# ============================================================

def calcular_ev(prob_real: float, odd_decimal: float):
    if prob_real is None or odd_decimal is None:
        return None
    return round((prob_real * odd_decimal) - 1, 4)


def kelly_fracionado(prob_real: float, odd_decimal: float, fracao=0.25, teto_unidades=2.5) -> Optional[float]:
    """
    Kelly fracionado em unidades (escala de referência: banca = 10u).
    SEM piso artificial -- um edge minúsculo gera stake minúscula, um edge forte
    gera stake maior (até o teto). Arredondado em degraus de 0.25u.
    """
    if prob_real is None or odd_decimal is None or odd_decimal <= 1:
        return None
    b = odd_decimal - 1
    p = prob_real
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0:
        return None

    unidades = f_star * fracao * 10.0
    unidades = min(teto_unidades, unidades)
    unidades_arredondadas = round(round(unidades * 4) / 4, 2)

    # só descarta se arredondar pra zero (edge existe mas é desprezível)
    if unidades_arredondadas <= 0:
        return None
    return unidades_arredondadas


# ============================================================
# MSC (Moneyball Score) -- ponderado por persona
# ============================================================

# Pesos por persona, sobre 3 componentes normalizados 0-1: EV, Delta, Robustez/Prob.
# Carlos (individual/correlações): prioriza EV e Delta -- aceita volatilidade
#   em troca da maior distorção de preço.
# Cris (coletivo/assimetrias): prioriza Probabilidade Real Ajustada e Robustez --
#   rejeita odd maior se a chance estatística cair.
PESOS_MSC = {
    "carlos": {"ev": 0.60, "delta": 0.25, "robustez_ou_prob": 0.15},
    "cris":   {"ev": 0.15, "delta": 0.25, "robustez_ou_prob": 0.60},
}

EV_TETO_NORMALIZACAO = 0.30   # EV de 30%+ já conta como "EV máximo" pra normalização
DELTA_TETO_NORMALIZACAO = 15.0  # delta_pct de 15%+ já conta como "delta máximo"


def calcular_msc(ev: Optional[float], delta_pct: Optional[float],
                 prob_real_ajustada: Optional[float], robustez: float,
                 persona: str = "carlos") -> Optional[int]:
    """
    MSC (Moneyball Score), 0-100, ponderado pela personalidade do analista.
    Nunca inventado pela LLM -- sempre calculado aqui a partir de números reais.
    """
    if ev is None or delta_pct is None or prob_real_ajustada is None:
        return None

    pesos = PESOS_MSC.get(persona.lower(), PESOS_MSC["carlos"])

    ev_norm = max(0.0, min(1.0, ev / EV_TETO_NORMALIZACAO))
    delta_norm = max(0.0, min(1.0, abs(delta_pct) / DELTA_TETO_NORMALIZACAO))

    if persona.lower() == "cris":
        componente_terciario = (prob_real_ajustada + robustez) / 2  # solidez: prob + robustez
    else:
        componente_terciario = robustez

    score = (
        pesos["ev"] * ev_norm +
        pesos["delta"] * delta_norm +
        pesos["robustez_ou_prob"] * componente_terciario
    )
    return round(max(0, min(100, score * 100)))


# ============================================================
# ESTIMATIVA DE LAMBDA (expectativa real a partir de médias do MDM)
# ============================================================

def estimar_lambda(mercado: dict) -> Optional[float]:
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


# ============================================================
# ROTEIRO DE JOGO (Metodologia Nexus Cap. V) -- classificação determinística
# ============================================================
# Mesma regra de ouro do resto deste arquivo: calculado aqui a partir de dados
# reais sempre que possível, nunca inventado pela LLM. Quando o esporte não tem
# dado de grounding suficiente (ver campos exigidos por função abaixo), a função
# retorna None -- nesse caso o MIE2 volta a classificar a hipótese de forma
# narrativa, sem sub-tipo, com confiança baixa (ver prompts_mie2.py).
#
# Os thresholds numéricos abaixo (ex: delta_xg >= 0.6, xg_combinado >= 2.6) são
# um ponto de partida razoável, NÃO calibrado empiricamente ainda -- precisam
# ser validados contra dados históricos antes de pesar decisões de stake.

CONFIANCA_ROTEIRO_GROUNDED = {
    "futebol": 0.75,
    "basquete": 0.70,
    "beisebol": 0.65,
    "nfl": 0.30,  # status experimental -- sem validação prática ainda
}


def classificar_roteiro_futebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Modelo territorial completo (5 arquétipos -- B1/B2/A1/A2/C1).
    Campos exigidos em cada dict: xg_medio, xg_sofrido_medio (posse_media é opcional,
    só refina B1 vs B2 quando presente)."""
    xg_a = dados_time_a.get("xg_medio")
    xg_b = dados_time_b.get("xg_medio")
    xg_sofrido_a = dados_time_a.get("xg_sofrido_medio")
    xg_sofrido_b = dados_time_b.get("xg_sofrido_medio")
    posse_a = dados_time_a.get("posse_media")
    posse_b = dados_time_b.get("posse_media")

    if None in (xg_a, xg_b, xg_sofrido_a, xg_sofrido_b):
        return None

    delta_xg = xg_a - xg_b
    xg_combinado = xg_a + xg_b
    evidencias = []
    dominante = None  # quem acaba classificado como lado forte, se houver (p/ instabilidade)

    # Passo 1: quem domina TERRITORIALMENTE (posse), quando esse dado existe --
    # é o sinal que decide entre B1 (domínio real) e B2 (domínio de posse vazio).
    dominante_posse = None
    if posse_a is not None and posse_b is not None:
        if posse_a >= 55:
            dominante_posse = "A"
        elif posse_b >= 55:
            dominante_posse = "B"

    if dominante_posse:
        dominante = dominante_posse
        xg_dom = xg_a if dominante_posse == "A" else xg_b
        xg_advers = xg_b if dominante_posse == "A" else xg_a
        posse_dom = posse_a if dominante_posse == "A" else posse_b

        if xg_dom - xg_advers >= 0.5:
            macro, sub = "TIPO B", "B1_dominio_total"
            evidencias.append(
                f"Time {dominante_posse} com posse média de {posse_dom}% e xG de {xg_dom}, "
                f"contra {xg_advers} do adversário -- posse e qualidade ofensiva convergem."
            )
        else:
            macro, sub = "TIPO B", "B2_contra_ataque_letal"
            evidencias.append(
                f"Time {dominante_posse} com posse média de {posse_dom}%, mas xG de {xg_dom} "
                f"próximo ou inferior ao xG do adversário ({xg_advers}) -- domínio territorial "
                f"sem tradução proporcional em qualidade ofensiva; risco de contra-ataque."
            )
    elif abs(delta_xg) >= 0.6:
        # Sem posse disponível pra confirmar/refutar, mas diferença de xG é grande --
        # assume domínio real (B1) por padrão, já que não há sinal de "posse vazia" pra checar.
        dominante = "A" if delta_xg > 0 else "B"
        macro, sub = "TIPO B", "B1_dominio_total"
        evidencias.append(f"Diferença de xG de {round(abs(delta_xg), 2)} a favor do time {dominante} (sem dado de posse disponível para refinar).")
    else:
        if xg_combinado >= 2.6:
            macro, sub = "TIPO A", "A1_jogo_aberto"
            evidencias.append(f"xG combinado de {round(xg_combinado, 2)} entre as duas equipes -- jogo com espaço para os dois lados.")
        else:
            macro, sub = "TIPO A", "A2_gato_e_rato"
            evidencias.append(f"xG combinado baixo ({round(xg_combinado, 2)}) -- jogo tende a ficar truncado, disputado no meio-campo.")

    instabilidade = None
    if xg_sofrido_a is not None and xg_sofrido_b is not None:
        # Proxy simples: quanto maior o xG sofrido do lado favorito, maior o risco
        # de o roteiro colapsar via transição/gol adversário.
        xg_sofrido_favorito = xg_sofrido_a if dominante == "A" else (xg_sofrido_b if dominante == "B" else max(xg_sofrido_a, xg_sofrido_b))
        instabilidade = round(min(1.0, max(0.0, xg_sofrido_favorito / 2.0)), 3)

    return {
        "macro": macro,
        "sub_tipo": sub,
        "confianca_classificacao": CONFIANCA_ROTEIRO_GROUNDED["futebol"],
        "evidencias": evidencias,
        "probabilidade_instabilidade_roteiro": instabilidade,
    }


def classificar_roteiro_basquete(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Modelo de pace + eficiência líquida (ORTG do ataque vs DRTG da defesa
    adversária). Sem B2/A2 -- não fazem sentido com posse constante em basquete.
    Campos exigidos: ortg, drtg (pace é opcional, só refina o sub-tipo A1)."""
    ortg_a, drtg_a = dados_time_a.get("ortg"), dados_time_a.get("drtg")
    ortg_b, drtg_b = dados_time_b.get("ortg"), dados_time_b.get("drtg")
    pace_a, pace_b = dados_time_a.get("pace"), dados_time_b.get("pace")

    if None in (ortg_a, drtg_a, ortg_b, drtg_b):
        return None

    # Eficiência líquida deste confronto específico: ataque de um lado contra a
    # defesa real do adversário (não a média geral da liga).
    net_a = ortg_a - drtg_b
    net_b = ortg_b - drtg_a
    delta_net = net_a - net_b

    evidencias = []
    pace_medio = round((pace_a + pace_b) / 2, 1) if pace_a is not None and pace_b is not None else None

    if abs(delta_net) >= 6.0:
        dominante = "A" if delta_net > 0 else "B"
        macro, sub = "TIPO B", "B1_dominio_por_eficiencia"
        evidencias.append(
            f"Diferencial de eficiência líquida de {round(abs(delta_net), 1)} pontos a favor "
            f"do time {dominante} neste confronto (ataque próprio vs. defesa real do adversário)."
        )
    else:
        macro = "TIPO A"
        if pace_medio is not None and pace_medio >= 100:
            sub = "A1_jogo_aberto_pace_alto"
            evidencias.append(f"Pace médio combinado de {pace_medio} posses, com eficiências equilibradas entre as equipes.")
        else:
            sub = None
            evidencias.append("Eficiências ofensiva/defensiva equilibradas entre as equipes, sem domínio líquido claro.")

    return {
        "macro": macro,
        "sub_tipo": sub,
        "confianca_classificacao": CONFIANCA_ROTEIRO_GROUNDED["basquete"],
        "evidencias": evidencias,
        # Sinal de banco curto / fadiga em back-to-back ainda não é buscado pelo
        # MIE1 -- fica null até esse campo existir.
        "probabilidade_instabilidade_roteiro": None,
    }


def classificar_roteiro_nfl(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Ataque vs. defesa via success rate. STATUS EXPERIMENTAL -- sem validação
    prática ainda, por isso não classifica sub_tipo e trava confiança no piso.
    Campos exigidos: success_rate_of, success_rate_def."""
    sr_a_of = dados_time_a.get("success_rate_of")
    sr_a_def = dados_time_a.get("success_rate_def")
    sr_b_of = dados_time_b.get("success_rate_of")
    sr_b_def = dados_time_b.get("success_rate_def")

    if None in (sr_a_of, sr_a_def, sr_b_of, sr_b_def):
        return None

    vantagem_a = sr_a_of - sr_b_def
    vantagem_b = sr_b_of - sr_a_def
    delta = vantagem_a - vantagem_b

    if abs(delta) >= 0.08:
        dominante = "A" if delta > 0 else "B"
        macro = "TIPO B"
        evidencia = f"Vantagem de success rate ofensivo do time {dominante} sobre a defesa adversária neste confronto específico."
    else:
        macro = "TIPO A"
        evidencia = "Matchups de ataque vs. defesa equilibrados entre as duas equipes."

    return {
        "macro": macro,
        "sub_tipo": None,  # experimental -- sem sub-tipo até validação empírica
        "confianca_classificacao": CONFIANCA_ROTEIRO_GROUNDED["nfl"],
        "evidencias": [evidencia],
        "probabilidade_instabilidade_roteiro": None,
    }


def classificar_roteiro_beisebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Beisebol não é territorial -- é uma sequência de duelos individuais.
    Por isso o macro default é TIPO C (arremessador titular vs. lineup adversário),
    exceto quando os dois duelos do jogo favorecem claramente o MESMO lado (aí vira
    TIPO B -- domínio geral, não só individual). Bullpen vira o sinal de instabilidade
    (relevante para props de innings finais).
    Campos exigidos: pitcher_era, lineup_ops (bullpen_era é opcional)."""
    era_a = dados_time_a.get("pitcher_era")
    era_b = dados_time_b.get("pitcher_era")
    ops_a = dados_time_a.get("lineup_ops")
    ops_b = dados_time_b.get("lineup_ops")
    bullpen_a = dados_time_a.get("bullpen_era")
    bullpen_b = dados_time_b.get("bullpen_era")

    if None in (era_a, era_b, ops_a, ops_b):
        return None

    evidencias = [
        f"Arremessador titular do time A (ERA {era_a}) contra lineup adversário (OPS {ops_b}).",
        f"Arremessador titular do time B (ERA {era_b}) contra lineup adversário (OPS {ops_a}).",
    ]

    # Thresholds de referência (aprox. média de liga MLB) -- calibrar depois com histórico.
    duelo_a_favoravel = era_a <= 3.80 and ops_b <= 0.720
    duelo_b_favoravel = era_b <= 3.80 and ops_a <= 0.720

    if duelo_a_favoravel and not duelo_b_favoravel:
        macro, sub = "TIPO B", None
        evidencias.append("Duelo pitcher x lineup favorece claramente o time A dos dois lados do jogo -- sinal de domínio geral, não só individual.")
    elif duelo_b_favoravel and not duelo_a_favoravel:
        macro, sub = "TIPO B", None
        evidencias.append("Duelo pitcher x lineup favorece claramente o time B dos dois lados do jogo -- sinal de domínio geral, não só individual.")
    else:
        macro, sub = "TIPO C", "C1_duelo_pitcher_lineup"

    instabilidade = None
    if bullpen_a is not None and bullpen_b is not None:
        pior_bullpen = max(bullpen_a, bullpen_b)
        instabilidade = round(min(1.0, max(0.0, (pior_bullpen - 3.50) / 3.0)), 3)

    return {
        "macro": macro,
        "sub_tipo": sub,
        "confianca_classificacao": CONFIANCA_ROTEIRO_GROUNDED["beisebol"],
        "evidencias": evidencias,
        "probabilidade_instabilidade_roteiro": instabilidade,
    }


_CLASSIFICADORES_ROTEIRO = {
    "futebol": classificar_roteiro_futebol,
    "basquete": classificar_roteiro_basquete,
    "nfl": classificar_roteiro_nfl,
    "beisebol": classificar_roteiro_beisebol,
}


def classificar_roteiro_jogo(esporte: str, dados_time_a: Optional[dict], dados_time_b: Optional[dict]) -> Optional[dict]:
    """
    Classificador determinístico de roteiro de jogo (Metodologia Nexus, Cap. V).
    Retorna None se faltar dado de grounding suficiente pro esporte -- nesse caso
    o MIE2 classifica hipotese_partida de forma narrativa, como já fazia antes,
    sem sub_tipo e com confiança baixa.
    """
    if not dados_time_a or not dados_time_b:
        return None

    fn = _CLASSIFICADORES_ROTEIRO.get(esporte.lower())
    if not fn:
        return None

    try:
        return fn(dados_time_a, dados_time_b)
    except Exception:
        return None


# ============================================================
# CÁLCULO POR MERCADO ISOLADO (usado pelo endpoint utilitário /api/v1/calc)
# ============================================================

def calcular_mercado(mercado: dict, esporte: str = "futebol") -> dict:
    linha = mercado.get("linha")
    if linha is None:
        return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}

    esporte_key = esporte.lower()

    if esporte_key in ("basquete", "nfl") and mercado.get("modelo") != "poisson":
        media_esperada = mercado.get("media_esperada") or estimar_lambda(mercado)
        if media_esperada is None:
            return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}
        std_dev = mercado.get("desvio_padrao", 12.0 if esporte_key == "basquete" else 18.5)
        p_over, p_under = prob_over_under_normal(linha, media_esperada, std_dev)
        lam_ref = media_esperada
    else:
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
    """Usado só pelo endpoint utilitário /api/v1/calc -- devolve uma LISTA
    de resultados por mercado, sem seleção de dupla de elite (isso é
    responsabilidade do prompt_mie2 + Groq no fluxo real /api/v1/analyze)."""
    resultados = []
    for m in mercados:
        try:
            resultados.append(calcular_mercado(m, esporte=esporte))
        except Exception as e:
            resultados.append({"id": m.get("id"), "status": "erro_calculo", "detalhe": str(e)})
    return resultados
