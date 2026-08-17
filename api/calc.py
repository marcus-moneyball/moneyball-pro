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
