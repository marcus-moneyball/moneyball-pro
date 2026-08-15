"""
MIE2 - Camada de cálculo determinístico (Delta + Poisson + Kelly Fracionado)
Sem dependências externas (só math da stdlib) -> roda em qualquer host Python simples.

Princípio: cada mercado é calculado isoladamente. Falta de dado em UM mercado
nunca derruba os outros. Nunca inventa número que não veio no MDM/odd real.
"""

import math


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def prob_over_under(linha: float, lam: float):
    """Ex: linha=7.5 -> Under 7.5 = P(X<=7), Over 7.5 = 1 - P(X<=7)."""
    piso = math.floor(linha)
    p_under = poisson_cdf(piso, lam)
    p_over = 1 - p_under
    return round(p_over, 4), round(p_under, 4)


def calcular_delta(lam: float, linha: float):
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
    """
    Constrói o lambda (expectativa real) a partir das médias reais do MDM.
    Espera SEMPRE as 4 taxas base de cada time (marcada/sofrida), porque
    'Total da Partida' precisa somar os dois lados, e 'Total do Time' precisa
    isolar só um lado -- usar a mesma fonte de dado pros dois evita duplicar
    lógica e evita o bug de "total da partida" usando só um time.

    tipo: "total_jogo" | "total_time_a" | "total_time_b"
    """
    tipo = mercado.get("tipo", "total_jogo")
    marcada_a = mercado.get("media_marcada_time_a")
    sofrida_a = mercado.get("media_sofrida_time_a")
    marcada_b = mercado.get("media_marcada_time_b")
    sofrida_b = mercado.get("media_sofrida_time_b")

    if None in (marcada_a, sofrida_a, marcada_b, sofrida_b):
        return None

    esperado_a = (marcada_a + sofrida_b) / 2  # ataque do A vs defesa do B
    esperado_b = (marcada_b + sofrida_a) / 2  # ataque do B vs defesa do A

    if tipo == "total_time_a":
        return round(esperado_a, 3)
    if tipo == "total_time_b":
        return round(esperado_b, 3)
    return round(esperado_a + esperado_b, 3)  # total_jogo (padrão)


def calcular_mercado(mercado: dict) -> dict:
    """
    Espera um dict por mercado, vindo do dossiê MIE1 + odd real colada/raspada:
    {
        "id": "total_runs_jogo",
        "tipo": "total_jogo",                 # ou "total_time_a" / "total_time_b"
        "linha": 7.5,
        "odd_real_decimal": 1.87,             # opcional
        "lado_odd": "over" | "under",
        "media_marcada_time_a": 4.6,
        "media_sofrida_time_a": 4.5,
        "media_marcada_time_b": 5.1,
        "media_sofrida_time_b": 4.7,
    }
    Se faltar dado pra montar lambda, retorna status "sem_dados_suficientes"
    e não trava o resto -- esse mercado só sai da lista de elegíveis.
    """
    linha = mercado.get("linha")
    if linha is None:
        return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}

    lam = estimar_lambda(mercado)
    if lam is None:
        return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}

    odd = mercado.get("odd_real_decimal")

    p_over, p_under = prob_over_under(linha, lam)
    delta_abs, delta_pct = calcular_delta(lam, linha)

    resultado = {
        "id": mercado.get("id"),
        "status": "calculado",
        "lambda_estimado": lam,
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


def calcular_dossie(mercados: list) -> list:
    resultados = []
    for m in mercados:
        try:
            resultados.append(calcular_mercado(m))
        except Exception as e:
            resultados.append({"id": m.get("id"), "status": "erro_calculo", "detalhe": str(e)})
    return resultados
