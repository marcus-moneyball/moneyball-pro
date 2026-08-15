"""
Monta os candidatos de aposta a partir do lambda (esperado_partida) calculado
pelo Python e das linhas/odds reais extraídas dos prints. Essa é a ponte entre
a camada de cálculo (calc.py) e o texto que vai pro MIE2 (Groq).
"""

from typing import Optional
from calc import (
    prob_over_under_normal,
    prob_over_under_poisson,
    poisson_pmf,
    calcular_ev,
    kelly_fracionado,
)


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
