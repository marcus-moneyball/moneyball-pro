"""
Monta os candidatos de aposta a partir do lambda (esperado_partida) calculado
pelo Python e das linhas/odds reais extraídas dos prints. Essa é a ponte entre
a camada de cálculo (calc.py) e o texto que vai pro MIE2 (Groq).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
from calc import (
    prob_over_under_normal,
    prob_over_under_poisson,
    poisson_pmf,
    calcular_ev,
    kelly_fracionado,
    calcular_nivel_confianca_dados,
    calcular_fator_robustez,
    calcular_probabilidade_real_ajustada,
    calcular_msc,
)


def _montar_metricas_candidato(prob_bruta: float, odd: float, persona: str,
                                fatores_incerteza: Optional[list], delta_pct: Optional[float]):
    """Calcula robustez, prob ajustada, EV, Kelly e MSC pra um lado específico
    (over ou under) de um mercado -- reaproveitado pelos dois construtores abaixo."""
    nivel_confianca = calcular_nivel_confianca_dados(fatores_incerteza=fatores_incerteza)
    robustez = calcular_fator_robustez(nivel_confianca)
    prob_ajustada = calcular_probabilidade_real_ajustada(prob_bruta, robustez)

    prob_implicita_odd = round(1 / odd, 4) if odd else None
    edge_pct = round((prob_ajustada - prob_implicita_odd) * 100, 2) if prob_implicita_odd is not None else None
    ev = calcular_ev(prob_ajustada, odd)
    kelly = kelly_fracionado(prob_ajustada, odd) if ev is not None and ev > 0 else None
    # Mercados sem linha numérica (ex: BTTS) não têm delta_pct -- usa o edge
    # percentual (prob ajustada vs prob implícita da odd) como sinal equivalente.
    sinal_distorcao = delta_pct if delta_pct is not None else edge_pct
    msc = calcular_msc(ev, sinal_distorcao, prob_ajustada, robustez, persona=persona) if ev is not None else None

    return {
        "robustez": robustez,
        "probabilidade_real_ajustada": prob_ajustada,
        "probabilidade_implicita_odd": prob_implicita_odd,
        "delta_edge_pct_calculado": edge_pct,
        "ev": ev,
        "kelly_unidades_sugerido": kelly,
        "msc_calculado": msc,
    }


def montar_candidatos_over_under_calculados(
    mercados: list, lam_total: Optional[float], nome_mercado: str, unidade_selecao: str,
    esporte: str = "futebol", persona: str = "carlos", fatores_incerteza: Optional[list] = None
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

        prob_bruta = p_under if lado == "under" else p_over
        delta_abs = round(lam_total - linha, 3)
        delta_pct = round((delta_abs / linha) * 100, 2) if linha else None

        metricas = _montar_metricas_candidato(prob_bruta, odd, persona, fatores_incerteza, delta_pct)

        candidatos.append({
            "mercado": nome_mercado,
            "selecao": f"{'Mais' if lado == 'over' else 'Menos'} de {linha} {unidade_selecao}",
            "odd": odd,
            "esperado_partida": lam_total,
            "probabilidade_real_calculada": prob_bruta,
            **metricas,
        })

    return candidatos


def montar_candidato_btts(mercado_btts: Optional[dict], lam_a: Optional[float], lam_b: Optional[float],
                           persona: str = "carlos", fatores_incerteza: Optional[list] = None) -> list:
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
        metricas = _montar_metricas_candidato(p_sim, odd_sim, persona, fatores_incerteza, delta_pct=None)
        candidatos.append({
            "mercado": "Ambos Marcam (BTTS)",
            "selecao": "Sim",
            "odd": odd_sim,
            "probabilidade_real_calculada": p_sim,
            **metricas,
        })

    if odd_nao:
        metricas = _montar_metricas_candidato(p_nao, odd_nao, persona, fatores_incerteza, delta_pct=None)
        candidatos.append({
            "mercado": "Ambos Marcam (BTTS)",
            "selecao": "Não",
            "odd": odd_nao,
            "probabilidade_real_calculada": p_nao,
            **metricas,
        })

    return candidatos
