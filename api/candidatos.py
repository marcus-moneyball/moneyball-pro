"""
Monta os candidatos de aposta a partir do lambda (esperado_partida) calculado
pelo Python e das linhas/odds reais extraídas dos prints. Essa é a ponte entre
a camada de cálculo (calc.py) e o texto que vai pro MIE2 (Groq).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Tuple, Dict, Any
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
    calcular_probabilidades_1x2_skellam,
    calcular_probabilidade_vitoria_2vias,
    calcular_probabilidade_handicap_asiatico,
)
from utils import converter_odd_para_decimal

EDGE_MAXIMO_PLAUSIVEL_PCT = 25.0
STD_DEV_BASQUETE_DEFAULT = 12.0


def _montar_metricas_candidato(
    prob_bruta: float, 
    odd, 
    persona: str,
    fatores_incerteza: Optional[list], 
    delta_pct: Optional[float],
    contexto_log: Optional[str] = None
) -> Tuple[Optional[float], Dict[str, Any]]:
    odd_decimal = converter_odd_para_decimal(odd)

    if odd_decimal is None:
        return None, {}

    nivel_confianca = calcular_nivel_confianca_dados(fatores_incerteza=fatores_incerteza)
    robustez = calcular_fator_robustez(nivel_confianca)
    prob_ajustada = calcular_probabilidade_real_ajustada(prob_bruta, robustez)

    prob_implicita_odd = round(1 / odd_decimal, 4)
    edge_pct = round((prob_ajustada - prob_implicita_odd) * 100, 2)

    if edge_pct > EDGE_MAXIMO_PLAUSIVEL_PCT:
        print(
            f"[SANIDADE] Candidato descartado -- edge {edge_pct}% acima do "
            f"teto ({EDGE_MAXIMO_PLAUSIVEL_PCT}%). odd={odd_decimal} "
            f"prob_ajustada={prob_ajustada} contexto={contexto_log or 'n/d'}"
        )
        return None, {}

    ev = calcular_ev(prob_ajustada, odd_decimal)
    kelly = kelly_fracionado(prob_ajustada, odd_decimal) if ev is not None and ev > 0 else None
    sinal_distorcao = delta_pct if delta_pct is not None else edge_pct
    msc = calcular_msc(ev, sinal_distorcao, prob_ajustada, robustez, persona=persona) if ev is not None else None

    return odd_decimal, {
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

        # Correção: Removida a Binomial Negativa desalinhada; beisebol e futebol voltam a usar Poisson padronizado
        if esporte_key == "basquete" and "escanteios" not in nome_mercado.lower() and "cartoes" not in nome_mercado.lower():
            p_over, p_under = prob_over_under_normal(linha, lam_total, STD_DEV_BASQUETE_DEFAULT)
        else:
            p_over, p_under = prob_over_under_poisson(linha, lam_total)

        prob_bruta = p_under if lado == "under" else p_over
        delta_abs = round(lam_total - linha, 3)
        delta_pct = round((delta_abs / linha) * 100, 2) if linha and linha != 0 else None

        ctx_log = f"{esporte_key}/{nome_mercado} - {lado} {linha}"
        odd_decimal, metricas = _montar_metricas_candidato(
            prob_bruta, odd, persona, fatores_incerteza, delta_pct, contexto_log=ctx_log
        )
        if odd_decimal is None:
            continue

        candidatos.append({
            "mercado": nome_mercado,
            "selecao": f"{'Mais' if lado == 'over' else 'Menos'} de {linha} {unidade_selecao}",
            "odd": odd_decimal,
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
    opcoes = [
        ("Sim", mercado_btts.get("odd_sim"), p_sim),
        ("Não", mercado_btts.get("odd_nao"), p_nao),
    ]

    for selecao, odd, prob in opcoes:
        if odd:
            ctx_log = f"futebol/BTTS - {selecao}"
            odd_decimal, metricas = _montar_metricas_candidato(
                prob, odd, persona, fatores_incerteza, delta_pct=None, contexto_log=ctx_log
            )
            if odd_decimal is not None:
                candidatos.append({
                    "mercado": "Ambos Marcam (BTTS)",
                    "selecao": selecao,
                    "odd": odd_decimal,
                    "probabilidade_real_calculada": prob,
                    **metricas,
                })

    return candidatos


def montar_candidato_moneyline(mercado_moneyline: Optional[dict], lam_a: Optional[float], lam_b: Optional[float],
                                esporte: str, nome_time_a: str = "Time A", nome_time_b: str = "Time B",
                                persona: str = "carlos", fatores_incerteza: Optional[list] = None) -> list:
    if not mercado_moneyline or lam_a is None or lam_b is None:
        return []

    modelo = "normal" if esporte.lower() == "basquete" else "skellam"
    p_a, p_b = calcular_probabilidade_vitoria_2vias(lam_a, lam_b, modelo=modelo)

    candidatos = []
    opcoes = [
        (nome_time_a, mercado_moneyline.get("odd_time_a"), p_a),
        (nome_time_b, mercado_moneyline.get("odd_time_b"), p_b),
    ]

    for selecao, odd, prob in opcoes:
        if odd:
            ctx_log = f"{esporte}/Moneyline - {selecao}"
            odd_decimal, metricas = _montar_metricas_candidato(
                prob, odd, persona, fatores_incerteza, delta_pct=None, contexto_log=ctx_log
            )
            if odd_decimal is not None:
                candidatos.append({
                    "mercado": "Moneyline (Vencedor)",
                    "selecao": selecao,
                    "odd": odd_decimal,
                    "probabilidade_real_calculada": prob,
                    **metricas,
                })

    return candidatos


def montar_candidatos_chance_dupla(mercado_chance_dupla: Optional[dict], lam_a: Optional[float], lam_b: Optional[float],
                                    persona: str = "carlos", fatores_incerteza: Optional[list] = None) -> list:
    if not mercado_chance_dupla or lam_a is None or lam_b is None:
        return []

    p_a, p_empate, p_b = calcular_probabilidades_1x2_skellam(lam_a, lam_b)
    mapa = [
        ("odd_1x", round(p_a + p_empate, 4), "1X (Casa ou Empate)"),
        ("odd_x2", round(p_empate + p_b, 4), "X2 (Empate ou Fora)"),
        ("odd_12", round(p_a + p_b, 4), "12 (Casa ou Fora -- sem Empate)"),
    ]

    candidatos = []
    for campo_odd, prob, nome_selecao in mapa:
        odd = mercado_chance_dupla.get(campo_odd)
        if odd:
            ctx_log = f"futebol/Chance Dupla - {nome_selecao}"
            odd_decimal, metricas = _montar_metricas_candidato(
                prob, odd, persona, fatores_incerteza, delta_pct=None, contexto_log=ctx_log
            )
            if odd_decimal is not None:
                candidatos.append({
                    "mercado": "Chance Dupla",
                    "selecao": nome_selecao,
                    "odd": odd_decimal,
                    "probabilidade_real_calculada": prob,
                    **metricas,
                })
    return candidatos


def montar_candidatos_handicap_asiatico(mercados_handicap: Optional[list], lam_a: Optional[float], lam_b: Optional[float],
                                        persona: str = "carlos", fatores_incerteza: Optional[list] = None) -> list:
    if not mercados_handicap or lam_a is None or lam_b is None:
        return []

    candidatos = []
    for mercado in mercados_handicap:
        linha = mercado.get("linha")
        odd = mercado.get("odd") or mercado.get("odd_real_decimal")
        time_ref = mercado.get("time_referencia", "A")
        if linha is None or not odd:
            continue

        if time_ref == "B":
            p_cobre, p_push = calcular_probabilidade_handicap_asiatico(lam_b, lam_a, -linha)
        else:
            p_cobre, p_push = calcular_probabilidade_handicap_asiatico(lam_a, lam_b, linha)

        ctx_log = f"futebol/Handicap Asiatico - Time {time_ref} ({linha:+g})"
        odd_decimal, metricas = _montar_metricas_candidato(
            prob_bruta=p_cobre, 
            odd=odd, 
            persona=persona, 
            fatores_incerteza=fatores_incerteza, 
            delta_pct=None,
            contexto_log=ctx_log
        )
        if odd_decimal is None:
            continue

        candidatos.append({
            "mercado": "Handicap Asiático",
            "selecao": mercado.get("selecao_texto") or f"Time {time_ref} ({linha:+g})",
            "odd": odd_decimal,
            "probabilidade_real_calculada": p_cobre,
            "probabilidade_push": p_push,
            **metricas,
        })
    return candidatos
