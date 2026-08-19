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
    calcular_probabilidades_1x2_skellam,
    calcular_probabilidade_vitoria_2vias,
    calcular_probabilidade_handicap_asiatico,
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

        if esporte_key == "basquete" and "escanteios" not in nome_mercado.lower() and "cartoes" not in nome_mercado.lower():
            std_dev = 12.0
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


def montar_candidato_moneyline(mercado_moneyline: Optional[dict], lam_a: Optional[float], lam_b: Optional[float],
                                esporte: str, nome_time_a: str = "Time A", nome_time_b: str = "Time B",
                                persona: str = "carlos", fatores_incerteza: Optional[list] = None) -> list:
    """Moneyline (2 vias, sem empate) -- beisebol e basquete. `mercado_moneyline`
    deve trazer "odd_time_a"/"odd_time_b" (as odds reais extraídas do print).
    NÃO usar pra futebol -- lá o empate é resultado real, use
    montar_candidatos_chance_dupla."""
    if not mercado_moneyline or lam_a is None or lam_b is None:
        return []

    modelo = "normal" if esporte.lower() == "basquete" else "skellam"
    p_a, p_b = calcular_probabilidade_vitoria_2vias(lam_a, lam_b, modelo=modelo)

    candidatos = []
    odd_a = mercado_moneyline.get("odd_time_a")
    odd_b = mercado_moneyline.get("odd_time_b")

    if odd_a:
        metricas = _montar_metricas_candidato(p_a, odd_a, persona, fatores_incerteza, delta_pct=None)
        candidatos.append({
            "mercado": "Moneyline (Vencedor)", "selecao": nome_time_a, "odd": odd_a,
            "probabilidade_real_calculada": p_a, **metricas,
        })
    if odd_b:
        metricas = _montar_metricas_candidato(p_b, odd_b, persona, fatores_incerteza, delta_pct=None)
        candidatos.append({
            "mercado": "Moneyline (Vencedor)", "selecao": nome_time_b, "odd": odd_b,
            "probabilidade_real_calculada": p_b, **metricas,
        })
    return candidatos


def montar_candidatos_chance_dupla(mercado_chance_dupla: Optional[dict], lam_a: Optional[float], lam_b: Optional[float],
                                    persona: str = "carlos", fatores_incerteza: Optional[list] = None) -> list:
    """Chance Dupla (1X / X2 / 12) -- futebol. `mercado_chance_dupla` deve trazer
    "odd_1x"/"odd_x2"/"odd_12" (só as que existirem no print -- nem todo ticket
    mostra as três)."""
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
            metricas = _montar_metricas_candidato(prob, odd, persona, fatores_incerteza, delta_pct=None)
            candidatos.append({
                "mercado": "Chance Dupla", "selecao": nome_selecao, "odd": odd,
                "probabilidade_real_calculada": prob, **metricas,
            })
    return candidatos


def montar_candidatos_handicap_asiatico(mercados_handicap: Optional[list], lam_a: Optional[float], lam_b: Optional[float],
                                         persona: str = "carlos", fatores_incerteza: Optional[list] = None) -> list:
    """Handicap Asiático -- futebol. Cada item de `mercados_handicap` deve
    trazer "linha" (o handicap), "time_referencia" ("A" ou "B" -- de qual time
    é esse handicap) e "odd_real_decimal". Linhas de quarto de gol (.25/.75)
    são tratadas automaticamente via split -- ver calc.py."""
    if not mercados_handicap or lam_a is None or lam_b is None:
        return []

    candidatos = []
    for mercado in mercados_handicap:
        linha = mercado.get("linha")
        odd = mercado.get("odd_real_decimal")
        time_ref = mercado.get("time_referencia", "A")
        if linha is None or not odd:
            continue

        if time_ref == "B":
            # Simetria: handicap do time B é o espelho do handicap do time A com
            # sinal invertido -- inverte lam_a/lam_b em vez de duplicar a lógica.
            p_cobre, p_push = calcular_probabilidade_handicap_asiatico(lam_b, lam_a, -linha)
        else:
            p_cobre, p_push = calcular_probabilidade_handicap_asiatico(lam_a, lam_b, linha)

        metricas = _montar_metricas_candidato(p_cobre, odd, persona, fatores_incerteza, delta_pct=None)
        candidatos.append({
            "mercado": "Handicap Asiático",
            "selecao": mercado.get("selecao_texto") or f"Time {time_ref} ({linha:+g})",
            "odd": odd,
            "probabilidade_real_calculada": p_cobre,
            "probabilidade_push": p_push,
            **metricas,
        })
    return candidatos
