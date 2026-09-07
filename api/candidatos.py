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
from odds_sharp import buscar_odds_sharp

EDGE_MAXIMO_PLAUSIVEL_PCT = 25.0
STD_DEV_BASQUETE_DEFAULT = 12.0


def _ajustar_msc_por_sharp(
    msc: Optional[float], prob_ajustada: float, conn, esporte: Optional[str],
    time_a: Optional[str], time_b: Optional[str], data_jogo: Optional[str],
    mercado_sharp: Optional[str], indice_selecao: Optional[int],
    fair_prob_precalculado: Optional[float] = None,
) -> Optional[float]:
    """
    Ajusta o MSC comparando a probabilidade do modelo com o fair odds (sem
    vig) de uma fonte sharp externa. Segue o mesmo espírito da Convergência:
    bônus se confirma, NEUTRO se não tem dado (nunca penaliza ausência),
    penalidade só se discordar de verdade.

    Se `fair_prob_precalculado` for passado, usa direto (caso de mercados
    combinados como Chance Dupla, onde o fair prob já vem somado de fora).
    Caso contrário, busca no cache pelo índice de seleção (mercados 2 vias
    simples, como Moneyline).
    """
    if msc is None:
        return msc

    if fair_prob_precalculado is not None:
        fair_prob_sharp = fair_prob_precalculado
    else:
        if conn is None or not mercado_sharp or indice_selecao is None:
            return msc
        info = buscar_odds_sharp(conn, esporte or "", time_a or "", time_b or "", data_jogo or "", mercado_sharp)
        if not info or not info.get("fair_probs") or indice_selecao >= len(info["fair_probs"]):
            return msc  # neutro -- sharp não cobre esse jogo/mercado ainda
        fair_prob_sharp = info["fair_probs"][indice_selecao]

    diferenca = prob_ajustada - fair_prob_sharp
    if diferenca > 0.01:
        return min(100, round(msc + 8, 2))
    if diferenca < -0.03:
        return max(0, round(msc - 20, 2))
    return msc


def _montar_metricas_candidato(
    prob_bruta: float, 
    odd, 
    persona: str,
    fatores_incerteza: Optional[list], 
    delta_pct: Optional[float],
    contexto_log: Optional[str] = None,
    conn=None,
    esporte: Optional[str] = None,
    time_a: Optional[str] = None,
    time_b: Optional[str] = None,
    data_jogo: Optional[str] = None,
    mercado_sharp: Optional[str] = None,
    indice_selecao: Optional[int] = None,
    fair_prob_sharp_precalculado: Optional[float] = None,
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
    msc = _ajustar_msc_por_sharp(
        msc, prob_ajustada, conn, esporte, time_a, time_b, data_jogo, mercado_sharp, indice_selecao,
        fair_prob_precalculado=fair_prob_sharp_precalculado,
    )

    # --- TRAVA DE EMERGÊNCIA: FILTRO RIGOROSO DE MSC ---
    MSC_MINIMO_EXIGIDO = 80.0  # Sobe o sarrafo para aceitar apenas entradas de alta confiança
    if msc is None or msc < MSC_MINIMO_EXIGIDO:
        print(
            f"[MSC CIRCUIT BREAKER] Candidato descartado por baixa confiança -- "
            f"MSC gerado: {msc} (Mínimo exigido: {MSC_MINIMO_EXIGIDO}). "
            f"contexto={contexto_log or 'n/d'}"
        )
        return None, {}
    # --------------------------------------------------

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
                                persona: str = "carlos", fatores_incerteza: Optional[list] = None,
                                conn=None, data_jogo: Optional[str] = None) -> list:
    if not mercado_moneyline or lam_a is None or lam_b is None:
        return []

    modelo = "normal" if esporte.lower() == "basquete" else "skellam"
    p_a, p_b = calcular_probabilidade_vitoria_2vias(lam_a, lam_b, modelo=modelo)

    candidatos = []
    opcoes = [
        (nome_time_a, mercado_moneyline.get("odd_time_a"), p_a, 0),
        (nome_time_b, mercado_moneyline.get("odd_time_b"), p_b, 1),
    ]

    for selecao, odd, prob, indice in opcoes:
        if odd:
            ctx_log = f"{esporte}/Moneyline - {selecao}"
            odd_decimal, metricas = _montar_metricas_candidato(
                prob, odd, persona, fatores_incerteza, delta_pct=None, contexto_log=ctx_log,
                conn=conn, esporte=esporte, time_a=nome_time_a, time_b=nome_time_b,
                data_jogo=data_jogo, mercado_sharp="moneyline", indice_selecao=indice,
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
                                    persona: str = "carlos", fatores_incerteza: Optional[list] = None,
                                    conn=None, time_a: Optional[str] = None, time_b: Optional[str] = None,
                                    data_jogo: Optional[str] = None) -> list:
    if not mercado_chance_dupla or lam_a is None or lam_b is None:
        return []

    p_a, p_empate, p_b = calcular_probabilidades_1x2_skellam(lam_a, lam_b)

    # Uma única consulta ao cache sharp -- serve pras três combinações,
    # em vez de bater no banco três vezes pro mesmo jogo.
    fair_1x2 = None
    if conn is not None:
        info_sharp = buscar_odds_sharp(conn, "futebol", time_a or "", time_b or "", data_jogo or "", "1x2")
        if info_sharp and info_sharp.get("fair_probs") and len(info_sharp["fair_probs"]) == 3:
            fair_1x2 = info_sharp["fair_probs"]  # [fair_casa, fair_empate, fair_fora]

    mapa = [
        ("odd_1x", round(p_a + p_empate, 4), "1X (Casa ou Empate)",
         (fair_1x2[0] + fair_1x2[1]) if fair_1x2 else None),
        ("odd_x2", round(p_empate + p_b, 4), "X2 (Empate ou Fora)",
         (fair_1x2[1] + fair_1x2[2]) if fair_1x2 else None),
        ("odd_12", round(p_a + p_b, 4), "12 (Casa ou Fora -- sem Empate)",
         (fair_1x2[0] + fair_1x2[2]) if fair_1x2 else None),
    ]

    candidatos = []
    for campo_odd, prob, nome_selecao, fair_combo in mapa:
        odd = mercado_chance_dupla.get(campo_odd)
        if odd:
            ctx_log = f"futebol/Chance Dupla - {nome_selecao}"
            odd_decimal, metricas = _montar_metricas_candidato(
                prob, odd, persona, fatores_incerteza, delta_pct=None, contexto_log=ctx_log,
                fair_prob_sharp_precalculado=fair_combo,
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
            p_cobre, p_push = calcular_probabilidade_handicap_asiatico(lam_b, lam_a, linha)
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
