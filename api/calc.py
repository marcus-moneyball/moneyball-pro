"""
Camada de cálculo determinístico multi-esporte (Delta + Poisson + Normal + Kelly).
Integrado com Análise de Assimetrias (Cris) e Matriz de Correlações (Carlos).
Sem chamadas de rede — 100% testável isoladamente.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
from typing import Optional, List, Dict, Any
import scipy.stats as stats


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


def calcular_fator_robustez(confianca_dados: float = 1.0) -> float:
    """
    Calcula o Fator de Robustez baseado na consistência dos dados de entrada.
    """
    confianca_dados = max(0.0, min(1.0, confianca_dados))
    return min(1.0, 0.85 + 0.15 * confianca_dados)


def calcular_probabilidade_real_ajustada(p_modelo: float, confianca_dados: float = 1.0) -> float:
    """
    Calcula a Probabilidade Real Ajustada aplicando o fator de robustez.
    """
    robustez = calcular_fator_robustez(confianca_dados)
    return round(max(0.01, min(0.99, p_modelo * robustez)), 4)


def calcular_ev(prob_real: float, odd_decimal: float):
    if prob_real is None or odd_decimal is None:
        return None
    return round((prob_real * odd_decimal) - 1, 4)


def kelly_fracionado(prob_real: float, odd_decimal: float, fracao=0.25, teto_unidades=2.5) -> Optional[float]:
    """
    Calcula o Critério de Kelly Fracionado em unidades (ex: 1.5).
    """
    if prob_real is None or odd_decimal is None or odd_decimal <= 1:
        return None
    b = odd_decimal - 1
    p = prob_real
    q = 1 - p
    f_star = (b * p - q) / b
    if f_star <= 0:
        return None
    
    kelly_frac = f_star * fracao
    unidades_calculadas = kelly_frac * 20.0 
    
    stake_final = min(teto_unidades, max(0.5, unidades_calculadas))
    stake_arredondada = round(round(stake_final * 4) / 4, 2)
    
    return float(stake_arredondada)


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
        p_over_bruto, p_under_bruto = prob_over_under_normal(linha, media_esperada, std_dev)
        lam_ref = media_esperada
    else:
        lam_ref = estimar_lambda(mercado) if mercado.get("media_esperada") is None else mercado.get("media_esperada")
        if lam_ref is None:
            return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}
        p_over_bruto, p_under_bruto = prob_over_under_poisson(linha, lam_ref)

    confianca = mercado.get("confianca_dados", 1.0)
    p_over = calcular_probabilidade_real_ajustada(p_over_bruto, confianca)
    p_under = calcular_probabilidade_real_ajustada(p_under_bruto, confianca)

    odd = mercado.get("odd_real_decimal")
    delta_abs, delta_pct = calcular_delta_mercado(lam_ref, linha)

    resultado = {
        "id": mercado.get("id"),
        "status": "calculado",
        "mercado_nome": mercado.get("nome", "Linha Geral"),
        "esperado_estimado": lam_ref,
        "probabilidade_over": p_over,
        "probabilidade_under": p_under,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "ev": None,
        "kelly_unidades": None,
        "odd_real_decimal": odd,
        "lado_odd": mercado.get("lado_odd", "over")
    }

    if odd is not None:
        lado = mercado.get("lado_odd", "over")
        prob_desse_lado = p_over if lado == "over" else p_under
        ev = calcular_ev(prob_desse_lado, odd)
        resultado["ev"] = ev
        if ev is not None and ev > 0:
            resultado["kelly_unidades"] = kelly_fracionado(prob_desse_lado, odd)

    return resultado


def calcular_dossie_com_analistas(mercados: list, esporte: str = "futebol", analista: str = "carlos") -> dict:
    """
    Função principal unificada:
    - Carlos: Foco em Matriz de Correlação, EV+ e Alinhamento Tático (Edge mínimo 3.5%).
    - Cris: Foco em Assimetrias Estatísticas, Desvios de Linha e Proteção (Edge mínimo 4.0%).
    """
    resultados_calculados = []
    for m in mercados:
        try:
            resultados_calculados.append(calcular_mercado(m, esporte=esporte))
        except Exception as e:
            resultados_calculados.append({"id": m.get("id"), "status": "erro_calculo", "detalhe": str(e)})

    validos = [r for r in resultados_calculados if r.get("status") == "calculado"]
    
    # Filtro de Edge mínimo ajustado para a Cris (4.0% em vez de 5.0% para aceitar mercados coletivos)
    edge_minimo = 4.0 if analista.lower() == "cris" else 3.5
    
    com_edge = sorted(
        [r for r in validos if r.get("delta_pct") is not None and abs(r.get("delta_pct")) >= edge_minimo and r.get("ev") is not None and r.get("ev") > 0],
        key=lambda x: x["ev"],
        reverse=True
    )

    # 1. MÓDULO DA CRIS: Assimetrias Estatísticas
    asymmetries = []
    for r in validos:
        delta_pct = r.get("delta_pct", 0) or 0
        # Gatilho abaixado para 4.5% para gerar assimetrias em mercados coletivos
        if abs(delta_pct) >= 4.5:
            direcao = "OVER" if delta_pct > 0 else "UNDER"
            asymmetries.append({
                "clash": f"Assimetria em {r.get('mercado_nome', 'Linha Principal')}",
                "statistical_evidence": f"Desvio estatístico de {delta_pct}% detectado entre o modelo e a linha oficial.",
                "betting_angle": f"Cris aponta forte assimetria para o lado do {direcao}. O mercado coletivo está desajustado em relação à média esperada."
            })

    # 2. MÓDULO DO CARLOS: Matriz de Correlação e Seleção da Dupla de Elite
    entrada_1 = com_edge[0] if len(com_edge) > 0 else None
    entrada_2 = None
    
    if len(com_edge) > 1:
        for cand in com_edge[1:]:
            # Carlos valida correlação cruzada para evitar sobreposição de risco no mesmo sentido
            if entrada_1 and cand.get("id") != entrada_1.get("id"):
                if cand.get("lado_odd") != entrada_1.get("lado_odd") or abs(cand.get("delta_pct", 0) - entrada_1.get("delta_pct", 0)) > 2.0:
                    entrada_2 = cand
                    break
        if not entrada_2:
            entrada_2 = com_edge[1]

    return {
        "analista_responsavel": analista.lower(),
        "key_asymmetries": asymmetries[:3],
        "dupla_de_elite": {
            "entrada_1": formatar_para_dupla(entrada_1),
            "entrada_2": formatar_para_dupla(entrada_2)
        }
    }


def formatar_para_dupla(mercado_calculado: Optional[dict]) -> Optional[dict]:
    if not mercado_calculado:
        return None
    
    lado = "Over" if mercado_calculado.get("lado_odd") == "over" else "Under"
    prob = mercado_calculado.get("probabilidade_over") if lado == "Over" else mercado_calculado.get("probabilidade_under")
    
    return {
        "categoria": "VALOR ENCONTRADO",
        "mercado": mercado_calculado.get("mercado_nome", "Linha do Jogo"),
        "selecao": f"{lado} (Lambda Ref: {mercado_calculado.get('esperado_estimado')})",
        "odd": mercado_calculado.get("odd_real_decimal", 1.90),
        "delta_edge": f"{mercado_calculado.get('delta_pct', 0)}%",
        "stake_recomendada": f"{mercado_calculado.get('kelly_unidades', 1.0)}u",
        "motivo": f"EV de {mercado_calculado.get('ev', 0)*100:.1f}% com probabilidade real ajustada de {prob*100:.1f}%."
    }

# ALIAS DE COMPATIBILIDADE PARA EVITAR O ERRO 500 DE IMPORTAÇÃO NO MAIN.PY
calcular_dossie = calcular_dossie_com_analistas
