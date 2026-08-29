"""
Camada de cálculo determinístico multi-esporte (Delta + Poisson + Normal + Kelly).[cite: 1]
Integrado com o motor de decisão de Carlos, analista único e generalista do sistema.[cite: 1]
Sem chamadas de rede — 100% testável isoladamente.[cite: 1]
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
    p_under = float(stats.norm.cdf(linha, loc=media, scale=desvio_padrao))
    p_over = 1.0 - p_under
    return round(p_over, 4), round(p_under, 4)


# Razão variância/média empírica pro beisebol -- derivada de dados acadêmicos[cite: 1]
# reais de superdispersão de corridas por entrada.[cite: 1]
RAZAO_VARIANCIA_MEDIA_BEISEBOL = 2.11


def _theta_binomial_negativa(media: float, razao_var_media: float = RAZAO_VARIANCIA_MEDIA_BEISEBOL) -> float:
    """Deriva o parâmetro de dispersão (theta) a partir da média esperada e
    da razão variância/média alvo. razao = 1 + media/theta.[cite: 1]"""
    if media <= 0:
        return 1.0  
    
    razao_segura = max(1.0001, razao_var_media)
    return media / (razao_segura - 1)


def prob_over_under_neg_binomial(linha: float, media: float,
                                  razao_var_media: float = RAZAO_VARIANCIA_MEDIA_BEISEBOL):
    """
    Probabilidade real de Over/Under uma linha, usando Binomial Negativa em
    vez de Poisson -- captura a superdispersão real de corridas no beisebol.[cite: 1]
    """
    theta = _theta_binomial_negativa(media, razao_var_media)
    p = theta / (theta + media)
    n = theta
    piso = math.floor(linha)
    p_under = float(stats.nbinom.cdf(piso, n, p))
    p_over = 1.0 - p_under
    return round(p_over, 4), round(p_under, 4)


def calcular_delta_mercado(lam: float, linha: float):
    delta_abs = round(lam - linha, 3)
    delta_pct = round((delta_abs / abs(linha)) * 100, 2) if linha != 0.0 else None
    return delta_abs, delta_pct


# ============================================================
# RESULTADO DA PARTIDA -- Moneyline (2 vias), 1X2/Chance Dupla, Handicap Asiático[cite: 1]
# ============================================================

def calcular_probabilidades_1x2_skellam(lam_a: float, lam_b: float):
    """P(vitória A), P(empate), P(vitória B) via Skellam.[cite: 1]"""
    p_empate = float(stats.skellam.pmf(0, lam_a, lam_b))
    p_vitoria_a = float(1 - stats.skellam.cdf(0, lam_a, lam_b))
    p_vitoria_b = float(stats.skellam.cdf(-1, lam_a, lam_b))
    return round(p_vitoria_a, 4), round(p_empate, 4), round(p_vitoria_b, 4)


def calcular_probabilidade_vitoria_2vias(lam_a: float, lam_b: float, modelo: str = "skellam",
                                          desvio_padrao: Optional[float] = None):
    """
    Moneyline (2 vias, sem empate possível) -- beisebol e basquete.[cite: 1]
    """
    if modelo == "normal":
        if desvio_padrao is None or desvio_padrao <= 0:
            desvio_padrao = 12.0  
        media_diff = lam_a - lam_b
        desvio_diff = desvio_padrao * math.sqrt(2) 
        p_a = float(1 - stats.norm.cdf(0, loc=media_diff, scale=desvio_diff))
        p_b = 1 - p_a
        return round(p_a, 4), round(p_b, 4)

    p_a, p_empate, p_b = calcular_probabilidades_1x2_skellam(lam_a, lam_b)
    p_a_final = round(p_a + p_empate / 2, 4)
    p_b_final = round(p_b + p_empate / 2, 4)
    return p_a_final, p_b_final


def _cobre_handicap_linha_simples(lam_a: float, lam_b: float, linha: float):
    """Probabilidade de cobertura (e de push) pra UMA linha inteira ou de meio
    gol -- nunca chamada direto de fora, só pelo split de quarto de gol abaixo.[cite: 1]"""
    limite = -linha
    if float(limite).is_integer():
        p_push = float(stats.skellam.pmf(int(limite), lam_a, lam_b))
        p_cobre = float(1 - stats.skellam.cdf(int(limite), lam_a, lam_b))
    else:
        p_push = 0.0
        p_cobre = float(1 - stats.skellam.cdf(math.floor(limite), lam_a, lam_b))
    return p_cobre, p_push


def calcular_probabilidade_handicap_asiatico(lam_a: float, lam_b: float, linha: float):
    """
    Handicap Asiático aplicado ao time A (pra calcular do lado do time B, chame
    invertendo lam_a/lam_b e o sinal da linha). Suporta linhas inteiras, de meio e quarto.[cite: 1]
    """
    linha_x4 = round(linha * 4)
    eh_quarto = linha_x4 % 4 not in (0, 2)  

    if not eh_quarto:
        p_cobre, p_push = _cobre_handicap_linha_simples(lam_a, lam_b, linha)
    else:
        linha_baixa = (linha_x4 - 1) / 4
        linha_alta = (linha_x4 + 1) / 4
        p_cobre_1, p_push_1 = _cobre_handicap_linha_simples(lam_a, lam_b, linha_baixa)
        p_cobre_2, p_push_2 = _cobre_handicap_linha_simples(lam_a, lam_b, linha_alta)
        p_cobre = (p_cobre_1 + p_cobre_2) / 2
        p_push = (p_push_1 + p_push_2) / 2

    return round(p_cobre, 4), round(p_push, 4)


# ============================================================
# ROBUSTEZ (confiança nos dados de entrada)[cite: 1]
# ============================================================

AMOSTRA_MINIMA_JOGOS = 10
PENALIDADE_FATOR_ALTO = 0.25
PENALIDADE_FATOR_MEDIO = 0.10


def calcular_nivel_confianca_dados(tamanho_amostra: Optional[int] = None,
                                   fatores_incerteza: Optional[list] = None) -> float:
    """
    Nível de confiança (0 a 1) nos dados que sustentam a projeção, com atenuação multiplicativa.
    """
    if tamanho_amostra is None:
        confianca_amostra = 0.5
    else:
        confianca_amostra = max(0.0, min(1.0, tamanho_amostra / AMOSTRA_MINIMA_JOGOS))

    confianca_contexto = 1.0
    for fator in (fatores_incerteza or []):
        impacto = (fator.get("impact_level") if isinstance(fator, dict) else None) or "low"
        if impacto == "high":
            confianca_contexto *= (1.0 - PENALIDADE_FATOR_ALTO)
        elif impacto == "medium":
            confianca_contexto *= (1.0 - PENALIDADE_FATOR_MEDIO)

    confianca_final = (confianca_amostra * 0.70) + (confianca_contexto * 0.30)
    return round(max(0.1, min(1.0, confianca_final)), 3)


def calcular_fator_robustez(nivel_confianca: float) -> float:
    """Robustez = min(1.0, 0.85 + 0.15 * nivel_confianca). Piso 0.85, teto 1.0.[cite: 1]"""
    nivel_confianca = max(0.0, min(1.0, nivel_confianca))
    return round(min(1.0, 0.85 + 0.15 * nivel_confianca), 4)


def calcular_probabilidade_real_ajustada(p_modelo: float, robustez: float) -> float:
    """
    Probabilidade real ajustada via encolhimento bayesiano (Bayesian Shrinkage) 
    em direção ao centro neutro (0.5).
    """
    if p_modelo is None:
        return None
        
    robustez_clamped = max(0.0, min(1.0, robustez))
    p_ajustada = (p_modelo * robustez_clamped) + (0.5 * (1.0 - robustez_clamped))
    
    return round(max(0.0001, min(0.9999, p_ajustada)), 4)


# ============================================================
# EV + KELLY FRACIONADO[cite: 1]
# ============================================================

def calcular_ev(prob_real: float, odd_decimal: float):
    if prob_real is None or odd_decimal is None:
        return None
    return round((prob_real * odd_decimal) - 1, 4)


def kelly_fracionado(prob_real: float, odd_decimal: float, fracao=0.25, teto_unidades=2.5) -> Optional[float]:
    """
    Kelly fracionado em unidades (escala de referência: banca = 10u).[cite: 1]
    SEM piso artificial -- um edge minúsculo gera stake minúscula, um edge forte
    gera stake maior (até o teto). Arredondado em degraus de 0.25u.[cite: 1]
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

    if unidades_arredondadas <= 0:
        return None
    return unidades_arredondadas


# ============================================================
# ESTIMATIVA DE LAMBDA (expectativa real a partir de médias do MDM)[cite: 1]
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
# ROTEIRO DE JOGO (Metodologia Nexus Cap. V) -- classificação determinística[cite: 1]
# ============================================================

CONFIANCA_ROTEIRO_GROUNDED = {
    "futebol": 0.85,
    "basquete": 0.80,
    "beisebol": 0.80,
}


def classificar_roteiro_futebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Modelo territorial completo (5 arquétipos -- B1/B2/A1/A2/C1).[cite: 1]"""
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
    dominante = None  

    dominante_posse = None
    if posse_a is not None and posse_b is not None:
        if posse_a >= 60:
            dominante_posse = "A"
        elif posse_b >= 60:
            dominante_posse = "B"

    if dominante_posse:
        xg_dom = xg_a if dominante_posse == "A" else xg_b
        xg_advers = xg_b if dominante_posse == "A" else xg_a
        posse_dom = posse_a if dominante_posse == "A" else posse_b
        lado_contra_ataque = "B" if dominante_posse == "A" else "A"

        if xg_dom - xg_advers >= 0.5:
            dominante = dominante_posse 
            macro, sub = "TIPO B", "B1_dominio_total"
            evidencias.append(
                f"Time {dominante_posse} com posse média de {posse_dom}% e xG de {xg_dom}, "
                f"contra {xg_advers} do adversário -- posse e qualidade ofensiva convergem."
            )
        else:
            dominante = lado_contra_ataque
            macro, sub = "TIPO B", "B2_contra_ataque_letal"
            evidencias.append(
                f"Time {dominante_posse} com posse média de {posse_dom}%, mas xG de {xg_dom} "
                f"próximo ou inferior ao xG do adversário ({xg_advers}) -- domínio territorial "
                f"sem tradução proporcional em qualidade ofensiva; risco de contra-ataque do time {lado_contra_ataque}."
            )
    elif abs(delta_xg) >= 0.6:
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
        xg_sofrido_favorito = xg_sofrido_a if dominante == "A" else (xg_sofrido_b if dominante == "B" else max(xg_sofrido_a, xg_sofrido_b))
        instabilidade = round(min(1.0, max(0.0, xg_sofrido_favorito / 2.0)), 3)

    return {
        "macro": macro,
        "sub_tipo": sub,
        "confianca_classificacao": CONFIANCA_ROTEIRO_GROUNDED["futebol"],
        "evidencias": evidencias,
        "probabilidade_instabilidade_roteiro": instabilidade,
        "lado_favorecido": dominante, 
    }


def classificar_roteiro_basquete(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Modelo de pace + eficiência líquida (ORTG do ataque vs DRTG da defesa adversária).[cite: 1]"""
    ortg_a, drtg_a = dados_time_a.get("ortg"), dados_time_a.get("drtg")
    ortg_b, drtg_b = dados_time_b.get("ortg"), dados_time_b.get("drtg")
    pace_a, pace_b = dados_time_a.get("pace"), dados_time_b.get("pace")

    if None in (ortg_a, drtg_a, ortg_b, drtg_b):
        return None

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
        dominante = None
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
        "probabilidade_instabilidade_roteiro": (
            round(max(dados_time_a.get("fatigue_index", 0) or 0, dados_time_b.get("fatigue_index", 0) or 0), 3)
            if dados_time_a.get("fatigue_index") is not None or dados_time_b.get("fatigue_index") is not None
            else None
        ),
        "lado_favorecido": dominante, 
    }


def classificar_roteiro_beisebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Beisebol não é territorial -- é uma sequência de duelos individuais.[cite: 1]"""
    era_a = dados_time_a.get("pitcher_era")
    era_b = dados_time_b.get("pitcher_era")
    ops_a = dados_time_a.get("lineup_ops_vs_mao_adversaria")
    ops_b = dados_time_b.get("lineup_ops_vs_mao_adversaria")
    bullpen_a = dados_time_a.get("bullpen_era_last_30")
    bullpen_b = dados_time_b.get("bullpen_era_last_30")

    if None in (era_a, era_b, ops_a, ops_b):
        return None

    evidencias = [
        f"Arremessador titular do time A (ERA {era_a}) contra lineup adversário (OPS {ops_b}).",
        f"Arremessador titular do time B (ERA {era_b}) contra lineup adversário (OPS {ops_a}).",
    ]

    duelo_a_favoravel = era_a <= 3.80 and ops_b <= 0.720
    duelo_b_favoravel = era_b <= 3.80 and ops_a <= 0.720

    if duelo_a_favoravel and not duelo_b_favoravel:
        macro, sub, dominante = "TIPO B", None, "A"
        evidencias.append("Duelo pitcher x lineup favorece claramente o time A dos dois lados do jogo -- sinal de domínio geral, não só individual.")
    elif duelo_b_favoravel and not duelo_a_favoravel:
        macro, sub, dominante = "TIPO B", None, "B"
        evidencias.append("Duelo pitcher x lineup favorece claramente o time B dos dois lados do jogo -- sinal de domínio geral, não só individual.")
    else:
        macro, sub, dominante = "TIPO C", "C1_duelo_pitcher_lineup", None

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
        "lado_favorecido": dominante, 
    }


_CLASSIFICADORES_ROTEIRO = {
    "futebol": classificar_roteiro_futebol,
    "basquete": classificar_roteiro_basquete,
    "beisebol": classificar_roteiro_beisebol,
}


def classificar_roteiro_jogo(esporte: str, dados_time_a: Optional[dict], dados_time_b: Optional[dict]) -> Optional[dict]:
    """Classificador determinístico de roteiro de jogo (Metodologia Nexus, Cap. V).[cite: 1]"""
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
# MATCHUP ENGINE (Framework Mestre da Análise Esportiva -- Pilar 1)[cite: 1]
# ============================================================

def calcular_matchup_futebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Pulo do gato do futebol: Pressão (PPDA) x Fragilidade na Construção.[cite: 1]"""
    ppda_a = dados_time_a.get("ppda_medio")
    ppda_b = dados_time_b.get("ppda_medio")

    if ppda_a is None and ppda_b is None:
        return None

    def _fragil_sob_pressao(dados_alvo: dict) -> bool:
        posse = dados_alvo.get("posse_media")
        xg_sofrido = dados_alvo.get("xg_sofrido_medio")
        return posse is not None and xg_sofrido is not None and posse >= 50 and xg_sofrido >= 1.3

    PPDA_PRESSAO_ALTA = 8.0  
    sinais = []

    if ppda_b is not None and ppda_b <= PPDA_PRESSAO_ALTA and _fragil_sob_pressao(dados_time_a):
        sinais.append({
            "favorece": "B",
            "tipo": "pressao_quebra_construcao",
            "descricao": (
                f"PPDA do time B em {ppda_b} (pressão sufocante) contra o time A, que "
                f"sofre xG elevado mesmo com posse alta -- indício de fragilidade na "
                f"saída de bola sob pressão, risco de erros forçados e transições."
            ),
        })
    if ppda_a is not None and ppda_a <= PPDA_PRESSAO_ALTA and _fragil_sob_pressao(dados_time_b):
        sinais.append({
            "favorece": "A",
            "tipo": "pressao_quebra_construcao",
            "descricao": (
                f"PPDA do time A em {ppda_a} (pressão sufocante) contra o time B, que "
                f"sofre xG elevado mesmo com posse alta -- indício de fragilidade na "
                f"saída de bola sob pressão, risco de erros forçados e transições."
            ),
        })

    if not sinais:
        return {"matchup_detectado": False, "sinais": [], "evidencias": []}

    return {"matchup_detectado": True, "sinais": sinais, "evidencias": [s["descricao"] for s in sinais]}


def calcular_matchup_basquete(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Pulo do gato do basquete: Ritmo (Pace) x Fadiga (fatigue_index -- back-to-back ou desfalques).[cite: 1]"""
    pace_a = dados_time_a.get("pace")
    pace_b = dados_time_b.get("pace")
    fadiga_a = dados_time_a.get("fatigue_index")
    fadiga_b = dados_time_b.get("fatigue_index")

    if pace_a is None or pace_b is None:
        return None
    if fadiga_a is None and fadiga_b is None:
        return None

    FADIGA_ALTA = 0.6  
    PACE_RAPIDO = 100.0
    sinais = []

    if fadiga_b is not None and fadiga_b >= FADIGA_ALTA and pace_a >= PACE_RAPIDO:
        sinais.append({
            "favorece": "A",
            "tipo": "ritmo_explora_fadiga",
            "descricao": (
                f"Time A joga em ritmo acelerado (pace {pace_a}) contra Time B com "
                f"índice de fadiga elevado ({fadiga_b}) -- pernas cansadas tendem a "
                f"ceder espaço no half-court, favorecendo Over de pontos e handicap do Time A."
            ),
        })
    if fadiga_a is not None and fadiga_a >= FADIGA_ALTA and pace_b >= PACE_RAPIDO:
        sinais.append({
            "favorece": "B",
            "tipo": "ritmo_explora_fadiga",
            "descricao": (
                f"Time B joga em ritmo acelerado (pace {pace_b}) contra Time A com "
                f"índice de fadiga elevado ({fadiga_a}) -- pernas cansadas tendem a "
                f"ceder espaço no half-court, favorecendo Over de pontos e handicap do Time B."
            ),
        })

    if not sinais:
        return {"matchup_detectado": False, "sinais": [], "evidencias": []}

    return {"matchup_detectado": True, "sinais": sinais, "evidencias": [s["descricao"] for s in sinais]}


def calcular_matchup_beisebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:
    """Pulo do gato do beisebol: Platoon Split -- a mão do arremessador titular
    contra o desempenho do lineup adversário especificamente contra essa mão.[cite: 1]"""
    mao_pitcher_a = dados_time_a.get("pitcher_mao")
    mao_pitcher_b = dados_time_b.get("pitcher_mao")
    ops_a_vs_b = dados_time_a.get("lineup_ops_vs_mao_adversaria")
    ops_b_vs_a = dados_time_b.get("lineup_ops_vs_mao_adversaria")

    if mao_pitcher_a is None and mao_pitcher_b is None:
        return None

    OPS_FORTE_CONTRA_MAO = 0.780  
    sinais = []

    if mao_pitcher_b is not None and ops_a_vs_b is not None and ops_a_vs_b >= OPS_FORTE_CONTRA_MAO:
        sinais.append({
            "favorece": "A",
            "tipo": "platoon_split_favoravel",
            "descricao": (
                f"Lineup do time A tem OPS de {ops_a_vs_b} especificamente contra "
                f"arremessadores {'destros' if mao_pitcher_b == 'R' else 'canhotos'} -- "
                f"exatamente o perfil do arremessador titular do time B -- vantagem "
                f"que a média geral de OPS do time não capturaria."
            ),
        })
    if mao_pitcher_a is not None and ops_b_vs_a is not None and ops_b_vs_a >= OPS_FORTE_CONTRA_MAO:
        sinais.append({
            "favorece": "B",
            "tipo": "platoon_split_favoravel",
            "descricao": (
                f"Lineup do time B tem OPS de {ops_b_vs_a} especificamente contra "
                f"arremessadores {'destros' if mao_pitcher_a == 'R' else 'canhotos'} -- "
                f"exatamente o perfil do arremessador titular do time A -- vantagem "
                f"que a média geral de OPS do time não capturaria."
            ),
        })

    if not sinais:
        return {"matchup_detectado": False, "sinais": [], "evidencias": []}

    return {"matchup_detectado": True, "sinais": sinais, "evidencias": [s["descricao"] for s in sinais]}


_CALCULADORES_MATCHUP = {
    "futebol": calcular_matchup_futebol,
    "basquete": calcular_matchup_basquete,
    "beisebol": calcular_matchup_beisebol,
}


def calcular_matchup(esporte: str, dados_time_a: Optional[dict], dados_time_b: Optional[dict]) -> Optional[dict]:
    """Matchup Engine determinístico (Framework Mestre, Pilar 1: Força vs. Encaixe).[cite: 1]"""
    if not dados_time_a or not dados_time_b:
        return None

    fn = _CALCULADORES_MATCHUP.get(esporte.lower())
    if not fn:
        return None

    try:
        return fn(dados_time_a, dados_time_b)
    except Exception:
        return None


# ============================================================
# SCORE DE CONVERGÊNCIA (Framework Mestre -- Parte 3: Gestão de Confiança)[cite: 1]
# ============================================================

def _lado_favorecido_pelo_roteiro(roteiro: Optional[dict]) -> Optional[str]:
    """Lê o lado (A/B) que o roteiro favorece, quando aplicável.[cite: 1]"""
    if not roteiro:
        return None
    return roteiro.get("lado_favorecido")


def calcular_convergencia(roteiro: Optional[dict], matchup: Optional[dict]) -> dict:
    """Mede se roteiro (Força) e matchup (Encaixe) apontam pro mesmo lado.[cite: 1]"""
    lado_roteiro = _lado_favorecido_pelo_roteiro(roteiro)
    sinais_matchup = (matchup or {}).get("sinais", []) if matchup and matchup.get("matchup_detectado") else []
    lados_matchup = {s["favorece"] for s in sinais_matchup}

    if lado_roteiro is None and not lados_matchup:
        return {
            "nivel": "NEUTRO",
            "direcao_favorecida": None,
            "teto_stake_unidades": 1.0,
            "motivo": "Nem roteiro nem matchup indicam um lado estrutural favorecido -- convergência não avaliável com os dados disponíveis.",
        }

    if lado_roteiro is None or not lados_matchup:
        lado_unico = lado_roteiro or next(iter(lados_matchup))
        origem = "roteiro" if lado_roteiro else "matchup"
        return {
            "nivel": "MEDIA",
            "direcao_favorecida": lado_unico,
            "teto_stake_unidades": 1.0,
            "motivo": f"Apenas o {origem} indica o time {lado_unico} favorecido -- sem segundo pilar pra confirmar convergência, stake permanece no padrão.",
        }

    if lado_roteiro in lados_matchup:
        return {
            "nivel": "ALTA",
            "direcao_favorecida": lado_roteiro,
            "teto_stake_unidades": 2.0,
            "motivo": f"Roteiro (Força) e Matchup (Encaixe) convergem no time {lado_roteiro} -- convergência absoluta entre os dois pilares disponíveis.",
        }

    return {
        "nivel": "BAIXA",
        "direcao_favorecida": None,
        "teto_stake_unidades": 0.5,
        "motivo": f"Roteiro aponta para o time {lado_roteiro}, mas o Matchup aponta para {sorted(lados_matchup)} -- sinais conflitantes, reduzir exposição.",
    }


# ============================================================
# MSC (Moneyball Score) -- selo de confiabilidade pro usuário[cite: 1]
# ============================================================

PESOS_MSC = {
    "carlos": {"ev": 0.60, "delta": 0.25, "robustez_ou_prob": 0.15},
}

EV_TETO_NORMALIZACAO = 0.30    
DELTA_TETO_NORMALIZACAO = 15.0  


def calcular_msc(ev: Optional[float], delta_pct: Optional[float],
                  prob_real_ajustada: Optional[float], robustez: float,
                  persona: str = "carlos") -> Optional[int]:
    """MSC base, 0-100 -- só a força matemática do candidato isolado.[cite: 1]"""
    if ev is None or delta_pct is None or prob_real_ajustada is None:
        return None

    pesos = PESOS_MSC.get(persona.lower(), PESOS_MSC["carlos"])

    ev_norm = max(0.0, min(1.0, ev / EV_TETO_NORMALIZACAO))
    delta_norm = max(0.0, min(1.0, abs(delta_pct) / DELTA_TETO_NORMALIZACAO))
    componente_terciario = robustez

    score = (
        pesos["ev"] * ev_norm +
        pesos["delta"] * delta_norm +
        pesos["robustez_ou_prob"] * componente_terciario
    )
    return round(max(0, min(100, score * 100)))


AJUSTE_MSC_POR_CONVERGENCIA = {
    "ALTA": 12,
    "MEDIA": 0,
    "NEUTRO": 0,
    "BAIXA": -25,
}


def ajustar_msc_por_convergencia(msc_base: Optional[int], nivel_convergencia: Optional[str]) -> Optional[int]:
    """Aplica o ajuste de convergência ao MSC base.[cite: 1]"""
    if msc_base is None:
        return None
    ajuste = AJUSTE_MSC_POR_CONVERGENCIA.get(nivel_convergencia, 0)
    return max(0, min(100, msc_base + ajuste))


ROTULOS_CONFIANCA = [
    (80, "Convicção Elite"),
    (60, "Convicção Alta"),
    (40, "Convicção Moderada"),
    (0, "Convicção Baixa"),
]


def rotulo_confianca(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    for limite, rotulo in ROTULOS_CONFIANCA:
        if score >= limite:
            return rotulo
    return "Convicção Baixa"


# ============================================================
# APOSTA COMBINADA (Dupla de Elite como bet builder / múltipla única)[cite: 1]
# ============================================================

MAPA_STAKE_COMBINADA = {2.0: 1.0, 1.0: 0.5, 0.5: 0.5}

MARGEM_MINIMA_COMBINADA_PCT = 3.0


def calcular_aposta_combinada(prob_1: float, odd_1: float, prob_2: float, odd_2: float,
                               teto_stake_convergencia: float = 1.0) -> dict:
    """Calcula odd/probabilidade/edge estimados de uma aposta combinada (2 pernas do mesmo jogo).[cite: 1]"""
    prob_combinada_estimada = round(prob_1 * prob_2, 4)
    odd_combinada_estimada = round(odd_1 * odd_2, 2)
    prob_implicita_combinada = round(1 / odd_combinada_estimada, 4) if odd_combinada_estimada else None
    edge_combinado_pct = (
        round((prob_combinada_estimada - prob_implicita_combinada) * 100, 2)
        if prob_implicita_combinada is not None else None
    )

    aprovada = edge_combinado_pct is not None and edge_combinado_pct >= MARGEM_MINIMA_COMBINADA_PCT
    stake_combinada = MAPA_STAKE_COMBINADA.get(teto_stake_convergencia, 0.5) if aprovada else 0.0

    if aprovada:
        aviso = (
            "Odd e probabilidade estimadas a partir das duas pernas separadas -- "
            "a casa pode ajustar a odd pra baixo ao montar a aposta combinada de "
            "verdade (bet builder / múltipla do mesmo jogo), por causa da "
            "correlação entre as entradas. Confira a odd real oferecida antes de "
            "apostar -- se vier mais baixa, o edge real também cai."
        )
    else:
        aviso = (
            f"Margem combinada estimada ({edge_combinado_pct}%) abaixo do piso de "
            f"segurança ({MARGEM_MINIMA_COMBINADA_PCT}%) -- não é uma recomendação, "
            "é um alerta. As duas pernas continuam válidas separadamente; considere "
            "apostar nelas de forma isolada em vez de combinada."
        )

    return {
        "probabilidade_combinada_estimada": prob_combinada_estimada,
        "odd_combinada_estimada": odd_combinada_estimada,
        "probabilidade_implicita_combinada": prob_implicita_combinada,
        "edge_combinado_estimado_pct": edge_combinado_pct,
        "aprovada": aprovada,
        "stake_recomendada": f"{stake_combinada}u",
        "aviso": aviso,
    }

# ============================================================
# CÁLCULO POR MERCADO ISOLADO (usado pelo endpoint utilitário /api/v1/calc)[cite: 1]
# ============================================================

def calcular_mercado(mercado: dict, esporte: str = "futebol") -> dict:
    linha = mercado.get("linha")
    if linha is None:
        return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}

    esporte_key = esporte.lower()

    if esporte_key == "basquete" and mercado.get("modelo") != "poisson":
        media_esperada = mercado.get("media_esperada") or estimar_lambda(mercado)
        if media_esperada is None:
            return {"id": mercado.get("id"), "status": "sem_dados_suficientes"}
        std_dev = mercado.get("desvio_padrao", 12.0)
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
    de resultados por mercado.[cite: 1]"""
    resultados = []
    for m in mercados:
        try:
            resultados.append(calcular_mercado(m, esporte=esporte))
        except Exception as e:
            resultados.append({"id": m.get("id"), "status": "erro_calculo", "detalhe": str(e)})
    return resultados
