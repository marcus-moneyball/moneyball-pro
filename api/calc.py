"""

Camada de cálculo determinístico multi-esporte (Delta + Poisson + Normal + Kelly).

Integrado com o motor de decisão de Carlos, analista único e generalista do sistema.

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

# RESULTADO DA PARTIDA -- Moneyline (2 vias), 1X2/Chance Dupla, Handicap Asiático

# ============================================================

# Diferente do Over/Under (que compara UM lambda contra UMA linha), esses

# mercados precisam da DIFERENÇA entre os dois times -- usa Skellam (diferença

# de duas distribuições de Poisson independentes) pra esportes de placar baixo

# e discreto (futebol, beisebol), e Normal pra esportes de placar alto

# (basquete), consistente com o modelo já usado no Over/Under de cada esporte.



def calcular_probabilidades_1x2_skellam(lam_a: float, lam_b: float):

    """P(vitória A), P(empate), P(vitória B) via Skellam. Empate só é um

    resultado real em esportes que o admitem (futebol) -- em esportes sem

    empate, o chamador deve redistribuir p_empate (ver

    calcular_probabilidade_vitoria_2vias)."""

    p_empate = float(stats.skellam.pmf(0, lam_a, lam_b))

    p_vitoria_a = float(1 - stats.skellam.cdf(0, lam_a, lam_b))

    p_vitoria_b = float(stats.skellam.cdf(-1, lam_a, lam_b))

    return round(p_vitoria_a, 4), round(p_empate, 4), round(p_vitoria_b, 4)





def calcular_probabilidade_vitoria_2vias(lam_a: float, lam_b: float, modelo: str = "skellam",

                                          desvio_padrao: Optional[float] = None):

    """

    Moneyline (2 vias, sem empate possível) -- beisebol e basquete.

    - modelo="skellam" (beisebol -- placar baixo, discreto): a massa de "empate"

      matemático (jogo zerado na diferença) é redistribuída 50/50 entre os dois

      lados, já que o esporte sempre resolve o empate (extra innings) e o

      mercado nunca oferece odd de empate.

    - modelo="normal" (basquete -- placar alto, quase contínuo): P(diff > 0)

      direto -- a massa de empate exato é desprezível numa Normal contínua.

    """

    if modelo == "normal":

        if desvio_padrao is None or desvio_padrao <= 0:

            desvio_padrao = 12.0  # mesmo default já usado no Over/Under de basquete

        media_diff = lam_a - lam_b

        desvio_diff = desvio_padrao * math.sqrt(2)  # combina os desvios dos dois times (independência)

        p_a = float(1 - stats.norm.cdf(0, loc=media_diff, scale=desvio_diff))

        p_b = 1 - p_a

        return round(p_a, 4), round(p_b, 4)



    p_a, p_empate, p_b = calcular_probabilidades_1x2_skellam(lam_a, lam_b)

    p_a_final = round(p_a + p_empate / 2, 4)

    p_b_final = round(p_b + p_empate / 2, 4)

    return p_a_final, p_b_final





def _cobre_handicap_linha_simples(lam_a: float, lam_b: float, linha: float):

    """Probabilidade de cobertura (e de push) pra UMA linha inteira ou de meio

    gol -- nunca chamada direto de fora, só pelo split de quarto de gol abaixo.

    `linha` é o handicap aplicado ao time A (negativo = A favorito). Push (a

    aposta "empata" e o dinheiro volta) só é matematicamente possível quando a

    linha é inteira."""

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

    invertendo lam_a/lam_b e o sinal da linha). Suporta linhas inteiras

    (ex: -1.0), de meio gol (ex: -0.5 -- nunca dá push) e de quarto de gol

    (ex: -0.25, -0.75 -- via split entre as duas linhas de meio ponto

    adjacentes, exatamente como o mercado asiático precifica na prática).



    Retorna (probabilidade_cobertura_liquida, probabilidade_push):

    a probabilidade de cobertura já vem NORMALIZADA excluindo a fração que

    seria push (devolução de aposta) -- é essa que deve ser comparada contra

    a odd real pra cálculo de edge, já que o push não gera lucro nem prejuízo.

    """

    linha_x4 = round(linha * 4)

    eh_quarto = linha_x4 % 4 not in (0, 2)  # nem múltiplo de 4 (inteira) nem resto 2 (meio gol)



    if not eh_quarto:

        p_cobre, p_push = _cobre_handicap_linha_simples(lam_a, lam_b, linha)

    else:

        linha_baixa = (linha_x4 - 1) / 4

        linha_alta = (linha_x4 + 1) / 4

        p_cobre_1, p_push_1 = _cobre_handicap_linha_simples(lam_a, lam_b, linha_baixa)

        p_cobre_2, p_push_2 = _cobre_handicap_linha_simples(lam_a, lam_b, linha_alta)

        p_cobre = (p_cobre_1 + p_cobre_2) / 2

        p_push = (p_push_1 + p_push_2) / 2



    p_cobre_liquida = p_cobre / (1 - p_push) if p_push < 1 else p_cobre

    return round(p_cobre_liquida, 4), round(p_push, 4)





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

    "futebol": 0.85,

    "basquete": 0.80,

    "beisebol": 0.80,

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

    # Threshold de 60% (não 55%) -- 55% de posse é só leve oscilação natural de

    # jogo (a maioria das partidas passa por isso em algum momento da média sem

    # ninguém realmente "dominar"), 60%+ é o patamar que a análise de futebol

    # normalmente trata como domínio territorial de fato. Com 55%, simulações

    # mostraram ~76% dos jogos caindo em TIPO B por puro ruído estatístico da

    # posse -- valor mal calibrado, corrigido aqui.

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

            dominante = dominante_posse  # domínio real -- quem domina posse também domina a qualidade

            macro, sub = "TIPO B", "B1_dominio_total"

            evidencias.append(

                f"Time {dominante_posse} com posse média de {posse_dom}% e xG de {xg_dom}, "

                f"contra {xg_advers} do adversário -- posse e qualidade ofensiva convergem."

            )

        else:

            # Contra-ataque letal: quem domina a POSSE não é necessariamente quem está

            # taticamente favorecido -- o time que contra-ataca (posse minoritária) é

            # o lado que o roteiro está sinalizando como perigoso/valorizado.

            dominante = lado_contra_ataque

            macro, sub = "TIPO B", "B2_contra_ataque_letal"

            evidencias.append(

                f"Time {dominante_posse} com posse média de {posse_dom}%, mas xG de {xg_dom} "

                f"próximo ou inferior ao xG do adversário ({xg_advers}) -- domínio territorial "

                f"sem tradução proporcional em qualidade ofensiva; risco de contra-ataque do time {lado_contra_ataque}."

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

        "lado_favorecido": dominante,  # "A" | "B" | None (None em TIPO A -- sem lado estrutural único)

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

        # fatigue_index já existe no catálogo desde o Matchup Engine -- usa o maior

        # dos dois times como proxy de risco de colapso do roteiro.

        "probabilidade_instabilidade_roteiro": (

            round(max(dados_time_a.get("fatigue_index", 0) or 0, dados_time_b.get("fatigue_index", 0) or 0), 3)

            if dados_time_a.get("fatigue_index") is not None or dados_time_b.get("fatigue_index") is not None

            else None

        ),

        "lado_favorecido": dominante,  # "A" | "B" | None (None em TIPO A -- sem lado estrutural único)

    }









def classificar_roteiro_beisebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:

    """Beisebol não é territorial -- é uma sequência de duelos individuais.

    Por isso o macro default é TIPO C (arremessador titular vs. lineup adversário),

    exceto quando os dois duelos do jogo favorecem claramente o MESMO lado (aí vira

    TIPO B -- domínio geral, não só individual). Bullpen vira o sinal de instabilidade

    (relevante para props de innings finais).

    Campos exigidos: pitcher_era, lineup_ops_vs_mao_adversaria (bullpen_era_last_30 é opcional)."""

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



    # Thresholds de referência (aprox. média de liga MLB) -- calibrar depois com histórico.

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

        "lado_favorecido": dominante,  # "A" | "B" | None (None em TIPO C -- duelo individual, não estrutural)

    }





_CLASSIFICADORES_ROTEIRO = {

    "futebol": classificar_roteiro_futebol,

    "basquete": classificar_roteiro_basquete,

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

# MATCHUP ENGINE (Framework Mestre da Análise Esportiva -- Pilar 1)

# ============================================================

# Diferença fundamental em relação ao roteiro (acima): o roteiro mede FORÇA

# relativa (quem é melhor, no agregado). O matchup mede ENCAIXE -- o estilo

# específico de um time quebra o sistema do outro, independente de "quem é

# melhor" no geral. Um time forte pode ter um matchup ruim contra um estilo

# específico, e um time mediano pode ter um matchup ótimo contra ele.

#

# Mesma regra de ouro do resto do arquivo: calculado aqui a partir de dado

# real, nunca inventado pela LLM. Retorna None quando falta o par mínimo de

# campos necessário -- nesse caso não existe bloco de matchup pro Carlos usar,

# e ele segue análise só com força/roteiro, como já fazia antes desta camada.

#

# Cada função retorna o mesmo formato:

# {

#   "matchup_detectado": bool,

#   "sinais": [{"favorece": "A"|"B", "tipo": str, "descricao": str}, ...],

#   "evidencias": [str, ...],  # lista plana, pronta pra injetar no prompt

# }



def calcular_matchup_futebol(dados_time_a: dict, dados_time_b: dict) -> Optional[dict]:

    """Pulo do gato do futebol: Pressão (PPDA) x Fragilidade na Construção.

    Um time que pressiona muito (PPDA baixo) contra um adversário que sofre xG

    alto mesmo com posse (sinal de que não sabe sair jogando sob pressão) tende

    a forçar erros e transições -- Game Script de Caos/Transição, não Domínio.

    Campos exigidos: ppda_medio de pelo menos um lado (posse_media e

    xg_sofrido_medio do lado avaliado, pra confirmar fragilidade)."""

    ppda_a = dados_time_a.get("ppda_medio")

    ppda_b = dados_time_b.get("ppda_medio")



    if ppda_a is None and ppda_b is None:

        return None



    def _fragil_sob_pressao(dados_alvo: dict) -> bool:

        posse = dados_alvo.get("posse_media")

        xg_sofrido = dados_alvo.get("xg_sofrido_medio")

        # Tem posse (não está simplesmente sendo dominado no volume), mas mesmo

        # assim sofre xG relevante -- indício de que a saída de bola quebra sob

        # pressão, não que o time é simplesmente inferior.

        return posse is not None and xg_sofrido is not None and posse >= 50 and xg_sofrido >= 1.3



    PPDA_PRESSAO_ALTA = 8.0  # abaixo disso = pressão sufocante (referência de mercado, calibrar depois)

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

    """Pulo do gato do basquete: Ritmo (Pace) x Fadiga (fatigue_index -- back-to-back

    ou desfalques). Um time rápido contra um adversário cansado tende a expor a

    defesa no meio-campo (transição), inflando o total de pontos e favorecendo

    o handicap do time descansado.

    Campos exigidos: pace de ambos e fatigue_index de pelo menos um lado."""

    pace_a = dados_time_a.get("pace")

    pace_b = dados_time_b.get("pace")

    fadiga_a = dados_time_a.get("fatigue_index")

    fadiga_b = dados_time_b.get("fatigue_index")



    if pace_a is None or pace_b is None:

        return None

    if fadiga_a is None and fadiga_b is None:

        return None



    FADIGA_ALTA = 0.6  # fatigue_index normalizado 0-1 -- calibrar depois com histórico

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

    contra o desempenho do lineup adversário especificamente contra essa mão.

    Um lineup com OPS muito melhor contra a mão do arremessador que vai enfrentar

    tem uma vantagem que a média geral de OPS do time esconde.

    Campos exigidos: pitcher_mao do adversário + lineup_ops_vs_mao_adversaria do

    lado que bate (esse campo já deve vir calculado especificamente contra a mão

    certa -- ver regra no MIE1)."""

    mao_pitcher_a = dados_time_a.get("pitcher_mao")

    mao_pitcher_b = dados_time_b.get("pitcher_mao")

    ops_a_vs_b = dados_time_a.get("lineup_ops_vs_mao_adversaria")

    ops_b_vs_a = dados_time_b.get("lineup_ops_vs_mao_adversaria")



    if mao_pitcher_a is None and mao_pitcher_b is None:

        return None



    OPS_FORTE_CONTRA_MAO = 0.780  # referência de mercado (liga MLB gira ~.720-.740) -- calibrar depois

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

    """

    Matchup Engine determinístico (Framework Mestre, Pilar 1: Força vs. Encaixe).

    Retorna None se faltar o par mínimo de campos, ou se nenhum sinal de matchup

    foi detectado retorna {"matchup_detectado": False, ...} -- ausência de sinal

    também é informação válida (não força o Carlos a inventar um matchup que não existe).

    """

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

# SCORE DE CONVERGÊNCIA (Framework Mestre -- Parte 3: Gestão de Confiança)

# ============================================================

# A Matriz de Decisão do documento tem 5 pilares (Força 30% / Matchup 25% /

# Forma 20% / Contexto 15% / Ruído 10%). Hoje só 2 desses 5 têm dado real no

# pipeline: Força (via roteiro, Cap. V) e Matchup (acima). Forma (janelas

# temporais), Contexto (multiplicador) e Ruído (regressão à média) ainda não

# existem -- por isso esta função NÃO tenta replicar os pesos exatos do

# documento (seria fingir uma precisão que não temos). Em vez disso, mede a

# convergência entre os dois sinais que já existem: roteiro e matchup apontam

# pro MESMO lado, ou entram em conflito? Quando Forma/Contexto/Ru 

