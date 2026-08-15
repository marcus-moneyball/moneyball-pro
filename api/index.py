import os
import json
import re
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from groq import Groq

app = FastAPI(title="MoneyballPro Engine", version="2.3.0")

# Habilita CORS para liberar requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# CAMADA DE CÁLCULO DETERMINÍSTICO (Delta + Poisson + Kelly Fracionado)
# Escrita originalmente pelo Léo em mie2_calc.py. Colada direto aqui
# (em vez de importada de outro arquivo) porque a Vercel empacota cada
# arquivo da pasta api/ como função isolada e não inclui automaticamente
# arquivos irmãos — um "from mie2_calc import ..." dava ModuleNotFoundError
# em produção mesmo com o arquivo presente no repositório.
#
# Princípio: cada mercado é calculado isoladamente. Falta de dado em UM
# mercado nunca derruba os outros. Nunca inventa número que não veio do
# MDM/odd real.
# ============================================================
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


def calcular_delta_mercado(lam: float, linha: float):
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
    delta_abs, delta_pct = calcular_delta_mercado(lam, linha)

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

# ============================================================
# FIM DA CAMADA DE CÁLCULO DETERMINÍSTICO
# ============================================================

# DICIONÁRIO NECESSÁRIO PARA A FUNÇÃO montar_system_prompt_mie2
REGRAS_ESPORTES = {
    "futebol": "Mercados: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor.",
    "nfl": "Mercados: Vencedor (Moneyline), Spread (Handicap), Total de Pontos, Yardas de Passe/Corrida/Recepção, Touchdown Qualquer Momento."
}

# ============================================================
# PERFIS NUMÉRICOS POR ANALISTA
# Esta é a diferença REAL entre Cris e Carlos — não é só tom de
# voz no prompt, é parâmetro objetivo aplicado em duas camadas:
# 1) injetado no prompt pro modelo já mirar nesses números
# 2) validado de novo em Python depois da resposta (ver
#    validar_e_sanear_entrada), porque o modelo pode ignorar
#    a instrução em texto e a gente não pode confiar só nisso.
# ============================================================
PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 3.5,
        "odd_min": 1.50,
        "odd_max": 3.50,
        "faixas_stake": [
            (3.5, 6.0, "1.0u"),
            (6.0, 8.5, "1.5u"),
            (8.5, float("inf"), "2.0u"),
        ],
    },
    "cris": {
        "delta_min": 4.0,
        "odd_min": 1.60,
        "odd_max": 2.80,
        "faixas_stake": [
            (4.0, 6.5, "1.0u"),
            (6.5, 9.0, "1.5u"),
            (9.0, float("inf"), "2.0u"),
        ],
    },
}

# Teto de sanidade: nenhum mercado líquido de futebol/NBA/NFL/MLB tem uma
# casa de apostas errando a precificação por mais que isso. Delta acima
# deste valor é sinal de que o modelo inventou a probabilidade em vez de
# estimar com base no que está nos prints.
DELTA_MAX_PLAUSIVEL = 15.0


def get_gemini_client():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não configurada na Vercel.")
    return genai.Client(api_key=GEMINI_API_KEY)


def get_groq_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada na Vercel.")
    return Groq(api_key=GROQ_API_KEY)


def _extrair_json_de_texto(texto: str) -> Optional[dict]:
    """Isola e parseia o primeiro bloco {...} de um texto, tolerando cercas
    de markdown (```json ... ```). Retorna None se não conseguir parsear —
    nunca lança exceção, porque busca/extração falhando não pode derrubar
    o resto da análise (o app cai de volta pra estimativa do Groq)."""
    if not texto:
        return None
    texto = texto.strip()
    if "```" in texto:
        texto = re.sub(r"^```(?:json)?\s*", "", texto)
        texto = re.sub(r"\s*```$", "", texto)
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extrair_time_e_mercados_gols(gemini_client, contents: list, sport: str) -> Optional[dict]:
    """
    Passo estruturado (separado do OCR de texto livre já existente): pede ao
    Gemini pra devolver JSON com os nomes dos times e os mercados de TOTAL DE
    GOLS presentes nos prints (linha, odd, lado). Isso é o que permite montar
    o input pro cálculo Poisson determinístico — sem isso, calcular_dossie não
    tem como ser alimentado automaticamente.

    Escopo hoje: só Total de Gols (mercado mais comum e o que o cálculo do
    Léo já cobre). Outros mercados continuam pela estimativa do Groq mais
    abaixo no pipeline.
    """
    if sport.lower() != "futebol":
        return None  # Poisson de gols só se aplica a futebol por enquanto

    prompt = """Extraia dos prints, em JSON estrito (sem markdown, sem texto fora do JSON):
{
  "time_a": "Nome do primeiro time mencionado",
  "time_b": "Nome do segundo time mencionado",
  "mercados_gols": [
    {"linha": 2.5, "odd": 1.85, "lado": "under"},
    {"linha": 2.5, "odd": 1.95, "lado": "over"}
  ]
}
Regras:
- "mercados_gols" deve conter APENAS linhas de Total de Gols da Partida (Over/Under) que estejam explicitamente visíveis nos prints, com odd real.
- "lado" deve ser exatamente "over" ou "under".
- Se não houver nenhum mercado de Total de Gols visível, retorne "mercados_gols": [].
- NUNCA invente time, linha ou odd que não esteja no print."""

    try:
        res = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents + [prompt],
            config=types.GenerateContentConfig(temperature=0)
        )
        dados = _extrair_json_de_texto(res.text)
        print(f"[EXTRACAO ESTRUTURADA DEBUG] {dados}")
        return dados
    except Exception as e:
        print(f"[EXTRACAO ESTRUTURADA ERRO] {str(e)}")
        return None  # falha aqui não derruba o resto — cai pro fluxo antigo


FONTES_AUTORIZADAS_POR_ESPORTE = {
    "futebol": "site:fbref.com OR site:sofascore.com",
    "basquete": "site:basketball-reference.com OR site:nba.com",
    "beisebol": "site:baseballsavant.com OR site:baseball-reference.com",
    "nfl": "site:pro-football-reference.com",
}


def executar_mie1(gemini_client, time_a: str, time_b: str, sport: str) -> Optional[dict]:
    """
    Estágio de investigação (MIE1): busca real via Google Search grounding,
    restrita a fontes de autoridade por esporte (mesmo princípio do MIE1
    original, adaptado de JS pra Python). Devolve os gols esperados de cada
    time já prontos (team_a_projected / team_b_projected) — o Poisson usa
    esses dois números direto como λ, sem precisar decompor em "marcada" e
    "sofrida" separadamente, porque essa parte (cruzar ataque de um time
    com defesa do outro) já é feita aqui, com base em pesquisa real.

    Se a busca falhar ou vier incompleta, retorna None — pipeline cai de
    volta pra estimativa do Groq (fallback seguro), nunca quebra a análise.
    """
    esporte_key = sport.lower()
    fontes = FONTES_AUTORIZADAS_POR_ESPORTE.get(esporte_key, FONTES_AUTORIZADAS_POR_ESPORTE["futebol"])

    prompt = f"""Você é um Investigador Quantitativo Esportivo. Busque na internet, OBRIGATORIAMENTE
usando o operador de busca {fontes}, as estatísticas mais recentes e confiáveis dos
times "{time_a}" e "{time_b}" para {sport.upper()}.

Investigue também fatores contextuais atuais: desfalques confirmados, lesões, clima
no horário do jogo, fadiga de calendário.

Retorne ESTRITAMENTE este JSON, sem markdown, sem texto fora do JSON:
{{
  "team_a_projected": 1.4,
  "team_b_projected": 1.1,
  "fonte": "nome do site usado",
  "contextual_factors": [
    {{"factor_type": "injury", "description": "...", "impact_level": "high|medium|low", "affected_team": "..."}}
  ],
  "key_asymmetries": [
    {{"clash": "...", "statistical_evidence": "...", "betting_angle": "..."}}
  ]
}}

Regras:
- "team_a_projected" e "team_b_projected" são a expectativa de GOLS (ou pontos/runs,
  conforme o esporte) de cada time NESTE confronto — já cruzando o ataque de um time
  com a defesa do outro, baseado em dados reais e atuais encontrados na busca.
- Se não encontrar dado recente e confiável para os dois times nas fontes indicadas,
  retorne null no lugar do JSON inteiro (sem inventar números)."""

    try:
        res = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        dados = _extrair_json_de_texto(res.text)
        print(f"[MIE1 DEBUG] time_a={time_a} time_b={time_b} sport={sport} resultado={dados}")

        if not dados:
            return None
        if dados.get("team_a_projected") is None or dados.get("team_b_projected") is None:
            return None  # busca incompleta — mais seguro descartar que usar parcial

        return dados
    except Exception as e:
        print(f"[MIE1 ERRO] {str(e)}")
        return None  # falha na busca não derruba o resto — cai pro fluxo antigo


def montar_candidatos_gols_calculados(mercados_gols: list, projecao_mie1: dict) -> list:
    """
    Roda cada mercado de Total de Gols extraído através do cálculo Poisson
    determinístico, usando λ = team_a_projected + team_b_projected (já
    calculado pelo MIE1 via pesquisa real). Computa o edge real:
    Δ = Probabilidade_Real_Calculada - Probabilidade_Implícita_da_Odd.
    """
    lam_total = round(projecao_mie1["team_a_projected"] + projecao_mie1["team_b_projected"], 3)

    candidatos = []
    for i, m in enumerate(mercados_gols):
        linha = m.get("linha")
        odd = m.get("odd")
        lado = m.get("lado")
        if linha is None or odd is None or lado not in ("over", "under"):
            continue

        p_over, p_under = prob_over_under(linha, lam_total)
        prob_real = p_under if lado == "under" else p_over
        prob_implicita_odd = round(1 / odd, 4) if odd else None
        edge_pct = round((prob_real - prob_implicita_odd) * 100, 2) if prob_implicita_odd is not None else None
        ev = calcular_ev(prob_real, odd)
        kelly = kelly_fracionado(prob_real, odd) if ev is not None and ev > 0 else None

        candidatos.append({
            "mercado": "Total de Gols da Partida",
            "selecao": f"{'Mais' if lado == 'over' else 'Menos'} de {linha} Gols",
            "odd": odd,
            "lambda_esperado_partida": lam_total,
            "probabilidade_real_calculada": prob_real,
            "probabilidade_implicita_odd": prob_implicita_odd,
            "delta_edge_pct_calculado": edge_pct,
            "ev": ev,
            "kelly_unidades_sugerido": kelly,
        })

    return candidatos


def montar_system_prompt_mie2(sport: str, analyst: str = "carlos") -> str:
    esporte_key = sport.lower()
    catalogo_esporte = REGRAS_ESPORTES.get(esporte_key, REGRAS_ESPORTES["futebol"])

    perfil = PERFIS_ANALISTA.get(analyst, PERFIS_ANALISTA["carlos"])
    delta_min = perfil["delta_min"]
    odd_min = perfil["odd_min"]
    odd_max = perfil["odd_max"]

    tabela_stake = "\n".join(
        f"   - Se {lo}% <= Δ < {hi}%: Stake {stake}."
        if hi != float("inf")
        else f"   - Se Δ >= {lo}%: Stake {stake}."
        for lo, hi, stake in perfil["faixas_stake"]
    )

    if analyst == "cris":
        persona_nome = "Cris (A Especialista em Tiro Certo)"
        persona_curto = "Cris"
        persona_regras = """- FILOSOFIA: Ultra-conservadora, focada primariamente na proteção implacável de banca.
- ANÁLISE: Rejeite qualquer risco desnecessário. Priorize apostas simples seguras ou duplas apenas com altíssima convicção.
- QUANDO HOUVER MAIS DE UMA LINHA POSSÍVEL NO MESMO MERCADO-BASE (ex: Under 1.5 e Under 2.5 de gols),
  PREFIRA SEMPRE A LINHA ESTATISTICAMENTE MAIS PROVÁVEL DE BATER (a mais alta em over, a mais alta em under),
  mesmo que isso signifique um delta_edge declarado menor. NUNCA escolha a linha mais extrema só para
  produzir um delta maior — isso é o oposto da sua filosofia.
- TOM DE VOZ: Sóbrio, direto, focado estritamente na mitigação de risco e proteção matemática do capital."""
    else:
        persona_nome = "Carlos (O Estrategista Técnico)"
        persona_curto = "Carlos"
        persona_regras = """- FILOSOFIA: Técnico, elegante e letal, atuando como um boxeador de elite no ringue do mercado financeiro esportivo.
- ANÁLISE: Varre os mercados em busca de valor oculto e assimetria que as casas de apostas não precificaram corretamente.
- TOM DE VOZ: Analítico, astuto, confiante, tático, usando o jargão de inteligência de mercado de forma fluida."""

    return f"""Você é {persona_nome}, e utiliza o Moneyball Intelligence Engine (MIE v2.5) como ferramenta quantitativa de alta precisão para a modalidade {sport.upper()}.

{persona_regras}

Você recebe a transcrição OCR de 2 a 5 prints contendo dados táticos, probabilísticos e odds reais capturadas. Sua missão é aplicar o funil quantitativo sobre TODOS os mercados presentes nos prints e devolver até 2 tomadas de decisão no formato JSON padronizado.

[1. MATRIZ OFICIAL DE MERCADOS]
Analise livremente os mercados permitidos dentro da modalidade {sport.upper()}:

{catalogo_esporte}

------------------------------------------------

[2. CLASSIFICAÇÃO DA HIPÓTESE DA PARTIDA]
Classifique a partida em EXCLUSIVAMENTE UMA das 3 hipóteses táticas:
1. TIPO A — PRODUÇÃO (Volume e Fluidez Distribuída): Ambas as partes contribuem.
2. TIPO B — DOMÍNIO (Superioridade e Controle): Um lado domina o resultado.
3. TIPO C — PRODUÇÃO ASSIMÉTRICA (Concentração Unilateral): Performance concentrada em um lado ou atleta.

------------------------------------------------

[3. FILTROS DE SEGURANÇA E REGRA DOS DOIS MELHORES EDGES]
0. DADOS JÁ CALCULADOS (se fornecidos abaixo, na mensagem do usuário, sob o
   cabeçalho "CANDIDATOS DE TOTAL DE GOLS JÁ CALCULADOS (Poisson)"):
   - Para o mercado de Total de Gols da Partida, o valor de Δ NÃO deve ser
     estimado por você — ele já vem calculado deterministicamente (Poisson)
     a partir de estatísticas reais buscadas na internet. Use EXATAMENTE
     o "delta_edge_pct_calculado", "odd" e "selecao" fornecidos ali, sem
     alterar nenhum número.
   - Você pode e deve continuar estimando Δ normalmente para os DEMAIS
     mercados (escanteios, cartões, props) — a regra acima vale só para
     Total de Gols quando os dados calculados estiverem presentes.
   - Se nenhum candidato de Total de Gols calculado for fornecido, estime
     esse mercado como os demais, com a desconfiança normal já prevista
     nas regras abaixo.

1. MARGEM DE SEGURANÇA (EDGE MÍNIMO - Δ_min = {delta_min}% para {persona_curto}):
   - A assimetria (Δ) é: Δ = Prob_Modelo - Prob_Odd.
   - SÓ É ELEGÍVEL QUALQUER SELEÇÃO COM Δ >= {delta_min}%. Este piso vale IGUALMENTE para Entrada 1 e Entrada 2 — não existe piso reduzido para a segunda vaga.
   - NUNCA declare um Δ acima de {DELTA_MAX_PLAUSIVEL}% a menos que a evidência nos prints seja explícita e inequívoca — deltas muito altos em mercados líquidos são raros e devem ser tratados com desconfiança, não como "grande oportunidade". Esta regra de teto NÃO se aplica ao Δ de Total de Gols quando ele vier do cálculo Poisson (item 0) — esse já é confiável por construção.
{tabela_stake}

2. SELEÇÃO DA DUPLA DE ELITE (LIVRE DE CATEGORIA):
   - Avalie TODOS os mercados extraídos de todos os prints fornecidos.
   - Entrada 1: A maior assimetria validada (Δ >= {delta_min}%) entre os mercados permitidos.
   - Entrada 2: A segunda maior assimetria validada (Δ >= {delta_min}%) entre os mercados permitidos.
   - UNICIDADE DE MERCADO: É ESTRITAMENTE PROIBIDO sugerir duas entradas do mesmo mercado base.
   - INDEPENDÊNCIA DE HIPÓTESE: Classifique cada seleção candidata como DEPENDENTE ou INDEPENDENTE da hipótese_partida.
     DEPENDENTE = só faz sentido se a leitura tática (TIPO A/B/C) estiver correta (ex: Over de Gols, Total de Pontos, Handicap ligado ao ritmo geral).
     INDEPENDENTE = resultado não depende da hipótese geral se sustentar (ex: escanteios de um time específico, cartões, prop isolado com padrão próprio).
     Se Entrada 1 for DEPENDENTE da hipótese_partida, a Entrada 2 DEVE ser INDEPENDENTE, se houver candidata elegível.
   - NUNCA invente seleções, linhas, atletas ou odds que não estejam explicitamente presentes nos prints.
   - NÃO FORCE PREENCHIMENTO: se não houver mercado elegível com Δ >= {delta_min}%, retorne "entrada_1" e/ou "entrada_2" como null. É preferível retornar vazio a sugerir uma seleção sem edge real ou correlacionada demais com a outra entrada.

3. JANELA DE ODDS ({persona_curto}): Cotações entre {odd_min} e {odd_max}.
   - Se as odds no JSON estiverem em formato americano (+120, -150), CONVERTA para decimal na saída.
   - Se não houver seleção elegível dentro desta janela, prefira retornar null a forçar uma odd fora do intervalo.

4. REGRA DO NOME EXPLÍCITO E COMPLETO:
   - É ESTRITAMENTE PROIBIDO retornar termos soltos como "Sim", "Não", "Mais" ou "Menos".
   - O campo "seleção" deve conter o nome completo e claro do mercado atrelado à escolha.

------------------------------------------------

[4. REGRAS DE BLOQUEIO E PROTEÇÃO - MONEYBALL 2.0]
1. BLOQUEIO TOTAL DE MONEYLINE (ML): Proibido sugerir vitória seca de equipes. Foque em mercados de volume, totais, handicaps ou estatísticas.
2. FILTRO ANTI-ESTRELA: Proibido sugerir apostas em favoritos abaixo de @1.50 sem linha de segurança robusta.
3. PROTEÇÃO CONTRA JOGOS TRUNCADOS: No Futebol, nunca sugira "Mais de 2.5 Gols" se houver indício de jogo travado e proíba props individuais de atletas (chutes/passes).
4. ISOLAMENTO DE CONTEXTO: Trate cada print de forma totalmente independente e valide a compatibilidade de esporte.

------------------------------------------------

[5. REGRA DE RETORNO JSON STRICT]
Sua resposta DEVE SER ESTRITAMENTE um JSON válido na estrutura exata abaixo, sem marcações markdown antes ou depois. "entrada_1" e "entrada_2" podem ser null se não houver candidata elegível — não force preenchimento.

{{
  "perfil_geral": "Síntese quantitativa da partida e leitura tática no tom de voz do analista...",
  "status_geral": "processado_com_sucesso",
  "hipotese_partida": "TIPO A | TIPO B | TIPO C",
  "stake_medio_partida": "1.0u",
  "match_info": {{
    "sport": "{sport.upper()}",
    "teams": "Time A vs Time B",
    "date": "Hoje"
  }},
  "expected_projections": {{
    "macro_projected": "Projeção relevante com delta",
    "micro_projected": "Projeção relevante com delta"
  }},
  "dupla_de_elite": {{
    "entrada_1": {{
      "categoria": "MACRO ou MICRO",
      "dependencia_hipotese": "DEPENDENTE ou INDEPENDENTE",
      "mercado": "Nome do Mercado",
      "selecao": "Seleção Explícita com Linha",
      "odd": "1.85",
      "delta_edge": "7.6%",
      "msc_score": 90,
      "stake_recomendada": "1.0u",
      "confiabilidade": "ALTA",
      "motivo": "Justificativa da entrada..."
    }},
    "entrada_2": null
  }},
  "key_asymmetries": [
    {{
      "clash": "Confronto ou Atleta Analisado",
      "statistical_evidence": "Evidência estatística extraída com delta",
      "betting_angle": "Direcionamento da aposta"
    }}
  ]
}}
"""


def _parse_float_seguro(valor) -> Optional[float]:
    """Tenta converter um valor (possivelmente string com '%', ',' etc) para float.
    Retorna None se não for possível — nunca lança exceção."""
    if valor is None:
        return None
    try:
        texto = str(valor).strip().replace("%", "").replace(",", ".")
        return float(texto)
    except (ValueError, TypeError):
        return None


def validar_e_sanear_entrada(entrada: Optional[dict], perfil: dict) -> Optional[dict]:
    """Confere a entrada retornada pelo LLM contra as regras objetivas do analista
    (janela de odds, piso de delta, teto de plausibilidade). Se qualquer regra for
    violada ou algum campo essencial estiver ausente/malformado, descarta a entrada
    (retorna None) em vez de deixar passar pro usuário. Isso NÃO depende do modelo
    ter obedecido a instrução em texto do prompt — é aplicado sempre, na força.
    """
    if not entrada or not isinstance(entrada, dict) or not entrada.get("mercado"):
        return None

    odd = _parse_float_seguro(entrada.get("odd"))
    delta = _parse_float_seguro(entrada.get("delta_edge"))

    if odd is None or delta is None:
        return None  # campo essencial ausente ou malformado

    if not (perfil["odd_min"] <= odd <= perfil["odd_max"]):
        return None  # violou a janela de odds do analista

    if delta < perfil["delta_min"]:
        return None  # abaixo do piso mínimo do analista

    if delta > DELTA_MAX_PLAUSIVEL:
        # Delta implausivelmente alto para mercados líquidos — provável estimativa
        # inventada pelo modelo, não edge real extraído dos prints.
        return None

    return entrada


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "engine": "MoneyballPro FastAPI v2.3 (com grounding real via Google Search)",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "groq_key_set": bool(GROQ_API_KEY),
        "perfis_analista": {
            nome: {"delta_min": p["delta_min"], "odd_min": p["odd_min"], "odd_max": p["odd_max"]}
            for nome, p in PERFIS_ANALISTA.items()
        },
    }


@app.post("/api/v1/calc")
async def calcular_mercados(payload: dict):
    """
    Camada de cálculo determinístico (Poisson + Kelly fracionado), sem LLM.
    Espera: { "mercados": [ {...}, {...} ] } — mesmo formato que o Léo
    definiu em mie2_calc.py. Cada mercado é calculado isoladamente; falta
    de dado em um não derruba os outros.
    """
    mercados = payload.get("mercados")
    if not mercados or not isinstance(mercados, list):
        raise HTTPException(
            status_code=400,
            detail='Corpo inválido. Esperado: { "mercados": [ {...}, {...} ] }'
        )
    return {"resultados": calcular_dossie(mercados)}


@app.post("/api/v1/analyze")
async def analyze_tickets(
    sport: str = Form(...),
    analyst: str = Form("carlos"),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    perfil = PERFIS_ANALISTA.get(analyst, PERFIS_ANALISTA["carlos"])

    # 1. OCR COM GEMINI
    gemini_client = get_gemini_client()
    contents = []

    for file in files:
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents.append(part)

    prompt_ocr = f"Extraia em PORTUGUÊS todo o texto, nomes de times, jogadores, confrontos, odds/cotações e linhas destas imagens para a modalidade: {sport}. Retorne apenas a transcrição direta do conteúdo presente nas imagens."
    contents.append(prompt_ocr)

    try:
        res_ocr = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            # temperature=0: a etapa de OCR é puramente extrativa (ler o que já
            # está escrito na imagem). Sem isso, o mesmo print pode ser lido de
            # formas diferentes em execuções diferentes — foi o que causou a
            # odd @2.27 vs @1.65 para o mesmo mercado no mesmo print.
            config=types.GenerateContentConfig(temperature=0)
        )
        texto_extraido_ocr = res_ocr.text.strip()
        # Log de auditoria: se uma odd/linha parecer errada depois, dá pra
        # conferir aqui se o erro nasceu na leitura do print (OCR) ou na
        # análise em cima do texto (Groq). Aparece nos logs da Vercel.
        print(f"[OCR DEBUG] sport={sport} analyst={analyst}\n{texto_extraido_ocr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no OCR (Gemini): {str(e)}")

    # 1b. GROUNDING REAL: extrai mercados de Total de Gols estruturados e busca
    #     médias reais dos times via Google Search, para alimentar o cálculo
    #     Poisson determinístico. Se qualquer etapa falhar ou vier incompleta,
    #     candidatos_gols_calculados fica vazio e o pipeline cai de volta pro
    #     Groq estimando esse mercado também — nunca quebra a análise inteira.
    candidatos_gols_calculados = []
    contexto_mie1_texto = ""
    try:
        dados_estruturados = extrair_time_e_mercados_gols(gemini_client, contents[:-1], sport)
        if dados_estruturados and dados_estruturados.get("mercados_gols"):
            time_a = dados_estruturados.get("time_a")
            time_b = dados_estruturados.get("time_b")
            if time_a and time_b:
                projecao_mie1 = executar_mie1(gemini_client, time_a, time_b, sport)
                if projecao_mie1:
                    candidatos_gols_calculados = montar_candidatos_gols_calculados(
                        dados_estruturados["mercados_gols"], projecao_mie1
                    )
                    partes_contexto = []
                    if projecao_mie1.get("contextual_factors"):
                        partes_contexto.append(
                            "FATORES CONTEXTUAIS (lesões, clima, desfalques — via MIE1):\n"
                            + json.dumps(projecao_mie1["contextual_factors"], ensure_ascii=False)
                        )
                    if projecao_mie1.get("key_asymmetries"):
                        partes_contexto.append(
                            "ASSIMETRIAS JÁ IDENTIFICADAS (via MIE1):\n"
                            + json.dumps(projecao_mie1["key_asymmetries"], ensure_ascii=False)
                        )
                    if projecao_mie1.get("fonte"):
                        partes_contexto.append(f"Fonte consultada: {projecao_mie1['fonte']}")
                    contexto_mie1_texto = "\n\n".join(partes_contexto)
    except Exception as e:
        # Grounding é um "bônus" sobre o fluxo já existente — se falhar, loga
        # e segue com a estimativa do Groq, não derruba a análise inteira.
        print(f"[GROUNDING PIPELINE ERRO] {str(e)}")
        candidatos_gols_calculados = []

    bloco_dados_calculados = ""
    if candidatos_gols_calculados:
        bloco_dados_calculados = (
            "\n\nCANDIDATOS DE TOTAL DE GOLS JÁ CALCULADOS (Poisson, com base em "
            "projeção real do MIE1 via busca em fontes de autoridade — use estes "
            "números EXATAMENTE como estão, não reestime):\n"
            + json.dumps(candidatos_gols_calculados, ensure_ascii=False, indent=2)
        )
    if contexto_mie1_texto:
        bloco_dados_calculados += f"\n\n{contexto_mie1_texto}"

    # 2. ANÁLISE QUANTITATIVA VIA GROQ COM GPT-OSS-120B
    groq_client = get_groq_client()
    system_instruction_mie2 = montar_system_prompt_mie2(sport=sport, analyst=analyst)

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_instruction_mie2},
                {"role": "user", "content": f"OCR DOS PRINTS:\n{texto_extraido_ocr}{bloco_dados_calculados}\n\nResponda APENAS com o JSON bruto, sem nenhuma explicação ou saudação."}
            ],
            temperature=0.1,
            max_tokens=4096
        )

        content_str = completion.choices[0].message.content or ""
        content_str = content_str.strip()

        if not content_str:
            raise HTTPException(status_code=500, detail="A Groq retornou uma resposta vazia com o modelo gpt-oss-120b.")

        # LIMPEZA UNIVERSAL (Isola perfeitamente o JSON entre chaves)
        match = re.search(r'\{.*\}', content_str, re.DOTALL)
        if match:
            content_str = match.group(0)

        if "```" in content_str:
            content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
            content_str = re.sub(r"\s*```$", "", content_str)
            content_str = content_str.strip()

        resultado = json.loads(content_str)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Erro de formatação JSON retornado pela IA: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento (Groq): {str(e)}")

    # 3. VALIDAÇÃO E SANEAMENTO — aplicado sempre, independente do modelo ter
    #    obedecido as regras do prompt. Isso é o que realmente impede odds fora
    #    da janela ou deltas implausíveis de chegarem até o usuário.
    dupla = resultado.get("dupla_de_elite", {}) or {}
    dupla["entrada_1"] = validar_e_sanear_entrada(dupla.get("entrada_1"), perfil)
    dupla["entrada_2"] = validar_e_sanear_entrada(dupla.get("entrada_2"), perfil)
    resultado["dupla_de_elite"] = dupla

    return resultado
