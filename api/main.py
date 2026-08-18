"""
MoneyballPro Engine -- ponto de entrada FastAPI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types
from groq import Groq

try:
    from api.catalogos import PERFIS_ANALISTA, CONFIG_MERCADO_PRINCIPAL
    from api.calc import calcular_dossie, classificar_roteiro_jogo, calcular_matchup
    from api.mie1_gemini import get_gemini_client, extrair_mercados_estruturados, executar_mie1
    from api.candidatos import montar_candidatos_over_under_calculados, montar_candidato_btts
    from api.prompts_mie2 import montar_system_prompt_mie2
    from api.validacao import validar_e_sanear_entrada
    from api.db import get_connection, fechar_conexao
    from api.projecao import obter_projecoes_partida
except ImportError:
    from catalogos import PERFIS_ANALISTA, CONFIG_MERCADO_PRINCIPAL
    from calc import calcular_dossie, classificar_roteiro_jogo, calcular_matchup
    from mie1_gemini import get_gemini_client, extrair_mercados_estruturados, executar_mie1
    from candidatos import montar_candidatos_over_under_calculados, montar_candidato_btts
    from prompts_mie2 import montar_system_prompt_mie2
    from validacao import validar_e_sanear_entrada
    from db import get_connection, fechar_conexao
    from projecao import obter_projecoes_partida

import os

app = FastAPI(title="MoneyballPro Engine", version="2.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_groq_client():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada na Vercel.")
    return Groq(api_key=GROQ_API_KEY)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "engine": "MoneyballPro FastAPI v2.6.0",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "groq_key_set": bool(GROQ_API_KEY),
        "perfis_analista": {
            nome: {"delta_min": p["delta_min"], "odd_min": p["odd_min"], "odd_max": p["odd_max"]}
            for nome, p in PERFIS_ANALISTA.items()
        },
    }


@app.post("/api/v1/calc")
async def calcular_mercados(payload: dict):
    mercados = payload.get("mercados")
    esporte = payload.get("esporte", "futebol")
    if not mercados or not isinstance(mercados, list):
        raise HTTPException(
            status_code=400,
            detail='Corpo inválido. Esperado: { "esporte": "basquete", "mercados": [ {...} ] }'
        )
    return {"resultados": calcular_dossie(mercados, esporte=esporte)}


@app.post("/api/v1/analyze")
async def analyze_tickets(
    sport: str = Form(...),
    analyst: str = Form("carlos"),  # Carlos é o único analista do sistema (generalista)
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    analista_key = analyst.lower() if analyst.lower() in PERFIS_ANALISTA else "carlos"
    perfil = PERFIS_ANALISTA[analista_key]

    gemini_client = get_gemini_client()
    contents = []

    for file in files:
        file_bytes = await file.read()
        contents.append(
            types.Part.from_bytes(
                data=file_bytes,
                mime_type=file.content_type or "image/jpeg",
            )
        )

    dados_estruturados = extrair_mercados_estruturados(gemini_client, contents, sport)

    candidatos_calculados = []
    mie1_data = None
    fonte_projecao = None
    roteiro_classificado = None
    matchup_calculado = None

    if dados_estruturados and dados_estruturados.get("time_a") and dados_estruturados.get("time_b"):
        time_a = dados_estruturados["time_a"]
        time_b = dados_estruturados["time_b"]

        # Cascata: banco local (só se os DOIS times estiverem lá) -> Gemini pros dois.
        # Nunca mistura banco com Gemini no mesmo confronto (ver projecao.py).
        db_conn = get_connection()
        try:
            projecoes = obter_projecoes_partida(
                time_a, time_b, sport, competicao=None,
                conn=db_conn, gemini_client=gemini_client,
                executar_mie1_fn=executar_mie1,
            )
        finally:
            fechar_conexao(db_conn)

        if projecoes:
            mie1_data = projecoes.get("mie1_data")  # só existe se veio do Gemini
            fonte_projecao = projecoes["fonte"]
            lam_a = projecoes["lam_a"]
            lam_b = projecoes["lam_b"]
            fatores_incerteza = mie1_data.get("contextual_factors", []) if mie1_data else []

            # Metodologia Nexus Cap. V (roteiro) + Framework Mestre Pilar 1 (matchup) --
            # ambos só são classificáveis de forma determinística quando os dados vieram
            # do MIE1 (Gemini com grounding); o banco local ainda não tem xG/pace/PPDA/
            # platoon splits/etc., só média marcada/sofrida.
            if mie1_data:
                roteiro_classificado = classificar_roteiro_jogo(
                    sport,
                    mie1_data.get("team_a_roteiro"),
                    mie1_data.get("team_b_roteiro"),
                )
                matchup_calculado = calcular_matchup(
                    sport,
                    mie1_data.get("team_a_roteiro"),
                    mie1_data.get("team_b_roteiro"),
                )

            if lam_a is not None and lam_b is not None:
                lam_total = lam_a + lam_b
                cfg = CONFIG_MERCADO_PRINCIPAL.get(sport.lower(), CONFIG_MERCADO_PRINCIPAL["futebol"])

                candidatos_calculados.extend(
                    montar_candidatos_over_under_calculados(
                        dados_estruturados.get("mercados_total_principal", []),
                        lam_total,
                        cfg["nome_mercado"],
                        cfg["unidade_selecao"],
                        esporte=sport,
                        persona=analista_key,
                        fatores_incerteza=fatores_incerteza,
                    )
                )

                if sport.lower() == "futebol":
                    cantos_a = (mie1_data or {}).get("team_a_escanteios_projected")
                    cantos_b = (mie1_data or {}).get("team_b_escanteios_projected")
                    if cantos_a and cantos_b:
                        candidatos_calculados.extend(
                            montar_candidatos_over_under_calculados(
                                dados_estruturados.get("mercados_escanteios", []),
                                cantos_a + cantos_b,
                                "Total de Escanteios da Partida",
                                "Escanteios",
                                esporte=sport,
                                persona=analista_key,
                                fatores_incerteza=fatores_incerteza,
                            )
                        )

                    cartoes_a = (mie1_data or {}).get("team_a_cartoes_projected")
                    cartoes_b = (mie1_data or {}).get("team_b_cartoes_projected")
                    if cartoes_a and cartoes_b:
                        candidatos_calculados.extend(
                            montar_candidatos_over_under_calculados(
                                dados_estruturados.get("mercados_cartoes", []),
                                cartoes_a + cartoes_b,
                                "Total de Cartões da Partida",
                                "Cartões",
                                esporte=sport,
                                persona=analista_key,
                                fatores_incerteza=fatores_incerteza,
                            )
                        )

                candidatos_calculados.extend(
                    montar_candidato_btts(
                        dados_estruturados.get("mercado_btts"), lam_a, lam_b,
                        persona=analista_key, fatores_incerteza=fatores_incerteza,
                    )
                )

    groq_client = get_groq_client()
    system_prompt = montar_system_prompt_mie2(sport, analista_key)

    user_prompt_content = "Analise os seguintes dados visuais dos tickets de apostas fornecidos e extraia o valor. Retorne APENAS o JSON limpo."

    if candidatos_calculados:
        user_prompt_content += f"\n\n[CANDIDATOS JÁ CALCULADOS PELO PYTHON]\n" + json.dumps(candidatos_calculados, indent=2, ensure_ascii=False)

    if mie1_data:
        contexto_mie1 = {"key_asymmetries": (mie1_data or {}).get("key_asymmetries", [])}
        if contexto_mie1["key_asymmetries"]:
            user_prompt_content += f"\n\n[DADOS DE ASSIMETRIAS DA PESQUISA WEB (MIE1)]\n" + json.dumps(contexto_mie1, indent=2, ensure_ascii=False)

    if roteiro_classificado:
        user_prompt_content += f"\n\n[ROTEIRO JÁ CLASSIFICADO PELO PYTHON]\n" + json.dumps(roteiro_classificado, indent=2, ensure_ascii=False)

    if matchup_calculado and matchup_calculado.get("matchup_detectado"):
        user_prompt_content += f"\n\n[MATCHUP JÁ CALCULADO PELO PYTHON]\n" + json.dumps(matchup_calculado, indent=2, ensure_ascii=False)

    ocr_res = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents + ["Transcreva de forma limpa e estruturada todo o texto e números visíveis nestes prints."],
        config=types.GenerateContentConfig(temperature=0)
    )
    texto_ocr = ocr_res.text or ""

    groq_response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt_content}\n\n[TRANSCRIÇÃO DOS PRINTS]\n{texto_ocr}"}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    resultado_final = json.loads(groq_response.choices[0].message.content)

    for chave_lixeira in ["analise_macro", "analise_micro", "macro", "micro", "analise"]:
        resultado_final.pop(chave_lixeira, None)

    # Auditoria: de onde veio a projeção de expectativa (banco local ou busca do Gemini)
    resultado_final["fonte_projecao"] = fonte_projecao
    # Auditoria: roteiro determinístico calculado pelo Python (None se não houve
    # dado de grounding suficiente pro esporte -- ver Metodologia Nexus Cap. V)
    resultado_final["roteiro_classificado_python"] = roteiro_classificado
    # Auditoria: matchup determinístico calculado pelo Python (None se não houve
    # dado suficiente; matchup_detectado=False se houve dado mas nenhum sinal --
    # ver Framework Mestre da Análise Esportiva, Pilar 1)
    resultado_final["matchup_calculado_python"] = matchup_calculado

    if resultado_final.get("dupla_de_elite"):
        e1 = resultado_final["dupla_de_elite"].get("entrada_1")
        e2 = resultado_final["dupla_de_elite"].get("entrada_2")

        resultado_final["dupla_de_elite"]["entrada_1"] = validar_e_sanear_entrada(e1, perfil)
        resultado_final["dupla_de_elite"]["entrada_2"] = validar_e_sanear_entrada(e2, perfil)

    return resultado_final
