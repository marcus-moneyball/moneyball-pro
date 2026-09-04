"""
MoneyballPro Engine -- ponto de entrada FastAPI.
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types
from google.genai.errors import ServerError
from groq import Groq

try:
    from api.catalogos import PERFIS_ANALISTA, CONFIG_MERCADO_PRINCIPAL
    from api.calc import (
        calcular_dossie, classificar_roteiro_jogo, calcular_matchup, calcular_convergencia,
        calcular_aposta_combinada, ajustar_msc_por_convergencia, rotulo_confianca,
    )
    from api.mie1_gemini import get_gemini_client, extrair_mercados_estruturados, executar_mie1
    from api.candidatos import (
        montar_candidatos_over_under_calculados, montar_candidato_btts,
        montar_candidato_moneyline, montar_candidatos_chance_dupla, montar_candidatos_handicap_asiatico,
    )
    from api.prompts_mie2 import montar_system_prompt_mie2
    from api.validacao import validar_e_sanear_entrada
    from api.utils import _parse_float_seguro
    from api.db import get_connection, fechar_conexao
    from api.projecao import obter_projecoes_partida
    from api.usuarios import checar_e_consumir_cota, sync_ghost_member, email_valido, LIMITE_CONSULTAS_FREE_DIARIO
    from api.notificacoes_telegram import publicar_recomendacao_publica, publicar_ouvidoria
    from api.utils import gerar_codigo_auditoria
    from api.telegram_membros import (
        gerar_token_vinculo, montar_link_vinculo, consumir_token_vinculo,
        salvar_telegram_user_id, obter_plano_e_telegram,
        enviar_convite_grupo_pro, remover_do_grupo_pro,
    )
except ImportError:
    from catalogos import PERFIS_ANALISTA, CONFIG_MERCADO_PRINCIPAL
    from calc import (
        calcular_dossie, classificar_roteiro_jogo, calcular_matchup, calcular_convergencia,
        calcular_aposta_combinada, ajustar_msc_por_convergencia, rotulo_confianca,
    )
    from mie1_gemini import get_gemini_client, extrair_mercados_estruturados, executar_mie1
    from candidatos import (
        montar_candidatos_over_under_calculados, montar_candidato_btts,
        montar_candidato_moneyline, montar_candidatos_chance_dupla, montar_candidatos_handicap_asiatico,
    )
    from prompts_mie2 import montar_system_prompt_mie2
    from validacao import validar_e_sanear_entrada
    from utils import _parse_float_seguro, gerar_codigo_auditoria
    from db import get_connection, fechar_conexao
    from projecao import obter_projecoes_partida
    from usuarios import checar_e_consumir_cota, sync_ghost_member, email_valido, LIMITE_CONSULTAS_FREE_DIARIO
    from notificacoes_telegram import publicar_recomendacao_publica, publicar_ouvidoria
    from telegram_membros import (
        gerar_token_vinculo, montar_link_vinculo, consumir_token_vinculo,
        salvar_telegram_user_id, obter_plano_e_telegram,
        enviar_convite_grupo_pro, remover_do_grupo_pro,
    )


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


def chamar_gemini_resiliente(gemini_client, contents, config):
    """Função auxiliar para tratar instabilidades 503 do Gemini com retry e fallback."""
    modelos = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
    
    for model_name in modelos:
        for tentativa in range(3):
            try:
                return gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except ServerError as e:
                if e.code == 503:
                    time.sleep(1.5 * (tentativa + 1))
                    continue
                raise e
            except Exception as e:
                raise e

    raise HTTPException(
        status_code=503, 
        detail="O motor de IA está temporariamente sobrecarregado. Tente novamente em instantes."
    )


@app.post("/api/webhooks/ghost")
async def webhook_ghost(payload: dict, request: Request):
    token_esperado = os.getenv("GHOST_WEBHOOK_SECRET")
    if token_esperado:
        token_recebido = request.query_params.get("token")
        if token_recebido != token_esperado:
            print("[WEBHOOK GHOST] Token ausente ou inválido -- requisição rejeitada.")
            raise HTTPException(status_code=401, detail="Token inválido.")
    else:
        print("[WEBHOOK GHOST] ATENÇÃO: GHOST_WEBHOOK_SECRET não configurado -- endpoint SEM proteção de token.")

    print(f"[WEBHOOK GHOST] Payload recebido: {json.dumps(payload, ensure_ascii=False)[:2000]}")

    try:
        member = (payload.get("member") or {}).get("current") or payload.get("member") or {}
        email = member.get("email")
        status = member.get("status")
        ghost_member_id = member.get("id")

        if not email_valido(email):
            print(f"[WEBHOOK GHOST] E-mail ausente ou inválido no payload -- ignorando. member={member}")
            return {"ok": True, "processado": False, "motivo": "email ausente ou inválido"}

        plano = "pro" if status in ("paid", "comped") else "free"

        conn = get_connection()
        try:
            sync_info = sync_ghost_member(conn, email, ghost_member_id, plano)

            if sync_info["sucesso"] and sync_info["plano_anterior"] == "pro" and plano == "free":
                info = obter_plano_e_telegram(conn, email)
                if info.get("telegram_user_id"):
                    try:
                        remover_do_grupo_pro(info["telegram_user_id"])
                    except Exception as e:
                        print(f"[WEBHOOK GHOST] Falha ao remover '{email}' do grupo Telegram: {e}")
        finally:
            fechar_conexao(conn)

        return {"ok": True, "processado": sync_info["sucesso"], "email": email, "plano": plano}

    except Exception as e:
        print(f"[WEBHOOK GHOST] Erro inesperado processando payload: {e}")
        return {"ok": True, "processado": False, "motivo": "erro interno -- ver logs"}


@app.post("/api/webhooks/telegram")
async def webhook_telegram(update: dict):
    try:
        mensagem = update.get("message") or {}
        texto = (mensagem.get("text") or "").strip()
        from_user = mensagem.get("from") or {}
        telegram_user_id = from_user.get("id")

        if not texto.startswith("/start") or not telegram_user_id:
            return {"ok": True, "processado": False}

        partes = texto.split(maxsplit=1)
        if len(partes) < 2:
            return {"ok": True, "processado": False, "motivo": "sem token"}
        token = partes[1].strip()

        conn = get_connection()
        try:
            email = consumir_token_vinculo(conn, token)
            if not email:
                return {"ok": True, "processado": False, "motivo": "token inválido ou já usado"}

            info = obter_plano_e_telegram(conn, email)
            if info.get("plano") != "pro":
                return {"ok": True, "processado": False, "motivo": "e-mail não é PRO"}

            salvar_telegram_user_id(conn, email, telegram_user_id)
        finally:
            fechar_conexao(conn)

        enviar_convite_grupo_pro(telegram_user_id)
        return {"ok": True, "processado": True, "email": email}

    except Exception as e:
        print(f"[WEBHOOK TELEGRAM] Erro inesperado: {e}")
        return {"ok": True, "processado": False, "motivo": "erro interno -- ver logs"}


@app.get("/api/v1/membro/status")
def status_membro(email: str):
    if not email_valido(email):
        raise HTTPException(status_code=400, detail="E-mail inválido.")

    conn = get_connection()
    try:
        info = obter_plano_e_telegram(conn, email)
        resultado = {
            "plano": info["plano"],
            "telegram_vinculado": bool(info.get("telegram_user_id")),
            "link_vinculo_telegram": None,
        }
        if info["plano"] == "pro" and not info.get("telegram_user_id"):
            token = gerar_token_vinculo(conn, email)
            resultado["link_vinculo_telegram"] = montar_link_vinculo(token) if token else None
        return resultado
    finally:
        fechar_conexao(conn)


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


SMARTCENTER_SERVICE_KEY = os.getenv("SMARTCENTER_SERVICE_KEY")


def _validar_chave_servico(request: Request):
    if not SMARTCENTER_SERVICE_KEY:
        print("[SERVICE AUTH] ATENÇÃO: SMARTCENTER_SERVICE_KEY não configurada -- endpoint SEM proteção.")
        return
    chave_recebida = request.headers.get("x-service-key")
    if chave_recebida != SMARTCENTER_SERVICE_KEY:
        raise HTTPException(status_code=401, detail="Chave de serviço inválida ou ausente.")


@app.post("/api/v1/calc")
async def calcular_mercados(payload: dict, request: Request):
    _validar_chave_servico(request)
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
    analyst: str = Form("carlos"),
    email: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    cota_info = None
    if email:
        if not email_valido(email):
            raise HTTPException(status_code=400, detail="E-mail inválido.")
        db_conn_cota = get_connection()
        try:
            cota_info = checar_e_consumir_cota(db_conn_cota, email)
        finally:
            fechar_conexao(db_conn_cota)

        if not cota_info["permitido"]:
            raise HTTPException(
                status_code=402,
                detail={
                    "cota_excedida": True,
                    "mensagem": "Limite diário de análises gratuitas atingido.",
                    "plano": cota_info["plano"],
                    "consultas_hoje": cota_info["consultas_hoje"],
                    "limite": cota_info["limite"],
                },
            )

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
    convergencia_calculada = None

    if dados_estruturados and dados_estruturados.get("time_a") and dados_estruturados.get("time_b"):
        time_a = dados_estruturados["time_a"]
        time_b = dados_estruturados["time_b"]

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
            mie1_data = projecoes.get("mie1_data")
            fonte_projecao = projecoes["fonte"]
            lam_a = projecoes["lam_a"]
            lam_b = projecoes["lam_b"]
            fatores_incerteza = mie1_data.get("contextual_factors", []) if mie1_data else []

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
                convergencia_calculada = calcular_convergencia(roteiro_classificado, matchup_calculado)

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

                if sport.lower() == "futebol":
                    candidatos_calculados.extend(
                        montar_candidatos_chance_dupla(
                            dados_estruturados.get("mercado_chance_dupla"), lam_a, lam_b,
                            persona=analista_key, fatores_incerteza=fatores_incerteza,
                        )
                    )
                    candidatos_calculados.extend(
                        montar_candidatos_handicap_asiatico(
                            dados_estruturados.get("mercados_handicap_asiatico"), lam_a, lam_b,
                            persona=analista_key, fatores_incerteza=fatores_incerteza,
                        )
                    )
                elif sport.lower() in ("basquete", "beisebol"):
                    nome_time_a = dados_estruturados.get("time_a", "Time A")
                    nome_time_b = dados_estruturados.get("time_b", "Time B")
                    candidatos_calculados.extend(
                        montar_candidato_moneyline(
                            dados_estruturados.get("mercado_moneyline"), lam_a, lam_b,
                            esporte=sport, nome_time_a=nome_time_a, nome_time_b=nome_time_b,
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

    if convergencia_calculada:
        user_prompt_content += f"\n\n[CONVERGÊNCIA JÁ CALCULADA PELO PYTHON]\n" + json.dumps(convergencia_calculada, indent=2, ensure_ascii=False)

    # Chamada resiliente para o Gemini OCR
    ocr_res = chamar_gemini_resiliente(
        gemini_client=gemini_client,
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
        temperature=0.0,
        top_p=0.1,
        response_format={"type": "json_object"}
    )

    resultado_final = json.loads(groq_response.choices[0].message.content)

    for chave_lixeira in ["analise_macro", "analise_micro", "macro", "micro", "analise"]:
        resultado_final.pop(chave_lixeira, None)

    resultado_final["fonte_projecao"] = fonte_projecao
    resultado_final["casa_apostas"] = dados_estruturados.get("casa_apostas") if dados_estruturados else None
    resultado_final["roteiro_classificado_python"] = roteiro_classificado
    resultado_final["matchup_calculado_python"] = matchup_calculado
    resultado_final["convergencia_calculada_python"] = convergencia_calculada
    resultado_final["cota_info"] = cota_info

    if resultado_final.get("dupla_de_elite"):
        e1 = resultado_final["dupla_de_elite"].get("entrada_1")
        e2 = resultado_final["dupla_de_elite"].get("entrada_2")

        resultado_final["dupla_de_elite"]["entrada_1"] = validar_e_sanear_entrada(e1, perfil)
        resultado_final["dupla_de_elite"]["entrada_2"] = validar_e_sanear_entrada(e2, perfil)

        nivel_convergencia = convergencia_calculada.get("nivel") if convergencia_calculada else None
        for chave_entrada in ("entrada_1", "entrada_2"):
            entrada_atual = resultado_final["dupla_de_elite"].get(chave_entrada)
            if entrada_atual and entrada_atual.get("msc_score") is not None:
                msc_base = _parse_float_seguro(entrada_atual.get("msc_score"))
                msc_ajustado = ajustar_msc_por_convergencia(msc_base, nivel_convergencia) if msc_base is not None else None
                entrada_atual["confianca_exibicao"] = {
                    "score": msc_ajustado,
                    "rotulo": rotulo_confianca(msc_ajustado),
                } if msc_ajustado is not None else None

        e1_valida = resultado_final["dupla_de_elite"]["entrada_1"]
        e2_valida = resultado_final["dupla_de_elite"]["entrada_2"]
        resultado_final["dupla_de_elite"]["aposta_combinada"] = None

        if e1_valida and e2_valida:
            odd_1 = _parse_float_seguro(e1_valida.get("odd"))
            delta_1 = _parse_float_seguro(e1_valida.get("delta_edge"))
            odd_2 = _parse_float_seguro(e2_valida.get("odd"))
            delta_2 = _parse_float_seguro(e2_valida.get("delta_edge"))

            if odd_1 and odd_2 and delta_1 is not None and delta_2 is not None:
                prob_1 = round(1 / odd_1 + delta_1 / 100, 4)
                prob_2 = round(1 / odd_2 + delta_2 / 100, 4)
                teto_convergencia = (
                    convergencia_calculada.get("teto_stake_unidades", 1.0)
                    if convergencia_calculada else 1.0
                )
                resultado_final["dupla_de_elite"]["aposta_combinada"] = calcular_aposta_combinada(
                    prob_1, odd_1, prob_2, odd_2, teto_stake_convergencia=teto_convergencia,
                )

    resultado_final["codigo_auditoria"] = gerar_codigo_auditoria()

    try:
        publicar_ouvidoria(resultado_final)
    except Exception as e:
        print(f"[TELEGRAM] Falha inesperada ao publicar na ouvidoria: {e}")

    return resultado_final
