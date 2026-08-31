"""
MoneyballPro Engine -- ponto de entrada FastAPI.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types
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
    from utils import _parse_float_seguro
    from db import get_connection, fechar_conexao
    from projecao import obter_projecoes_partida
    from usuarios import checar_e_consumir_cota, sync_ghost_member, email_valido, LIMITE_CONSULTAS_FREE_DIARIO

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


@app.post("/api/webhooks/ghost")
async def webhook_ghost(payload: dict, request: Request):
    """
    Recebe eventos de member/subscription do Ghost e sincroniza com app_users.

    SEGURANÇA: o Ghost não permite adicionar headers customizados nas
    integrações custom, então a verificação de origem é feita via um token
    secreto embutido na PRÓPRIA URL de destino configurada no painel do Ghost
    -- não pela assinatura X-Ghost-Signature (o formato dela mudou algumas
    vezes ao longo do tempo e tem relatos frequentes de mismatch no fórum
    oficial do Ghost -- confiar nisso arriscaria rejeitar webhooks legítimos
    silenciosamente). A URL configurada no Ghost deve ser:
        https://SEU-DOMINIO/api/webhooks/ghost?token=SEU_GHOST_WEBHOOK_SECRET
    E a variável de ambiente GHOST_WEBHOOK_SECRET precisa estar configurada
    na Vercel com o mesmo valor.

    IMPORTANTE PRA CONFIGURAÇÃO NO PAINEL DO GHOST (Passo 3): o formato exato
    do payload pode variar um pouco entre eventos (`member.added` vs
    `member.updated`/`subscription`) e entre versões do Ghost. Este endpoint
    tenta extrair email/status de alguns formatos comuns, mas o ideal é:
    1. Configurar o webhook apontando pra cá (com o token na URL).
    2. Disparar um evento de teste real (o Ghost tem essa opção no painel da
       integração).
    3. Olhar os logs da Vercel (`[WEBHOOK GHOST] Payload recebido: ...`) pra
       confirmar que o formato bateu -- se não bateu, ajuste o parsing abaixo.

    Este endpoint SEMPRE responde 200 pra requisições com token válido, mesmo
    em erro interno de parsing -- o Ghost desativa webhooks automaticamente
    depois de falhas consecutivas, então nunca queremos que um payload
    inesperado derrube a integração inteira. Só requisições SEM o token
    correto são rejeitadas (401), o que é o comportamento esperado pra
    tentativas de forjar o webhook vindas de fora do Ghost.
    """
    token_esperado = os.getenv("GHOST_WEBHOOK_SECRET")
    if token_esperado:
        token_recebido = request.query_params.get("token")
        if token_recebido != token_esperado:
            print("[WEBHOOK GHOST] Token ausente ou inválido -- requisição rejeitada.")
            raise HTTPException(status_code=401, detail="Token inválido.")
    else:
        # GHOST_WEBHOOK_SECRET ainda não configurado na Vercel -- aceita mesmo
        # assim (pra não travar o Passo 3 antes de você configurar a variável),
        # mas avisa alto no log que está rodando sem proteção.
        print("[WEBHOOK GHOST] ATENÇÃO: GHOST_WEBHOOK_SECRET não configurado -- endpoint SEM proteção de token.")

    print(f"[WEBHOOK GHOST] Payload recebido: {json.dumps(payload, ensure_ascii=False)[:2000]}")

    try:
        member = (payload.get("member") or {}).get("current") or payload.get("member") or {}
        email = member.get("email")
        status = member.get("status")  # Ghost: 'free' | 'paid' | 'comped'
        ghost_member_id = member.get("id")

        if not email_valido(email):
            print(f"[WEBHOOK GHOST] E-mail ausente ou inválido no payload -- ignorando. member={member}")
            return {"ok": True, "processado": False, "motivo": "email ausente ou inválido"}

        plano = "pro" if status in ("paid", "comped") else "free"

        conn = get_connection()
        try:
            sucesso = sync_ghost_member(conn, email, ghost_member_id, plano)
        finally:
            fechar_conexao(conn)

        return {"ok": True, "processado": sucesso, "email": email, "plano": plano}

    except Exception as e:
        print(f"[WEBHOOK GHOST] Erro inesperado processando payload: {e}")
        return {"ok": True, "processado": False, "motivo": "erro interno -- ver logs"}


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
    """
    Protege endpoints chamados server-to-server (ex: SmartCenter), não por
    usuários finais do app. Mesmo padrão fail-open-com-aviso do webhook do
    Ghost: se a chave ainda não foi configurada na Vercel, deixa passar mas
    avisa alto no log -- pra não travar o Passo 1 da integração antes de você
    configurar a variável. Depois de configurada, passa a bloquear de verdade.
    """
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
    analyst: str = Form("carlos"),  # Carlos é o único analista do sistema (generalista)
    email: Optional[str] = Form(None),  # opcional por enquanto -- ver nota abaixo
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    # Freemium: checagem de cota ANTES de qualquer chamada cara ao Gemini/Groq,
    # senão gastamos orçamento de API em requisições que vão ser bloqueadas de
    # qualquer jeito. `email` é opcional por enquanto pra não quebrar o
    # frontend atual (que ainda não envia esse campo -- isso muda no Passo 4,
    # quando o app passa a pedir e-mail antes de analisar). Sem e-mail, o
    # comportamento antigo é preservado (sem checagem de cota nenhuma).
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
                # Score de Convergência (Framework Mestre Parte 3) -- sempre calculável
                # (nunca None), mesmo que roteiro/matchup individualmente sejam None,
                # porque a própria ausência de sinal já é informação (fica NEUTRO).
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
    # Auditoria: Score de Convergência (Framework Mestre Parte 3) -- teto de
    # unidades sugerido a partir da convergência entre roteiro e matchup
    resultado_final["convergencia_calculada_python"] = convergencia_calculada
    # Freemium: info de cota consumida nesta chamada (None se email não foi
    # enviado -- ver nota no início do endpoint)
    resultado_final["cota_info"] = cota_info

    if resultado_final.get("dupla_de_elite"):
        e1 = resultado_final["dupla_de_elite"].get("entrada_1")
        e2 = resultado_final["dupla_de_elite"].get("entrada_2")

        resultado_final["dupla_de_elite"]["entrada_1"] = validar_e_sanear_entrada(e1, perfil)
        resultado_final["dupla_de_elite"]["entrada_2"] = validar_e_sanear_entrada(e2, perfil)

        # MSC reformulado -- ajusta o MSC base (matemática isolada, já veio no
        # msc_score de cada entrada) pelo nível de Convergência da partida, e
        # traduz o resultado num rótulo pro usuário -- ver calc.py pro
        # racional completo. Feito aqui (não em candidatos.py) porque
        # Convergência é uma leitura de partida inteira, só existe depois que
        # sabemos QUAL entrada o Carlos escolheu.
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

        # Aposta combinada (bet builder/múltipla única) -- só faz sentido
        # calcular quando as DUAS entradas sobreviveram à validação. A
        # matemática é 100% Python (regra de ouro do projeto) -- reconstrói a
        # probabilidade de cada perna a partir de odd + delta_edge, já que o
        # JSON do Carlos não retorna a probabilidade bruta diretamente.
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

    return resultado_final
