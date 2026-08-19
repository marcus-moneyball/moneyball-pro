"""
Gestão de usuários do app -- sincronizados a partir do Ghost via webhook.

A fonte de verdade de "quem é Pro" é o Ghost (membros/assinaturas via Stripe).
A tabela `app_users` no Postgres é só um CACHE local dessa informação, pra não
precisar chamar a Admin API do Ghost a cada análise -- o webhook (ver
sync_ghost_member) mantém esse cache atualizado.

Reset de cota diária: não usa cron job nenhum. A cota é resetada de forma
"preguiçosa" -- no momento da consulta, se `data_ultima_consulta` for
diferente de hoje, zera o contador antes de checar. Isso elimina a necessidade
de manter uma tarefa agendada rodando na Vercel só pra isso.
"""

import re
from datetime import date, datetime
from typing import Optional

LIMITE_CONSULTAS_FREE_DIARIO = 3  # ajustável -- é só esse número que muda o freemium

_REGEX_EMAIL_SIMPLES = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def email_valido(email: Optional[str]) -> bool:
    """Validação leve, só pra barrar lixo óbvio (sem @, sem domínio) --
    não é RFC 5322 completo de propósito, isso é responsabilidade do Ghost
    na hora do cadastro real."""
    if not email or not isinstance(email, str):
        return False
    return bool(_REGEX_EMAIL_SIMPLES.match(email.strip()))


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg2")


def obter_ou_criar_usuario(conn, email: str) -> dict:
    """
    Busca o usuário por e-mail. Se não existir ainda (ex: alguém digitou um
    e-mail que se cadastrou no Ghost há poucos segundos e o webhook ainda não
    chegou, ou nunca vai se cadastrar e só quer testar o app), CRIA a linha
    com plano='free' na hora -- o app nunca trava esperando o webhook, só
    trata como gratuito até a sincronização real chegar (que, quando chegar,
    faz UPDATE em cima dessa linha via sync_ghost_member).
    """
    email = email.strip().lower()
    cursor = conn.cursor()
    placeholder = "%s" if _is_postgres(conn) else "?"

    cursor.execute(
        f"SELECT email, plano, consultas_hoje, data_ultima_consulta FROM app_users WHERE email = {placeholder}",
        (email,),
    )
    row = cursor.fetchone()

    if row:
        return {
            "email": row[0],
            "plano": row[1],
            "consultas_hoje": row[2],
            "data_ultima_consulta": row[3],
        }

    hoje = date.today()
    cursor.execute(
        f"""
        INSERT INTO app_users (email, plano, consultas_hoje, data_ultima_consulta)
        VALUES ({placeholder}, 'free', 0, {placeholder})
        """,
        (email, hoje),
    )
    conn.commit()
    return {"email": email, "plano": "free", "consultas_hoje": 0, "data_ultima_consulta": hoje}


def checar_e_consumir_cota(conn, email: str, limite_free: int = LIMITE_CONSULTAS_FREE_DIARIO) -> dict:
    """
    Verifica se `email` ainda tem cota hoje e, se tiver, JÁ CONSOME uma
    unidade (incrementa consultas_hoje) -- isso deve ser chamado ANTES de
    qualquer chamada cara ao Gemini/Groq, nunca depois, pra não gastar
    orçamento de API em requisições que vão ser bloqueadas de qualquer jeito.

    Se `conn` for None (Postgres não configurado, ou falha de conexão), o
    comportamento é FAIL OPEN -- libera a consulta sem contar cota. Isso é
    intencional: o gate de freemium é uma decisão de produto, não uma
    fronteira de segurança, então uma falha de banco não deveria derrubar o
    produto inteiro pros usuários pagantes também.

    Retorna:
    {
        "permitido": bool,
        "plano": "free" | "pro",
        "consultas_hoje": int,   # já incluindo esta consulta, se permitido
        "limite": int | None,   # None quando plano == "pro" (ilimitado)
    }
    """
    if conn is None:
        return {"permitido": True, "plano": "free", "consultas_hoje": 0, "limite": None}

    try:
        usuario = obter_ou_criar_usuario(conn, email)
        hoje = date.today()

        # Reset preguiçoso: se a última consulta foi em outro dia, zera antes de checar.
        data_ultima = usuario["data_ultima_consulta"]
        if isinstance(data_ultima, str):
            data_ultima = datetime.strptime(data_ultima, "%Y-%m-%d").date()
        if data_ultima != hoje:
            usuario["consultas_hoje"] = 0

        if usuario["plano"] == "pro":
            # Pro é ilimitado -- ainda assim atualiza data_ultima_consulta pra
            # manter o registro "vivo", mas não bloqueia nem conta contra limite.
            _atualizar_contagem(conn, email, consultas_hoje=usuario["consultas_hoje"], data=hoje)
            return {"permitido": True, "plano": "pro", "consultas_hoje": usuario["consultas_hoje"], "limite": None}

        if usuario["consultas_hoje"] >= limite_free:
            # Já bateu o limite -- NÃO incrementa (não teve consulta real).
            _atualizar_contagem(conn, email, consultas_hoje=usuario["consultas_hoje"], data=hoje)
            return {"permitido": False, "plano": "free", "consultas_hoje": usuario["consultas_hoje"], "limite": limite_free}

        nova_contagem = usuario["consultas_hoje"] + 1
        _atualizar_contagem(conn, email, consultas_hoje=nova_contagem, data=hoje)
        return {"permitido": True, "plano": "free", "consultas_hoje": nova_contagem, "limite": limite_free}

    except Exception as e:
        print(f"[COTA] Falha ao checar cota para '{email}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return {"permitido": True, "plano": "free", "consultas_hoje": 0, "limite": None}


def _atualizar_contagem(conn, email: str, consultas_hoje: int, data: date):
    placeholder = "%s" if _is_postgres(conn) else "?"
    clausula_timestamp = "NOW()" if _is_postgres(conn) else "CURRENT_TIMESTAMP"
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE app_users
        SET consultas_hoje = {placeholder}, data_ultima_consulta = {placeholder}, atualizado_em = {clausula_timestamp}
        WHERE email = {placeholder}
        """,
        (consultas_hoje, data, email.strip().lower()),
    )
    conn.commit()


def sync_ghost_member(conn, email: str, ghost_member_id: Optional[str], plano: str) -> bool:
    """
    Upsert vindo do webhook do Ghost -- cria a linha se não existir, ou
    atualiza o plano se já existir. `plano` já deve vir normalizado como
    "free" ou "pro" (ver mapear_status_ghost_para_plano em main.py, que
    traduz o campo "status" do member do Ghost: 'free' -> 'free',
    'paid'/'comped' -> 'pro').

    Retorna True se aplicou com sucesso, False se falhou (nunca lança
    exceção -- o endpoint de webhook precisa responder 200 pro Ghost mesmo
    quando algo dá errado aqui, senão o Ghost desativa o webhook depois de
    falhas consecutivas).
    """
    if conn is None or not email_valido(email):
        return False

    email = email.strip().lower()
    placeholder = "%s" if _is_postgres(conn) else "?"

    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            INSERT INTO app_users (email, ghost_member_id, plano, consultas_hoje, data_ultima_consulta)
            VALUES ({placeholder}, {placeholder}, {placeholder}, 0, {placeholder})
            ON CONFLICT (email) DO UPDATE SET
                ghost_member_id = EXCLUDED.ghost_member_id,
                plano = EXCLUDED.plano,
                atualizado_em = NOW()
            """ if _is_postgres(conn) else
            f"""
            INSERT INTO app_users (email, ghost_member_id, plano, consultas_hoje, data_ultima_consulta)
            VALUES ({placeholder}, {placeholder}, {placeholder}, 0, {placeholder})
            ON CONFLICT(email) DO UPDATE SET
                ghost_member_id = excluded.ghost_member_id,
                plano = excluded.plano
            """,
            (email, ghost_member_id, plano, date.today()),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[GHOST SYNC] Falha ao sincronizar '{email}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
