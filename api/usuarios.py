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
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

LIMITE_CONSULTAS_FREE_DIARIO = 5
TZ_BRASIL = ZoneInfo("America/Sao_Paulo")

_REGEX_EMAIL_SIMPLES = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _hoje_brasil():
    return datetime.now(TZ_BRASIL).date()


def email_valido(email: Optional[str]) -> bool:
    if not email or not isinstance(email, str):
        return False
    return bool(_REGEX_EMAIL_SIMPLES.match(email.strip()))


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg2")


def obter_ou_criar_usuario(conn, email: str) -> dict:
    email = email.strip().lower()
    cursor = conn.cursor()
    placeholder = "%s" if _is_postgres(conn) else "?"
    hoje = _hoje_brasil()

    # Upsert atômico para evitar race condition na criação simultânea
    if _is_postgres(conn):
        sql = f"""
        INSERT INTO app_users (email, plano, consultas_hoje, data_ultima_consulta)
        VALUES ({placeholder}, 'free', 0, {placeholder})
        ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
        RETURNING email, plano, consultas_hoje, data_ultima_consulta;
        """
        cursor.execute(sql, (email, hoje))
        row = cursor.fetchone()
    else:
        cursor.execute(f"SELECT email, plano, consultas_hoje, data_ultima_consulta FROM app_users WHERE email = {placeholder}", (email,))
        row = cursor.fetchone()
        if not row:
            cursor.execute(f"INSERT OR IGNORE INTO app_users (email, plano, consultas_hoje, data_ultima_consulta) VALUES ({placeholder}, 'free', 0, {placeholder})", (email, hoje))
            conn.commit()
            cursor.execute(f"SELECT email, plano, consultas_hoje, data_ultima_consulta FROM app_users WHERE email = {placeholder}", (email,))
            row = cursor.fetchone()

    return {
        "email": row[0],
        "plano": row[1],
        "consultas_hoje": row[2],
        "data_ultima_consulta": row[3],
    }


def checar_e_consumir_cota(conn, email: str, limite_free: int = LIMITE_CONSULTAS_FREE_DIARIO) -> dict:
    if conn is None:
        return {"permitido": True, "plano": "free", "consultas_hoje": 0, "limite": None}

    try:
        usuario = obter_ou_criar_usuario(conn, email)
        hoje = _hoje_brasil()

        data_ultima = usuario["data_ultima_consulta"]
        if isinstance(data_ultima, str):
            data_ultima = datetime.strptime(data_ultima, "%Y-%m-%d").date()

        # Se for um novo dia, reseta a contagem no banco
        if data_ultima != hoje:
            usuario["consultas_hoje"] = 0

        if usuario["plano"] == "pro":
            _atualizar_contagem(conn, email, consultas_hoje=usuario["consultas_hoje"], data=hoje)
            return {"permitido": True, "plano": "pro", "consultas_hoje": usuario["consultas_hoje"], "limite": None}

        # Atualização atômica condicional ao limite
        placeholder = "%s" if _is_postgres(conn) else "?"
        clausula_timestamp = "NOW()" if _is_postgres(conn) else "CURRENT_TIMESTAMP"
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE app_users
            SET consultas_hoje = CASE WHEN data_ultima_consulta < {placeholder} THEN 1 ELSE consultas_hoje + 1 END,
                data_ultima_consulta = {placeholder},
                atualizado_em = {clausula_timestamp}
            WHERE email = {placeholder} AND (data_ultima_consulta < {placeholder} OR consultas_hoje < {placeholder})
            """,
            (hoje, hoje, email.strip().lower(), hoje, limite_free),
        )
        conn.commit()

        if cursor.rowcount > 0:
            nova_contagem = 1 if data_ultima != hoje else usuario["consultas_hoje"] + 1
            return {"permitido": True, "plano": "free", "consultas_hoje": nova_contagem, "limite": limite_free}

        # Se rowcount == 0, significa que o limite já foi atingido
        return {"permitido": False, "plano": "free", "consultas_hoje": usuario["consultas_hoje"], "limite": limite_free}

    except Exception as e:
        print(f"[COTA] Falha ao checar cota para '{email}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return {"permitido": True, "plano": "free", "consultas_hoje": 0, "limite": None}


def _atualizar_contagem(conn, email: str, consultas_hoje: int, data):
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
    if conn is None or not email_valido(email):
        return False

    email = email.strip().lower()
    placeholder = "%s" if _is_postgres(conn) else "?"
    hoje = _hoje_brasil()

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
            (email, ghost_member_id, plano, hoje),
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
