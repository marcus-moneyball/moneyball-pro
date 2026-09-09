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

E-mails de DEV (env var DEV_EMAILS, separados por vírgula) sempre têm acesso
ilimitado, sem precisar existir linha nenhuma no banco -- útil pra testar sem
gastar cota real nem mexer no Postgres na mão.

Além do limite por e-mail, existe um limite por IP (checar_limite_por_ip) --
e-mail sozinho é fácil de burlar (aba anônima + e-mail novo a cada vez), o
IP não reseta junto com o localStorage.
"""
import os
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

LIMITE_CONSULTAS_FREE_DIARIO = 3
LIMITE_CONSULTAS_POR_IP_DIARIO = 10  # teto por IP, além do limite por e-mail
TZ_BRASIL = ZoneInfo("America/Sao_Paulo")

_REGEX_EMAIL_SIMPLES = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_DEV_EMAILS = {
    e.strip().lower() for e in os.getenv("DEV_EMAILS", "").split(",") if e.strip()
}


def _hoje_brasil():
    return datetime.now(TZ_BRASIL).date()


def email_valido(email: Optional[str]) -> bool:
    if not email or not isinstance(email, str):
        return False
    return bool(_REGEX_EMAIL_SIMPLES.match(email.strip()))


def eh_email_dev(email: Optional[str]) -> bool:
    return bool(email) and email.strip().lower() in _DEV_EMAILS


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg2")


def obter_ou_criar_usuario(conn, email: str) -> dict:
    """
    Retorna o usuário (criando se ainda não existir), com um campo extra
    `novo` (bool) indicando se ele acabou de ser criado agora mesmo -- usado
    pelo gatekeeper pra saber se deve criar o membro correspondente no Ghost.
    """
    email = email.strip().lower()
    cursor = conn.cursor()
    placeholder = "%s" if _is_postgres(conn) else "?"
    hoje = _hoje_brasil()

    if _is_postgres(conn):
        # (xmax = 0) é um truque do Postgres: só é TRUE quando a linha foi
        # de fato inserida agora -- se caiu no ON CONFLICT DO UPDATE porque
        # já existia, vem FALSE.
        sql = f"""
        INSERT INTO app_users (email, plano, consultas_hoje, data_ultima_consulta)
        VALUES ({placeholder}, 'free', 0, {placeholder})
        ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
        RETURNING email, plano, consultas_hoje, data_ultima_consulta, (xmax = 0) AS foi_inserido;
        """
        cursor.execute(sql, (email, hoje))
        row = cursor.fetchone()
        conn.commit()
        novo = bool(row[4])
    else:
        cursor.execute(f"SELECT email, plano, consultas_hoje, data_ultima_consulta FROM app_users WHERE email = {placeholder}", (email,))
        row = cursor.fetchone()
        novo = row is None
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
        "novo": novo,
    }


def checar_e_consumir_cota(conn, email: str, limite_free: int = LIMITE_CONSULTAS_FREE_DIARIO) -> dict:
    if conn is None:
        return {"permitido": True, "plano": "free", "consultas_hoje": 0, "limite": None, "novo_usuario": False}

    if eh_email_dev(email):
        return {"permitido": True, "plano": "dev", "consultas_hoje": 0, "limite": None, "novo_usuario": False}

    try:
        usuario = obter_ou_criar_usuario(conn, email)
        hoje = _hoje_brasil()
        novo_usuario = usuario["novo"]

        data_ultima = usuario["data_ultima_consulta"]
        if isinstance(data_ultima, str):
            data_ultima = datetime.strptime(data_ultima, "%Y-%m-%d").date()

        # Se for um novo dia, reseta a contagem no banco
        if data_ultima != hoje:
            usuario["consultas_hoje"] = 0

        if usuario["plano"] in ("pro", "dev"):
            _atualizar_contagem(conn, email, consultas_hoje=usuario["consultas_hoje"], data=hoje)
            return {
                "permitido": True, "plano": usuario["plano"], "consultas_hoje": usuario["consultas_hoje"],
                "limite": None, "novo_usuario": novo_usuario,
            }

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
            return {
                "permitido": True, "plano": "free", "consultas_hoje": nova_contagem,
                "limite": limite_free, "novo_usuario": novo_usuario,
            }

        # Se rowcount == 0, significa que o limite já foi atingido
        return {
            "permitido": False, "plano": "free", "consultas_hoje": usuario["consultas_hoje"],
            "limite": limite_free, "novo_usuario": novo_usuario,
        }

    except Exception as e:
        print(f"[COTA] Falha ao checar cota para '{email}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return {"permitido": True, "plano": "free", "consultas_hoje": 0, "limite": None, "novo_usuario": False}


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


def checar_limite_por_ip(conn, ip: Optional[str], limite_ip: int = LIMITE_CONSULTAS_POR_IP_DIARIO) -> bool:
    """
    Retorna True se o IP AINDA PODE consultar (não bateu o teto do dia),
    False se já bateu. É um limite mais alto e independente do limite por
    e-mail -- existe só pra impedir alguém de resetar a cota abrindo aba
    anônima e trocando de e-mail a cada vez. Falha aberta (True) se não
    houver conexão ou IP disponível -- nunca bloqueia por causa disso.
    """
    if conn is None or not ip:
        return True

    placeholder = "%s" if _is_postgres(conn) else "?"
    clausula_timestamp = "NOW()" if _is_postgres(conn) else "CURRENT_TIMESTAMP"
    hoje = _hoje_brasil()

    try:
        cursor = conn.cursor()
        upsert = (
            f"""
            INSERT INTO ip_daily_usage (ip, data, contagem, atualizado_em)
            VALUES ({placeholder}, {placeholder}, 1, {clausula_timestamp})
            ON CONFLICT (ip) DO UPDATE SET
                contagem = CASE WHEN ip_daily_usage.data < {placeholder} THEN 1 ELSE ip_daily_usage.contagem + 1 END,
                data = {placeholder},
                atualizado_em = {clausula_timestamp}
            WHERE ip_daily_usage.data < {placeholder} OR ip_daily_usage.contagem < {placeholder}
            RETURNING contagem
            """ if _is_postgres(conn) else
            f"""
            INSERT INTO ip_daily_usage (ip, data, contagem) VALUES ({placeholder}, {placeholder}, 1)
            ON CONFLICT(ip) DO UPDATE SET
                contagem = CASE WHEN data < {placeholder} THEN 1 ELSE contagem + 1 END,
                data = {placeholder}
            WHERE data < {placeholder} OR contagem < {placeholder}
            """
        )
        cursor.execute(upsert, (ip, hoje, hoje, hoje, hoje, limite_ip))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[COTA IP] Falha ao checar limite por IP '{ip}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return True  # falha aberta -- nunca bloqueia por erro de infra


def sync_ghost_member(conn, email: str, ghost_member_id: Optional[str], plano: str) -> dict:
    """
    Retorna {'sucesso': bool, 'plano_anterior': str|None, 'plano_novo': str}.
    plano_anterior vem None se o usuário nunca existiu no cache local antes
    (não dá pra saber se "desceu" de plano nesse caso -- trata como não-downgrade).
    """
    if conn is None or not email_valido(email):
        return {"sucesso": False, "plano_anterior": None, "plano_novo": plano}

    email = email.strip().lower()
    placeholder = "%s" if _is_postgres(conn) else "?"
    hoje = _hoje_brasil()

    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT plano FROM app_users WHERE email = {placeholder}", (email,))
        row = cursor.fetchone()
        plano_anterior = row[0] if row else None

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
        return {"sucesso": True, "plano_anterior": plano_anterior, "plano_novo": plano}
    except Exception as e:
        print(f"[GHOST SYNC] Falha ao sincronizar '{email}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return {"sucesso": False, "plano_anterior": None, "plano_novo": plano}
