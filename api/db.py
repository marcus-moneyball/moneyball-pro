"""
Conexão com o Postgres externo (banco em outro app na Vercel).
Lê a connection string de DATABASE_URL. Se não estiver configurada,
retorna None de propósito -- o resto do pipeline já sabe cair pro
próximo nível da cascata (api-sports.io ou Gemini) quando conn é None,
então "banco não configurado ainda" nunca quebra o fluxo.
"""

import os
from typing import Optional

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # ambiente sem psycopg2 instalado -- get_connection() vira no-op


def get_connection():
    """
    Abre uma conexão nova com o Postgres (vida curta, adequado pro modelo de
    função serverless -- não tenta manter pool entre invocações, já que cada
    invocação da Vercel pode ser um processo novo).
    Retorna None (nunca lança exceção) se DATABASE_URL não estiver configurada
    ou se psycopg2 não estiver instalado -- deixa o chamador decidir o fallback.
    """
    if psycopg2 is None:
        return None

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    try:
        # Vercel Postgres / Neon geralmente exigem SSL -- sslmode=require é
        # seguro mesmo se a connection string já não especificar.
        conn = psycopg2.connect(database_url, sslmode="require", connect_timeout=5)
        return conn
    except Exception as e:
        print(f"[DB] Falha ao conectar no Postgres externo: {e}")
        return None


def fechar_conexao(conn):
    """Fecha a conexão com segurança -- nunca lança exceção pro chamador."""
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass
