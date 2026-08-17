"""
Camada de obtenção de estatísticas de time, em cascata:
1. Banco local (Postgres) -- rápido, sem chamada de rede, dado que você já sincronizou.
2. api-sports.io ao vivo -- só quando o banco não tem o time/competição.
3. Gemini com busca -- último recurso, só pro que sobrar (competições muito obscuras).

Cada projeção volta marcada com "fonte" e "amostra_n", pra sempre saber de onde veio
o número -- resolve o "não sei o que o Gemini tá achando" de uma vez por todas.
"""

from typing import Optional, TypedDict


class ProjecaoTime(TypedDict):
    media_marcada: float
    media_sofrida: float
    amostra_n: int
    fonte: str  # "banco_local" | "api_sports" | "gemini_busca"
    janela: Optional[str]


# ============================================================
# CAMADA 1 -- BANCO LOCAL (Postgres)
# ============================================================

def buscar_estatisticas_no_banco(conn, time_nome: str, esporte: str,
                                   competicao: Optional[str] = None) -> Optional[ProjecaoTime]:
    """
    Busca a linha mais recente pra esse time/esporte(/competição) no banco local.
    `conn` é uma conexão DB-API 2.0 (psycopg2, ou sqlite3 nos testes -- a query usa
    apenas SQL padrão, compatível com os dois, pra facilitar o teste sem Postgres real).
    Retorna None se não achar -- nunca inventa, nunca aproxima de outro time.
    """
    cursor = conn.cursor()
    if competicao:
        cursor.execute(
            """
            SELECT media_marcada, media_sofrida, amostra_n, janela
            FROM team_season_stats
            WHERE time_nome = ? AND esporte = ? AND competicao = ?
            ORDER BY atualizado_em DESC
            LIMIT 1
            """.replace("?", "%s") if _is_postgres(conn) else
            """
            SELECT media_marcada, media_sofrida, amostra_n, janela
            FROM team_season_stats
            WHERE time_nome = ? AND esporte = ? AND competicao = ?
            ORDER BY atualizado_em DESC
            LIMIT 1
            """,
            (time_nome, esporte, competicao),
        )
    else:
        cursor.execute(
            """
            SELECT media_marcada, media_sofrida, amostra_n, janela
            FROM team_season_stats
            WHERE time_nome = ? AND esporte = ?
            ORDER BY atualizado_em DESC
            LIMIT 1
            """.replace("?", "%s") if _is_postgres(conn) else
            """
            SELECT media_marcada, media_sofrida, amostra_n, janela
            FROM team_season_stats
            WHERE time_nome = ? AND esporte = ?
            ORDER BY atualizado_em DESC
            LIMIT 1
            """,
            (time_nome, esporte),
        )

    row = cursor.fetchone()
    if not row:
        return None

    media_marcada, media_sofrida, amostra_n, janela = row
    return {
        "media_marcada": float(media_marcada),
        "media_sofrida": float(media_sofrida),
        "amostra_n": int(amostra_n),
        "fonte": "banco_local",
        "janela": janela,
    }


def _is_postgres(conn) -> bool:
    """Detecta se a conexão é Postgres (psycopg2) ou outra coisa (ex: sqlite3 em teste)."""
    return "psycopg2" in type(conn).__module__


# ============================================================
# CAMADA 2 -- api-sports.io AO VIVO (stub -- implementar por esporte)
# ============================================================

def buscar_estatisticas_api_sports(time_nome: str, esporte: str,
                                     api_key: str) -> Optional[ProjecaoTime]:
    """
    Chama api-sports.io ao vivo pra esse time (só quando o banco não tem o dado).
    TODO: implementar a chamada real por esporte (endpoints diferentes por API:
    v1.baseball.api-sports.io, v3.football.api-sports.io, etc). Deixei como stub
    porque preciso confirmar o endpoint exato e o formato de resposta com uma
    chamada real usando sua chave -- não quero adivinhar o schema de resposta
    e te entregar algo que quebra silenciosamente.
    """
    raise NotImplementedError(
        "Chamada ao vivo pro api-sports.io ainda não implementada -- "
        "preciso de uma resposta real da API (com sua chave) pra mapear os campos certos."
    )


# ============================================================
# CASCATA COMPLETA
# ============================================================

def obter_projecao_time(time_nome: str, esporte: str, competicao: Optional[str],
                          conn=None, api_sports_key: Optional[str] = None,
                          buscar_via_gemini=None) -> Optional[ProjecaoTime]:
    """
    Tenta, em ordem: banco local -> api-sports.io ao vivo -> Gemini (busca).
    `buscar_via_gemini` é uma função injetada (ex: wrapper do executar_mie1 já
    existente) -- só chamada se as duas primeiras camadas não acharem nada.
    """
    if conn is not None:
        resultado = buscar_estatisticas_no_banco(conn, time_nome, esporte, competicao)
        if resultado:
            return resultado

    if api_sports_key:
        try:
            resultado = buscar_estatisticas_api_sports(time_nome, esporte, api_sports_key)
            if resultado:
                return resultado
        except NotImplementedError:
            pass  # cai pro próximo nível da cascata

    if buscar_via_gemini:
        resultado = buscar_via_gemini(time_nome, esporte, competicao)
        if resultado:
            resultado["fonte"] = "gemini_busca"
            return resultado

    return None
