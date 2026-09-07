"""
Cliente direto do The Odds API pro MoneyballPro (sem depender do SmartCenter).

Estratégia de cota: UMA chamada por liga busca TODOS os jogos futuros
daquela liga de uma vez (custo tipicamente de 1 crédito -- mercados x
regiões, não jogos), e o resultado inteiro é gravado na odds_sharp_cache
compartilhada. As análises seguintes fazem só leitura no Postgres, sem
gastar cota de novo -- só refaz a chamada se o cache estiver velho.

Mercado usado: h2h (1X2 no futebol -- casa/empate/fora), com preferência
pela Pinnacle quando ela aparece entre as casas retornadas.
"""
import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Optional

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Confirmar a chave certa via GET /v4/sports (grátis) antes de usar em produção.
SPORT_KEYS = {
    "brasileirao": "soccer_brazil_campeonato",
    "premier_league": "soccer_epl",
}

HORAS_VALIDADE_CACHE = 6
BOOKMAKER_PREFERIDO = "pinnacle"


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg2")


def _cache_esta_fresco(conn, liga: str) -> bool:
    """Checa se já existe cache recente pra essa liga -- evita chamada redundante."""
    if conn is None:
        return False
    placeholder = "%s" if _is_postgres(conn) else "?"
    limite = datetime.now(timezone.utc) - timedelta(hours=HORAS_VALIDADE_CACHE)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT atualizado_em FROM odds_cache_liga_status WHERE liga = {placeholder}",
            (liga,),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return False
        atualizado_em = row[0]
        if isinstance(atualizado_em, str):
            atualizado_em = datetime.fromisoformat(atualizado_em.replace("Z", "+00:00"))
        return atualizado_em > limite
    except Exception:
        return False


def _marcar_liga_atualizada(conn, liga: str):
    placeholder = "%s" if _is_postgres(conn) else "?"
    upsert = (
        f"""INSERT INTO odds_cache_liga_status (liga, atualizado_em) VALUES ({placeholder}, NOW())
            ON CONFLICT (liga) DO UPDATE SET atualizado_em = NOW()"""
        if _is_postgres(conn) else
        f"""INSERT OR REPLACE INTO odds_cache_liga_status (liga, atualizado_em) VALUES ({placeholder}, CURRENT_TIMESTAMP)"""
    )
    cursor = conn.cursor()
    cursor.execute(upsert, (liga,))
    conn.commit()


def _extrair_odds_h2h(evento: dict) -> Optional[list]:
    """Extrai [odd_casa, odd_empate, odd_fora] de um evento, priorizando a Pinnacle."""
    bookmakers = evento.get("bookmakers", [])
    if not bookmakers:
        return None

    escolhido = next((b for b in bookmakers if b.get("key") == BOOKMAKER_PREFERIDO), bookmakers[0])
    mercado_h2h = next((m for m in escolhido.get("markets", []) if m.get("key") == "h2h"), None)
    if not mercado_h2h:
        return None

    outcomes = {o["name"]: o["price"] for o in mercado_h2h.get("outcomes", [])}
    home = evento.get("home_team")
    away = evento.get("away_team")
    odd_casa = outcomes.get(home)
    odd_fora = outcomes.get(away)
    odd_empate = outcomes.get("Draw")

    if odd_casa is None or odd_fora is None or odd_empate is None:
        return None
    return [odd_casa, odd_empate, odd_fora]


def _salvar_no_cache(conn, esporte: str, time_a: str, time_b: str, data_jogo: str, mercado: str, odds: list):
    from odds_sharp import montar_chave
    chave = montar_chave(esporte, time_a, time_b, data_jogo)
    placeholder = "%s" if _is_postgres(conn) else "?"
    upsert = (
        f"""INSERT INTO odds_sharp_cache (chave, esporte, time_a, time_b, data_jogo, mercado, odds_decimais)
            VALUES ({placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder})
            ON CONFLICT (chave) DO UPDATE SET odds_decimais = EXCLUDED.odds_decimais, atualizado_em = NOW()"""
        if _is_postgres(conn) else
        f"""INSERT OR REPLACE INTO odds_sharp_cache (chave, esporte, time_a, time_b, data_jogo, mercado, odds_decimais)
            VALUES ({placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder},{placeholder})"""
    )
    cursor = conn.cursor()
    cursor.execute(upsert, (chave, esporte, time_a, time_b, data_jogo, mercado, json.dumps(odds)))
    conn.commit()


def atualizar_cache_da_liga(conn, liga: str) -> int:
    """
    Busca TODOS os jogos futuros de uma liga em uma única chamada e grava no
    cache. Retorna quantos jogos foram salvos (0 se pulou por cache fresco,
    erro, ou falta de token).
    """
    if not ODDS_API_KEY:
        print("[ODDS API] ODDS_API_KEY não configurado -- pulando atualização.")
        return 0

    sport_key = SPORT_KEYS.get(liga)
    if not sport_key:
        return 0

    if _cache_esta_fresco(conn, liga):
        return 0  # já atualizado recentemente, não gasta cota de novo

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds?apiKey={ODDS_API_KEY}&regions=eu,uk&markets=h2h&oddsFormat=decimal"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            eventos = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[ODDS API] Falha ao buscar odds da liga '{liga}': {e}")
        return 0

    salvos = 0
    for evento in eventos:
        odds = _extrair_odds_h2h(evento)
        if not odds:
            continue
        data_jogo = (evento.get("commence_time") or "")[:10]  # YYYY-MM-DD
        try:
            _salvar_no_cache(
                conn, "futebol", evento.get("home_team"), evento.get("away_team"),
                data_jogo, "1x2", odds,
            )
            salvos += 1
        except Exception as e:
            print(f"[ODDS API] Falha ao salvar evento no cache: {e}")

    if salvos > 0:
        try:
            _marcar_liga_atualizada(conn, liga)
        except Exception as e:
            print(f"[ODDS API] Falha ao marcar liga '{liga}' como atualizada: {e}")

    return salvos
