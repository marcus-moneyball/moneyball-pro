"""
Leitura da tabela compartilhada `odds_sharp_cache` (escrita pelo SmartCenter,
que já coleta odds via The Odds API no ciclo de varredura dele). O
MoneyballPro NUNCA escreve nessa tabela nem chama o The Odds API diretamente
-- só lê o que já está lá e calcula o fair odds (sem vig) a partir da odd
crua salva.

Se a partida/mercado não estiver na tabela (SmartCenter ainda não varreu
esse jogo, ou o jogo não é de uma liga que ele cobre), retorna None --
NEUTRO, nunca penaliza por ausência de dado.
"""
import unicodedata
import re
from typing import List, Optional


def _normalizar_nome(nome: str) -> str:
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    nome = nome.lower().strip()
    nome = re.sub(r"[^a-z0-9\s]", "", nome)
    nome = re.sub(r"\s+", "", nome)
    return nome


def montar_chave(esporte: str, time_a: str, time_b: str, data_jogo: str) -> str:
    return f"{esporte.lower()}:{_normalizar_nome(time_a)}:{_normalizar_nome(time_b)}:{data_jogo}"


def calcular_fair_odds(odds_decimais: List[float]) -> Optional[List[float]]:
    """Remove o vig (margem da casa) via normalização multiplicativa."""
    if not odds_decimais or any(o is None or o <= 1.0 for o in odds_decimais):
        return None
    probs_brutas = [1 / o for o in odds_decimais]
    overround = sum(probs_brutas)
    if overround <= 0:
        return None
    return [round(p / overround, 6) for p in probs_brutas]


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg2")


def buscar_odds_sharp(conn, esporte: str, time_a: str, time_b: str, data_jogo: str, mercado: str) -> Optional[dict]:
    """
    Retorna {"odds_decimais": [...], "fair_probs": [...]} pra esse
    esporte+confronto+data+mercado, ou None se o SmartCenter ainda não
    coletou esse jogo (neutro -- não é erro, só ausência de dado).

    Tenta a chave na ordem recebida e, se não achar, na ordem invertida
    (time_b, time_a) -- a ordem "mandante primeiro" pode variar entre a
    fonte que o SmartCenter usa e o que o OCR extrai do print.
    """
    if conn is None:
        return None

    placeholder = "%s" if _is_postgres(conn) else "?"

    for a, b, invertido in [(time_a, time_b, False), (time_b, time_a, True)]:
        chave = montar_chave(esporte, a, b, data_jogo)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT odds_decimais FROM odds_sharp_cache WHERE chave = {placeholder} AND mercado = {placeholder}",
                (chave, mercado),
            )
            row = cursor.fetchone()
            if row and row[0]:
                fair = calcular_fair_odds(row[0])
                if fair is not None:
                    odds_final, fair_final = row[0], fair
                    if invertido and len(odds_final) == 2:
                        # a chave bateu na ordem (time_b, time_a) -- devolve
                        # sempre na ordem que o chamador pediu (time_a, time_b)
                        odds_final = list(reversed(odds_final))
                        fair_final = list(reversed(fair_final))
                    return {"odds_decimais": odds_final, "fair_probs": fair_final}
        except Exception as e:
            print(f"[ODDS SHARP] Falha ao consultar cache para '{esporte} {a} vs {b}': {e}")
            return None

    return None
