"""
Integração com football-data.org -- fonte estruturada e determinística pra
FORMA RECENTE (resultados, gols marcados/sofridos) de times do Brasileirão e
da Premier League. Não substitui a busca com grounding do MIE1 pro xG/PPDA/
posse (o football-data.org não tem esses dados no plano gratuito) -- só
resolve a fatia de "forma recente" que antes vinha 100% de busca na web e
já causou pelo menos um caso confirmado de alucinação (Bahia).

Best-effort: qualquer falha (rede, time não encontrado, liga não suportada)
retorna None em vez de lançar exceção -- a análise sempre pode seguir sem
esse reforço, caindo de volta no que já existia antes.
"""
import os
import json
import re
import unicodedata
import urllib.request
import urllib.error
from typing import Optional

FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
BASE_URL = "https://api.football-data.org/v4"

CODIGO_POR_LIGA = {
    "brasileirao": "BSA",
    "premier_league": "PL",
}


def _normalizar_nome(nome: str) -> str:
    """Remove acento, caixa e sufixos comuns pra facilitar o match de nome
    entre o que o OCR extraiu do print e o nome oficial do football-data.org."""
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    nome = nome.lower().strip()
    nome = re.sub(r"\b(fc|cf|ec|afc|sc|red bull)\b", "", nome)
    nome = re.sub(r"[^a-z0-9\s]", "", nome)
    nome = re.sub(r"\s+", " ", nome).strip()
    return nome


def _chamar_api(caminho: str) -> Optional[dict]:
    if not FOOTBALL_DATA_TOKEN:
        print("[FOOTBALL-DATA] FOOTBALL_DATA_TOKEN não configurado -- pulando reforço de forma recente.")
        return None

    url = f"{BASE_URL}{caminho}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        corpo = e.read().decode("utf-8", errors="replace")
        print(f"[FOOTBALL-DATA] Erro HTTP em {caminho}: {e.code} {corpo}")
        return None
    except Exception as e:
        print(f"[FOOTBALL-DATA] Erro em {caminho}: {e}")
        return None


def buscar_times_da_competicao(codigo_liga: str) -> dict:
    """
    Retorna {nome_normalizado: team_id} pra todos os times de uma competição.
    Chamada pensada pra ser cacheada por fora (não muda durante a temporada).
    """
    dados = _chamar_api(f"/competitions/{codigo_liga}/teams")
    if not dados or "teams" not in dados:
        return {}

    mapa = {}
    for time in dados["teams"]:
        for campo in ("name", "shortName", "tla"):
            valor = time.get(campo)
            if valor:
                mapa[_normalizar_nome(valor)] = time["id"]
    return mapa


def _encontrar_team_id(nome_time: str, mapa_times: dict) -> Optional[int]:
    alvo = _normalizar_nome(nome_time)
    if alvo in mapa_times:
        return mapa_times[alvo]
    # match parcial -- pega o primeiro nome oficial que contém o nome buscado
    # ou vice-versa (ex: "bragantino" dentro de "red bull bragantino")
    for nome_oficial, team_id in mapa_times.items():
        if alvo in nome_oficial or nome_oficial in alvo:
            return team_id
    return None


def calcular_forma_recente(matches: list, team_id: int, limite: int = 5) -> Optional[dict]:
    """
    A partir da lista de partidas FINALIZADAS de um time (mais recente
    primeiro), calcula gols marcados/sofridos e jogos sem sofrer gol nas
    últimas `limite` partidas.
    """
    finalizadas = [m for m in matches if m.get("status") == "FINISHED"]
    finalizadas.sort(key=lambda m: m.get("utcDate", ""), reverse=True)
    recentes = finalizadas[:limite]
    if not recentes:
        return None

    gols_marcados = 0
    gols_sofridos = 0
    jogos_sem_sofrer = 0
    resultados = []

    for m in recentes:
        placar = m.get("score", {}).get("fullTime", {})
        gols_casa = placar.get("home")
        gols_fora = placar.get("away")
        if gols_casa is None or gols_fora is None:
            continue

        time_e_casa = m.get("homeTeam", {}).get("id") == team_id
        gols_do_time = gols_casa if time_e_casa else gols_fora
        gols_do_adversario = gols_fora if time_e_casa else gols_casa

        gols_marcados += gols_do_time
        gols_sofridos += gols_do_adversario
        if gols_do_adversario == 0:
            jogos_sem_sofrer += 1
        resultados.append(f"{gols_do_time}x{gols_do_adversario}")

    return {
        "jogos_considerados": len(recentes),
        "gols_marcados_ultimos_5": gols_marcados,
        "gols_sofridos_ultimos_5": gols_sofridos,
        "jogos_sem_sofrer_gol": jogos_sem_sofrer,
        "resultados_recentes": resultados,  # mais recente primeiro
    }


def obter_forma_recente_estruturada(nome_time: str, liga: str) -> Optional[dict]:
    """
    Função principal: dado o nome do time (como veio do OCR) e a liga
    selecionada pelo usuário, retorna a forma recente estruturada, ou None
    se a liga não é suportada, o time não foi encontrado, ou a API falhou.
    """
    codigo_liga = CODIGO_POR_LIGA.get(liga)
    if not codigo_liga:
        return None  # liga "outra" ou desconhecida -- sem reforço, comportamento de hoje

    mapa_times = buscar_times_da_competicao(codigo_liga)
    if not mapa_times:
        return None

    team_id = _encontrar_team_id(nome_time, mapa_times)
    if not team_id:
        print(f"[FOOTBALL-DATA] Time '{nome_time}' não encontrado na competição {codigo_liga}.")
        return None

    dados_partidas = _chamar_api(f"/teams/{team_id}/matches?status=FINISHED&limit=10")
    if not dados_partidas or "matches" not in dados_partidas:
        return None

    return calcular_forma_recente(dados_partidas["matches"], team_id)
