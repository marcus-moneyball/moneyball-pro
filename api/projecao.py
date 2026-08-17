"""
Orquestra a obtenção dos lambdas (expectativa marcada/sofrida) para os DOIS times
de um confronto, em cascata: banco local (só se os DOIS estiverem lá) -> MIE1
(Gemini com grounding, que já calcula os dois times cruzados numa única chamada).

Regra de ouro: nunca mistura fonte no mesmo confronto -- ou os dois lambdas vêm
do banco local, ou os dois vêm do Gemini. Misturar geraria uma assimetria de
confiabilidade entre os dois lados que o CALC não consegue detectar depois
(um lado com amostra robusta do banco, o outro estimado pela busca).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Callable
from dados_time import buscar_estatisticas_no_banco


def obter_projecoes_partida(time_a: str, time_b: str, sport: str,
                             competicao: Optional[str] = None,
                             conn=None, gemini_client=None,
                             executar_mie1_fn: Optional[Callable] = None) -> Optional[dict]:
    """
    Retorna:
    {
        "fonte": "banco_local" | "gemini_busca",
        "lam_a": float,
        "lam_b": float,
        "mie1_data": dict | None,  # só populado quando fonte == "gemini_busca"
    }
    ou None se nenhuma camada encontrou dado suficiente pros dois times.
    """
    if conn is not None:
        stats_a = buscar_estatisticas_no_banco(conn, time_a, sport, competicao)
        stats_b = buscar_estatisticas_no_banco(conn, time_b, sport, competicao)

        if stats_a and stats_b:
            # Mesma lógica de cruzamento do estimar_lambda em calc.py: expectativa
            # de cada time neste confronto = (média marcada por ele + média sofrida
            # pelo adversário) / 2.
            lam_a = round((stats_a["media_marcada"] + stats_b["media_sofrida"]) / 2, 3)
            lam_b = round((stats_b["media_marcada"] + stats_a["media_sofrida"]) / 2, 3)
            return {
                "fonte": "banco_local",
                "lam_a": lam_a,
                "lam_b": lam_b,
                "mie1_data": None,
            }

    if executar_mie1_fn and gemini_client:
        dados = executar_mie1_fn(gemini_client, time_a, time_b, sport)
        if dados:
            return {
                "fonte": "gemini_busca",
                "lam_a": dados.get("team_a_projected"),
                "lam_b": dados.get("team_b_projected"),
                "mie1_data": dados,
            }

    return None
