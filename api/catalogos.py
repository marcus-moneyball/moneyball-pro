"""
Dados de configuração estática: catálogo de mercados por esporte, perfis de analista.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

REGRAS_ESPORTES = {
    "futebol": "Mercados: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor.",
    "nfl": "Mercados: Vencedor (Moneyline), Spread (Handicap), Total de Pontos, Yardas de Passe/Corrida/Recepção, Touchdown Qualquer Momento."
}

PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 3.5,
        "odd_min": 1.50,
        "odd_max": 3.50,
    },
    "cris": {
        "delta_min": 4.0,
        "odd_min": 1.60,
        "odd_max": 3.00,
    },
}

DELTA_MAX_PLAUSIVEL = 15.0

CONFIG_MERCADO_PRINCIPAL = {
    "futebol": {"nome_stat": "gols", "nome_mercado": "Total de Gols da Partida", "unidade_selecao": "Gols"},
    "beisebol": {"nome_stat": "runs", "nome_mercado": "Total de Runs da Partida", "unidade_selecao": "Runs"},
    "basquete": {"nome_stat": "pontos", "nome_mercado": "Total de Pontos da Partida", "unidade_selecao": "Pontos"},
    "nfl": {"nome_stat": "pontos", "nome_mercado": "Total de Pontos da Partida", "unidade_selecao": "Pontos"},
}

FONTES_AUTORIZADAS_POR_ESPORTE = {
    "futebol": "site:fbref.com OR site:sofascore.com",
    "basquete": "site:basketball-reference.com OR site:nba.com",
    "beisebol": "site:baseballsavant.com OR site:baseball-reference.com",
    "nfl": "site:pro-football-reference.com",
}

# Metodologia Nexus Cap. V -- campos extras de roteiro de jogo que o MIE1 tenta
# buscar por esporte, além do team_a_projected/team_b_projected já existente.
# Cada esporte usa o vocabulário de dado que realmente existe e é confiável via
# grounding -- não existe um framework universal de roteiro (ver documentação).
# Campos não encontrados voltam null; nunca bloqueiam o restante do JSON do MIE1.
CAMPOS_ROTEIRO_POR_ESPORTE = {
    "futebol": ["xg_medio", "xg_sofrido_medio", "posse_media"],
    "basquete": ["pace", "ortg", "drtg"],
    "nfl": ["success_rate_of", "success_rate_def"],  # status experimental
    "beisebol": ["pitcher_era", "lineup_ops", "bullpen_era"],
}
