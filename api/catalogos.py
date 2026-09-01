"""
Dados de configuração estática: catálogo de mercados por esporte, perfis de analista.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

REGRAS_ESPORTES = {
    "futebol": "Mercados: Chance Dupla (1X/X2/12), Handicap Asiático, Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor, Hits Permitidos pelo Pitcher (Player Props)."
}

# Carlos agora é o único analista do sistema -- generalista, cobre mercados
# coletivos E individuais (antes divididos entre Carlos e Cris).
PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 2.0,
        "odd_min": 1.50,
        "odd_max": 3.20,
    },
}

DELTA_MAX_PLAUSIVEL = 14.0

CONFIG_MERCADO_PRINCIPAL = {
    "futebol": {"nome_stat": "gols", "nome_mercado": "Total de Gols da Partida", "unidade_selecao": "Gols"},
    "beisebol": {"nome_stat": "runs", "nome_mercado": "Total de Runs da Partida", "unidade_selecao": "Runs"},
    "basquete": {"nome_stat": "pontos", "nome_mercado": "Total de Pontos da Partida", "unidade_selecao": "Pontos"},
}

FONTES_AUTORIZADAS_POR_ESPORTE = {
    "futebol": "site:fotmob.com OR site:sofascore.com OR site:understat.com",
    "basquete": "site:basketball-reference.com OR site:nba.com OR site:espn.com",
    "beisebol": "site:baseballsavant.com OR site:baseball-reference.com OR site:fangraphs.com",
}

# Metodologia Nexus Cap. V -- campos extras de roteiro de jogo que o MIE1 tenta
# buscar por esporte, além do team_a_projected/team_b_projected já existente.
CAMPOS_ROTEIRO_POR_ESPORTE = {
    "futebol": [
        "xg_medio", "xg_sofrido_medio", "shots_on_target_medio",
        "posse_media", "ppda_medio",
        "conversion_rate", "forma_recente_xg",
    ],
    "basquete": [
        "pace", "ortg", "drtg",
        "efg_pct", "ts_pct",
        "tov_pct", "reb_pct",
        "fatigue_index",
    ],
    "beisebol": [
        "pitcher_era", "pitcher_whip", "pitcher_k_per_9", "pitcher_h_per_9", "pitcher_opp_ba", "pitcher_xfip",
        "pitcher_mao",
        "lineup_ops_vs_mao_adversaria",
        "lineup_wrc_plus",
        "bullpen_era_last_30", "bullpen_workload_fatigue",
    ],
}

