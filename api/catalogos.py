"""
Dados de configuração estática: catálogo de mercados por esporte, perfis de analista.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

REGRAS_ESPORTES = {
    "futebol": "Mercados: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor."
}

# Carlos agora é o único analista do sistema -- generalista, cobre mercados
# coletivos E individuais (antes divididos entre Carlos e Cris).
PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 3.5,
        "odd_min": 1.50,
        "odd_max": 3.50,
    },
}

DELTA_MAX_PLAUSIVEL = 15.0

CONFIG_MERCADO_PRINCIPAL = {
    "futebol": {"nome_stat": "gols", "nome_mercado": "Total de Gols da Partida", "unidade_selecao": "Gols"},
    "beisebol": {"nome_stat": "runs", "nome_mercado": "Total de Runs da Partida", "unidade_selecao": "Runs"},
    "basquete": {"nome_stat": "pontos", "nome_mercado": "Total de Pontos da Partida", "unidade_selecao": "Pontos"},
}

FONTES_AUTORIZADAS_POR_ESPORTE = {
    "futebol": "site:fbref.com OR site:sofascore.com",
    "basquete": "site:basketball-reference.com OR site:nba.com",
    "beisebol": "site:baseballsavant.com OR site:baseball-reference.com",
}

# Metodologia Nexus Cap. V -- campos extras de roteiro de jogo que o MIE1 tenta
# buscar por esporte, além do team_a_projected/team_b_projected já existente.
# Cada esporte usa o vocabulário de dado que realmente existe e é confiável via
# grounding -- não existe um framework universal de roteiro (ver documentação).
# Campos não encontrados voltam null; nunca bloqueiam o restante do JSON do MIE1.
# catalogos.py

CAMPOS_ROTEIRO_POR_ESPORTE = {
    "futebol": [
        # Produção e Volume Ofensivo
        "team_a_xg_medio", "team_b_xg_medio",
        "team_a_xg_sofrido_medio", "team_b_xg_sofrido_medio",
        "team_a_shots_on_target_medio", "team_b_shots_on_target_medio",
        # Controle Territorial e Ritmo
        "team_a_posse_media", "team_b_posse_media",
        "team_a_ppda_medio", "team_b_ppda_medio", # Intensidade de pressão (Passes Per Defensive Action)
        # Eficiência e Momento Recente
        "team_a_conversion_rate", "team_b_conversion_rate",
        "team_a_forma_recente_xg", "team_b_forma_recente_xg"
    ],
    
    "basquete": [
        # Ritmo e Eficiências Macro
        "team_a_pace", "team_b_pace",
        "team_a_ortg", "team_b_ortg",
        "team_a_drtg", "team_b_drtg",
        # Eficiência de Arremesso (Shooting Splits)
        "team_a_efg_pct", "team_b_efg_pct", # Effective Field Goal %
        "team_a_ts_pct", "team_b_ts_pct",   # True Shooting %
        # Controle de Posse e Rebotes
        "team_a_tov_pct", "team_b_tov_pct", # Turnover Rate
        "team_a_reb_pct", "team_b_reb_pct", # Rebound Rate
        # Contexto de Elenco
        "fatigue_index_a", "fatigue_index_b" # Back-to-back / Desfalques críticos
    ],
    
    
    "beisebol": [
        # Duelo do Titular (Pitcher)
        "pitcher_a_era", "pitcher_b_era",
        "pitcher_a_whip", "pitcher_b_whip",
        "pitcher_a_k_per_9", "pitcher_b_k_per_9",
        "pitcher_a_xfip", "pitcher_b_xfip", # Expected FIP (filtra defesa)
        # Produção do Lineup Contra o Estilo do Pitcher
        "lineup_a_vs_rhp_lhs_ops", "lineup_b_vs_rhp_lhs_ops",
        "lineup_a_wrc_plus", "lineup_b_wrc_plus", # Weighted Runs Created Plus
        # Fatores de Bullpen (O calcanhar de Aquiles para Props finais)
        "bullpen_a_era_last_30", "bullpen_b_era_last_30",
        "bullpen_a_workload_fatigue", "bullpen_b_workload_fatigue"
    ],
}
