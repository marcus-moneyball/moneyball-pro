"""
Dados de configuração estática: catálogo de mercados por esporte, perfis de analista.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

REGRAS_ESPORTES = {
    "futebol": "Mercados: Chance Dupla (1X/X2/12), Handicap Asiático, Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões e Mercado de Jogadores (Chutes/Gols).",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Hits de Rebatidor."
}

# Carlos agora é o único analista do sistema -- generalista, cobre mercados
# coletivos E individuais (antes divididos entre Carlos e Cris).
PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 3.0,
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
        "xg_medio", "xg_sofrido_medio", "shots_on_target_medio",
        # Controle Territorial e Ritmo
        "posse_media", "ppda_medio",  # PPDA = intensidade de pressão (Passes Per Defensive Action)
        # Eficiência e Momento Recente
        "conversion_rate", "forma_recente_xg",
    ],

    "basquete": [
        # Ritmo e Eficiências Macro
        "pace", "ortg", "drtg",
        # Eficiência de Arremesso (Shooting Splits)
        "efg_pct", "ts_pct",   # Effective Field Goal % / True Shooting %
        # Controle de Posse e Rebotes
        "tov_pct", "reb_pct",  # Turnover Rate / Rebound Rate
        # Contexto de Elenco
        "fatigue_index",       # Back-to-back / Desfalques críticos
    ],

    "beisebol": [
        # Duelo do Titular (Pitcher)
        "pitcher_era", "pitcher_whip", "pitcher_k_per_9", "pitcher_xfip",
        "pitcher_mao",  # "R" ou "L" -- necessário pro matchup de platoon split
        # Produção do Lineup Contra a Mão do Arremessador Adversário
        "lineup_ops_vs_mao_adversaria",  # OPS deste lineup especificamente contra a mão do pitcher rival
        "lineup_wrc_plus",  # Weighted Runs Created Plus
        # Fatores de Bullpen (O calcanhar de Aquiles para Props finais)
        "bullpen_era_last_30", "bullpen_workload_fatigue",
    ],
}
