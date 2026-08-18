"""
Dados de configuração estática: catálogo de mercados por esporte, perfis de analista.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

REGRAS_ESPORTES = {
    "futebol": "Mercados: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios, Cartões, Chutes a gol.",
    "basquete": "Mercados: Vencedor (Moneyline), Handicap (Spread), Total de Pontos (Over/Under), Player Props (Pontos, Rebotes, Assistências, Bolas de 3).",
    "beisebol": "Mercados: Moneyline, Run Line (Handicap), Total de Runs (Over/Under), F5 (Primeiras 5 Entradas), Strikeouts do Pitcher, Eliminações do Pitcher, Hits Permitidos do Pitcher."
}

# Carlos é o único analista do sistema -- generalista, cobre mercados coletivos e props válidas
PERFIS_ANALISTA = {
    "carlos": {
        "delta_min": 3.0,
        "odd_min": 1.50,
        "odd_max": 4.50,
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

CAMPOS_ROTEIRO_POR_ESPORTE = {
    "futebol": [
        # Produção e Volume Ofensivo
        "xg_medio", "xg_sofrido_medio", "shots_on_target_medio",
        # Controle Territorial e Ritmo (Field Tilt e PPDA são essenciais para Game Scripts)
        "posse_media", "field_tilt", "ppda_medio",
        # Eficiência e Momento Recente
        "conversion_rate", "forma_recente_xg",
    ],

    "basquete": [
        # Ritmo e Eficiências Macro
        "pace", "ortg", "drtg", "net_rating",
        # Eficiência de Arremesso (Shooting Splits)
        "efg_pct", "ts_pct",
        # Controle de Posse e Rebotes
        "tov_pct", "reb_pct",
        # Contexto de Elenco e Desgaste
        "fatigue_index",
    ],

    "beisebol": [
        # Duelo do Titular (Pitcher) - FIP e xFIP são os preditores primários
        "pitcher_era", "pitcher_fip", "pitcher_xfip", "pitcher_whip", "pitcher_k_per_9",
        "pitcher_mao",  # "R" ou "L" -- necessário para o matchup de platoon split
        # Produção do Lineup Contra a Mão do Arremessador Adversário
        "lineup_ops_vs_mao_adversaria",
        "lineup_wrc_plus",
        # Fatores de Bullpen
        "bullpen_era_last_30", "bullpen_workload_fatigue",
    ],
}
