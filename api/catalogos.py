"""
Dados de configuração estática: catálogo de mercados por esporte, perfis
de analista (Cris/Carlos), mapeamento de mercado principal por esporte,
fontes autorizadas de busca web pro MIE1. Sem lógica, só dados.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

REGRAS_ESPORTES = {
    "futebol": {
        "cris": "Mercados Coletivos permitidos: Vencedor do Jogo (1X2), Both Teams to Score (BTTS), Over/Under Gols, Escanteios Totais e Cartões Totais.",
        "carlos": "Mercados Individuais permitidos: Chutes ao gol do jogador, Gols a qualquer momento do jogador, Desarmes e Faltas cometidas."
    },
    "basquete": {
        "cris": "Mercados Coletivos permitidos: Vencedor (Moneyline), Handicap (Spread), Total de Pontos da Partida (Over/Under).",
        "carlos": "Mercados Individuais permitidos: Player Props de Pontos, Rebotes, Assistências e Bolas de 3 convertidas do jogador."
    },
    "beisebol": {
        "cris": "Mercados Coletivos permitidos: Moneyline, Run Line (Handicap), Total de Runs da Partida (Over/Under), F5 (Primeiras 5 Entradas).",
        "carlos": "Mercados Individuais permitidos: Strikeouts do Pitcher e Hits do Rebatidor."
    },
    "nfl": {
        "cris": "Mercados Coletivos permitidos: Vencedor (Moneyline), Spread (Handicap), Total de Pontos da Partida (Over/Under).",
        "carlos": "Mercados Individuais permitidos: Jardas de Passe/Corrida/Recepção e Touchdown a qualquer momento do jogador."
    }
}

PERFIS_ANALISTA = {
    "cris": {
        "nome": "Cris (Coletivos e Assimetrias)",
        "delta_min": 4.0,
        "odd_min": 1.40,
        "odd_max": 2.80,
        "faixas_stake": [
            (4.0, 6.5, "1.0u"),
            (6.5, 9.0, "1.5u"),
            (9.0, float("inf"), "2.0u"),
        ],
    },
    "carlos": {
        "nome": "Carlos (Individuais e Correlações)",
        "delta_min": 3.5,
        "odd_min": 1.50,
        "odd_max": 3.50,
        "faixas_stake": [
            (3.5, 6.0, "1.0u"),
            (6.0, 8.5, "1.5u"),
            (8.5, float("inf"), "2.0u"),
        ],
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
