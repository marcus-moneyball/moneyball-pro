"""Validação e saneamento das entradas devolvidas pelo MIE2, contra o perfil do analista."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
from catalogos import DELTA_MAX_PLAUSIVEL
from utils import _parse_float_seguro


def validar_e_sanear_entrada(entrada: Optional[dict], perfil: dict,
                              candidatos_verificados: Optional[list] = None) -> Optional[dict]:
    if not entrada or not isinstance(entrada, dict) or not entrada.get("mercado"):
        return None

    odd = _parse_float_seguro(entrada.get("odd"))
    delta = _parse_float_seguro(entrada.get("delta_edge"))

    if odd is None or delta is None:
        return None

    if not (perfil["odd_min"] <= odd <= perfil["odd_max"]):
        return None

    # --- Trava anti-alucinação: o "delta_edge" e o "msc_score" que chegam
    # aqui vêm do texto que o próprio Carlos (LLM) escreveu -- nada garantia
    # até agora que esses números batem com o que o Python calculou de forma
    # determinística pra aquele candidato específico. Se recebemos a lista de
    # candidatos_calculados (já filtrados por MSC >= 80 antes de irem pro
    # LLM), cruzamos por mercado+seleção e SUBSTITUÍMOS os valores
    # auto-reportados pelos valores reais -- o LLM não pode "arredondar pra
    # cima" a própria margem. Se citou mercado/seleção que não está na lista
    # de candidatos verificados, é tratado como possível alucinação e
    # descartado (exceto a entrada de fallback "abaixo_do_edge_minimo", que
    # é intencionalmente a melhor opção disponível mesmo sem bater o piso).
    if candidatos_verificados and not entrada.get("abaixo_do_edge_minimo"):
        candidato_real = next(
            (c for c in candidatos_verificados
             if c.get("mercado") == entrada.get("mercado")
             and c.get("selecao") == entrada.get("selecao")),
            None,
        )
        if candidato_real is None:
            return None

        delta_real = candidato_real.get("delta_edge_pct_calculado")
        if delta_real is not None:
            entrada["delta_edge"] = delta_real
            delta = delta_real

        if candidato_real.get("msc_calculado") is not None:
            entrada["msc_score"] = candidato_real["msc_calculado"]

    # Regra "sempre 1 bilhete": quando a entrada já veio marcada como a melhor
    # opção disponível abaixo do edge mínimo (ver prompts_mie2.py seção 4,
    # regra 1), ela é INTENCIONALMENTE abaixo de delta_min -- não pode ser
    # derrubada aqui, senão a funcionalidade inteira de "nunca retornar bilhete
    # vazio" fica sem efeito. Isso só vale pra entrada marcada assim -- a
    # entrada_2 nunca deveria vir com essa flag (regra 2 da mesma seção), então
    # continua sujeita ao filtro normal.
    if not entrada.get("abaixo_do_edge_minimo"):
        if delta < perfil["delta_min"]:
            return None

    if delta > DELTA_MAX_PLAUSIVEL:
        return None

    return entrada
