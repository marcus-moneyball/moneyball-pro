"""Validação e saneamento das entradas devolvidas pelo MIE2, contra o perfil do analista."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
from catalogos import DELTA_MAX_PLAUSIVEL
from utils import _parse_float_seguro


def validar_e_sanear_entrada(entrada: Optional[dict], perfil: dict) -> Optional[dict]:
    if not entrada or not isinstance(entrada, dict) or not entrada.get("mercado"):
        return None

    odd = _parse_float_seguro(entrada.get("odd"))
    delta = _parse_float_seguro(entrada.get("delta_edge"))

    if odd is None or delta is None:
        return None

    if not (perfil["odd_min"] <= odd <= perfil["odd_max"]):
        return None

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
