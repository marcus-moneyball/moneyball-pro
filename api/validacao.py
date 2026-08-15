"""Validação e saneamento das entradas devolvidas pelo MIE2, contra o perfil do analista."""

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

    if delta < perfil["delta_min"]:
        return None

    if delta > DELTA_MAX_PLAUSIVEL:
        return None

    return entrada
