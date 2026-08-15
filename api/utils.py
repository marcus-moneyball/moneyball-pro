"""Utilitários pequenos e sem dependência de nenhum outro módulo do projeto."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import re
from typing import Optional


def _extrair_json_de_texto(texto: str) -> Optional[dict]:
    if not texto:
        return None
    texto = texto.strip()
    if "```" in texto:
        texto = re.sub(r"^```(?:json)?\s*", "", texto)
        texto = re.sub(r"\s*```$", "", texto)
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_float_seguro(valor) -> Optional[float]:
    if valor is None:
        return None
    try:
        texto = str(valor).strip().replace("%", "").replace(",", ".")
        return float(texto)
    except (ValueError, TypeError):
        return None
