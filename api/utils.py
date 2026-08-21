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


def converter_odd_para_decimal(valor) -> Optional[float]:
    """
    Conversão de odds americanas para decimais -- determinística, sem LLM.
    A entrada pode trazer odds em qualquer um dos dois formatos (dependendo da
    fonte do print: Scores24 costuma ser decimal, Action Network costuma ser
    americana) -- essa função normaliza tudo pra decimal antes de qualquer
    cálculo de EV/Kelly.

    - Já decimal (ex: 1.85, "1.85") -> devolve como está.
    - Americana positiva (ex: "+150", 150) -> decimal = 1 + (valor/100)
    - Americana negativa (ex: "-150") -> decimal = 1 + (100/abs(valor))
    - Entrada inválida/vazia -> None (nunca inventa um número)
    """
    if valor is None:
        return None
    texto = str(valor).strip()
    if not texto:
        return None

    tem_sinal_explicito = texto.startswith("+") or texto.startswith("-")
    try:
        numero = float(texto.replace("+", ""))
    except ValueError:
        return None

    # Odd americana tem sinal explícito, OU é um número inteiro grande (abaixo
    # de -100 ou acima de 100) -- odds decimais de mercado real não costumam
    # passar de ~50, então esse limiar é seguro na prática.
    eh_americana = tem_sinal_explicito or abs(numero) >= 100
    if not eh_americana:
        if numero <= 1:
            return None  # odd decimal tem que ser > 1.0, nunca inventa correção
        return round(numero, 4)

    if numero == 0:
        return None
    if numero > 0:
        decimal = 1 + (numero / 100)
    else:
        decimal = 1 + (100 / abs(numero))
    return round(decimal, 4)
