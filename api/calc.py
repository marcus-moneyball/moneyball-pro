"""
MIE2 - Endpoint HTTP (Vercel Python Function)
Recebe o dossiê de mercados do SmartCenter (Worker), calcula Delta + Poisson + Kelly
fracionado de forma determinística, e devolve o resultado calculado por mercado.

Rota no Vercel: POST /api/calc
"""

from flask import Flask, request, jsonify
from mie2_calc import calcular_dossie

app = Flask(__name__)


@app.route("/api/calc", methods=["POST"])
def calc():
    body = request.get_json(silent=True)

    if not body or "mercados" not in body or not isinstance(body["mercados"], list):
        return jsonify({
            "erro": "Corpo inválido. Esperado: { \"mercados\": [ {...}, {...} ] }"
        }), 400

    resultados = calcular_dossie(body["mercados"])
    return jsonify({"resultados": resultados})


@app.route("/api/calc", methods=["GET"])
def health():
    return jsonify({"status": "MIE2 online", "uso": "POST /api/calc com { mercados: [...] }"})
