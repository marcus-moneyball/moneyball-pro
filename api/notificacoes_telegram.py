"""
Publicação no Telegram, em DOIS destinos separados:

1. Canal público (TELEGRAM_CHAT_ID) -- só recomendações reais (entrada_1
   preenchida). É a vitrine de bilhetes, ninguém quer ver "sem entrada hoje"
   ali.

2. Ouvidoria privada (TELEGRAM_CHAT_ID_OUVIDORIA) -- TODO bilhete gerado,
   com ou sem recomendação, sempre com o código de auditoria em destaque.
   É o log completo pra quando um usuário pedir auditoria de um bilhete
   específico pelo código -- basta pesquisar o código nesse chat.

Ambos são best-effort: falha aqui NUNCA pode derrubar a análise que o
usuário já recebeu.
"""
import os
import json
import urllib.request
import urllib.error

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHAT_ID_OUVIDORIA = os.getenv("TELEGRAM_CHAT_ID_OUVIDORIA")


def _formatar_entrada(entrada: dict, label: str) -> str:
    rotulo = (entrada.get("confianca_exibicao") or {}).get("rotulo", "")
    linha = f"*{label}:* {entrada.get('mercado')} — {entrada.get('selecao')}\n"
    linha += (
        f"Odd @{entrada.get('odd')} | Delta {entrada.get('delta_edge')} | "
        f"Stake {entrada.get('stake_recomendada')}"
    )
    if rotulo:
        linha += f" | {rotulo}"
    return linha


def _cabecalho_comum(resultado: dict) -> list:
    match = resultado.get("match_info") or {}
    linhas = [
        f"{match.get('sport', '')}: {match.get('teams', 'Análise de Partida')}",
        f"Hipótese Tática: {resultado.get('hipotese_partida', '—')}",
    ]
    if resultado.get("casa_apostas"):
        linhas.append(f"Casa: {resultado['casa_apostas']}")
    return linhas


def montar_mensagem_publica(resultado: dict) -> str:
    dupla = resultado.get("dupla_de_elite") or {}
    e1 = dupla.get("entrada_1")
    e2 = dupla.get("entrada_2")

    linhas = ["🔥 *MoneyballPro — Análise de CARLOS*", ""]
    linhas += _cabecalho_comum(resultado)
    linhas.append("")

    if e1 and e1.get("mercado"):
        linhas.append(_formatar_entrada(e1, "Entrada 1"))
    if e2 and e2.get("mercado"):
        linhas.append("")
        linhas.append(_formatar_entrada(e2, "Entrada 2"))

    return "\n".join(linhas)


def montar_mensagem_ouvidoria(resultado: dict) -> str:
    dupla = resultado.get("dupla_de_elite") or {}
    e1 = dupla.get("entrada_1")
    e2 = dupla.get("entrada_2")
    codigo = resultado.get("codigo_auditoria", "SEM-CODIGO")

    linhas = [f"🗂 *Ouvidoria* — código `{codigo}`", ""]
    linhas += _cabecalho_comum(resultado)
    linhas.append("")

    if e1 and e1.get("mercado"):
        linhas.append(_formatar_entrada(e1, "Entrada 1"))
    else:
        linhas.append("Entrada 1: nenhum mercado bateu o edge mínimo.")
    if e2 and e2.get("mercado"):
        linhas.append("")
        linhas.append(_formatar_entrada(e2, "Entrada 2"))

    return "\n".join(linhas)


def _enviar_telegram(chat_id: str, texto: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": texto,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
        return True
    except urllib.error.HTTPError as e:
        try:
            corpo_erro = e.read().decode("utf-8", errors="replace")
        except Exception:
            corpo_erro = "(sem corpo)"
        print(f"[TELEGRAM] Erro HTTP ao publicar em {chat_id}: {e.code} {corpo_erro}")
        return False
    except Exception as e:
        print(f"[TELEGRAM] Erro ao publicar em {chat_id}: {e}")
        return False


def publicar_recomendacao_publica(resultado: dict) -> bool:
    """Publica no canal público -- só quando existe recomendação real."""
    if not TELEGRAM_CHAT_ID:
        return False
    e1 = (resultado.get("dupla_de_elite") or {}).get("entrada_1")
    if not e1 or not e1.get("mercado"):
        return False
    return _enviar_telegram(TELEGRAM_CHAT_ID, montar_mensagem_publica(resultado))


def publicar_ouvidoria(resultado: dict) -> bool:
    """Publica na ouvidoria -- TODO bilhete gerado, com ou sem recomendação."""
    if not TELEGRAM_CHAT_ID_OUVIDORIA:
        return False
    return _enviar_telegram(TELEGRAM_CHAT_ID_OUVIDORIA, montar_mensagem_ouvidoria(resultado))
