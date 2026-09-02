"""
Vínculo entre e-mail (fonte de verdade: Ghost/Stripe) e conta do Telegram,
pra liberar/remover acesso automático ao grupo PRO.

Fluxo:
1. Usuário PRO pede um link de vínculo no app (gerar_token_vinculo).
2. Ele clica no link (t.me/<bot>?start=<token>), o Telegram manda um
   /start <token> pro webhook_telegram em main.py.
3. consumir_token_vinculo resolve o token -> e-mail, salva o telegram_user_id
   em app_users, e o bot manda o convite do grupo PRO de volta pro usuário.
4. Se o Ghost rebaixar esse e-mail (cancelamento), remover_do_grupo_pro tira
   a pessoa do grupo usando o telegram_user_id salvo.

Tudo aqui é best-effort com o Telegram (rede pode falhar) -- funções sempre
retornam bool/None, nunca lançam exceção pro chamador.
"""
import os
import json
import secrets
import urllib.request
import urllib.error
from typing import Optional

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_GROUP_ID_PRO = os.getenv("TELEGRAM_GROUP_ID_PRO")
TELEGRAM_BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME")  # ex: "nexusmoneyballbot", sem @

_ALFABETO_TOKEN = "23456789ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz"


def _is_postgres(conn) -> bool:
    return type(conn).__module__.startswith("psycopg2")


def _gerar_token(tamanho: int = 24) -> str:
    return "".join(secrets.choice(_ALFABETO_TOKEN) for _ in range(tamanho))


def gerar_token_vinculo(conn, email: str) -> Optional[str]:
    """Cria um token de uso único pra vincular o Telegram a esse e-mail."""
    if conn is None:
        return None
    email = email.strip().lower()
    placeholder = "%s" if _is_postgres(conn) else "?"
    token = _gerar_token()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            INSERT INTO telegram_link_tokens (token, email, usado)
            VALUES ({placeholder}, {placeholder}, FALSE)
            """,
            (token, email),
        )
        conn.commit()
        return token
    except Exception as e:
        print(f"[TELEGRAM VINCULO] Falha ao gerar token pra '{email}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def montar_link_vinculo(token: str) -> Optional[str]:
    if not TELEGRAM_BOT_USERNAME or not token:
        return None
    return f"https://t.me/{TELEGRAM_BOT_USERNAME}?start={token}"


def consumir_token_vinculo(conn, token: str) -> Optional[str]:
    """
    Resolve o token pro e-mail dono dele e marca como usado, de forma atômica
    (só marca se ainda não tinha sido usado). Retorna o e-mail, ou None se o
    token não existir ou já tiver sido consumido.
    """
    if conn is None or not token:
        return None
    placeholder = "%s" if _is_postgres(conn) else "?"
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE telegram_link_tokens
            SET usado = TRUE
            WHERE token = {placeholder} AND usado = FALSE
            RETURNING email
            """,
            (token,),
        )
        row = cursor.fetchone()
        conn.commit()
        return row[0] if row else None
    except Exception as e:
        print(f"[TELEGRAM VINCULO] Falha ao consumir token: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def salvar_telegram_user_id(conn, email: str, telegram_user_id: int) -> bool:
    if conn is None:
        return False
    email = email.strip().lower()
    placeholder = "%s" if _is_postgres(conn) else "?"
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE app_users SET telegram_user_id = {placeholder} WHERE email = {placeholder}",
            (telegram_user_id, email),
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[TELEGRAM VINCULO] Falha ao salvar telegram_user_id de '{email}': {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def obter_plano_e_telegram(conn, email: str) -> dict:
    """Retorna {'plano': ..., 'telegram_user_id': ...} pra decisões de vínculo/remoção."""
    if conn is None:
        return {"plano": "free", "telegram_user_id": None}
    email = email.strip().lower()
    placeholder = "%s" if _is_postgres(conn) else "?"
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT plano, telegram_user_id FROM app_users WHERE email = {placeholder}",
            (email,),
        )
        row = cursor.fetchone()
        if not row:
            return {"plano": "free", "telegram_user_id": None}
        return {"plano": row[0], "telegram_user_id": row[1]}
    except Exception as e:
        print(f"[TELEGRAM VINCULO] Falha ao consultar '{email}': {e}")
        return {"plano": "free", "telegram_user_id": None}


def _chamar_telegram(metodo: str, corpo: dict) -> Optional[dict]:
    if not TELEGRAM_BOT_TOKEN:
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{metodo}"
    payload = json.dumps(corpo).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            corpo_erro = e.read().decode("utf-8", errors="replace")
        except Exception:
            corpo_erro = "(sem corpo)"
        print(f"[TELEGRAM API] Erro em {metodo}: {e.code} {corpo_erro}")
        return None
    except Exception as e:
        print(f"[TELEGRAM API] Erro em {metodo}: {e}")
        return None


def enviar_convite_grupo_pro(telegram_user_id: int) -> bool:
    """
    Gera um convite de uso único pro grupo PRO e manda por DM pro usuário
    (o chat_id da DM é o próprio telegram_user_id, funciona porque ele acabou
    de mandar /start pro bot).
    """
    if not TELEGRAM_GROUP_ID_PRO:
        print("[TELEGRAM VINCULO] TELEGRAM_GROUP_ID_PRO não configurado -- convite não enviado.")
        return False

    convite = _chamar_telegram("createChatInviteLink", {
        "chat_id": TELEGRAM_GROUP_ID_PRO,
        "member_limit": 1,
        "name": f"convite-{telegram_user_id}",
    })
    if not convite or not convite.get("ok"):
        return False

    link = convite["result"]["invite_link"]
    resposta = _chamar_telegram("sendMessage", {
        "chat_id": telegram_user_id,
        "text": f"🎉 Você é PRO! Aqui está seu convite exclusivo pro grupo:\n{link}\n\nEsse link é de uso único, pessoal e intransferível.",
    })
    return bool(resposta and resposta.get("ok"))


def remover_do_grupo_pro(telegram_user_id: int) -> bool:
    """Remove (kick, não ban permanente) o usuário do grupo PRO -- usado quando a assinatura cai."""
    if not TELEGRAM_GROUP_ID_PRO or not telegram_user_id:
        return False

    ban = _chamar_telegram("banChatMember", {
        "chat_id": TELEGRAM_GROUP_ID_PRO,
        "user_id": telegram_user_id,
    })
    if not ban or not ban.get("ok"):
        return False

    # unban logo em seguida -- vira um "kick", não um banimento permanente,
    # então a pessoa pode voltar se assinar de novo no futuro
    _chamar_telegram("unbanChatMember", {
        "chat_id": TELEGRAM_GROUP_ID_PRO,
        "user_id": telegram_user_id,
        "only_if_banned": True,
    })
    return True
