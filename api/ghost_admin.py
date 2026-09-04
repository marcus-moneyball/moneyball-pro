"""
Integração com a Ghost Admin API -- hoje só usada pra criar membros FREE
(lista de espera de quem bateu o limite gratuito e quis entrar na fila).

IMPORTANTE: isso nunca desbloqueia a cota. O único jeito de virar "pro" no
app_users continua sendo o webhook_ghost detectar status "paid"/"comped" --
um membro criado aqui entra como "free" no Ghost, e o sync (quando o
webhook do member.added disparar) só vai gravar plano='free' no Postgres,
exatamente como qualquer outro usuário gratuito.

JWT montado manualmente (HS256) pra não precisar adicionar PyJWT como
dependência nova -- é só um HMAC-SHA256 simples, dá pra fazer com stdlib.
"""
import os
import json
import time
import hmac
import hashlib
import base64
import urllib.request
import urllib.error
from typing import Optional

GHOST_ADMIN_API_URL = os.getenv("GHOST_ADMIN_API_URL")  # ex: https://blog.moneyballpro.com.br
GHOST_ADMIN_API_KEY = os.getenv("GHOST_ADMIN_API_KEY")  # formato "id:secret", da tela Integrations do Ghost


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _gerar_jwt_ghost() -> Optional[str]:
    if not GHOST_ADMIN_API_KEY or ":" not in GHOST_ADMIN_API_KEY:
        return None
    key_id, secret_hex = GHOST_ADMIN_API_KEY.split(":", 1)
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError:
        print("[GHOST ADMIN] GHOST_ADMIN_API_KEY malformada -- secret não é hex válido.")
        return None

    iat = int(time.time())
    header = {"alg": "HS256", "typ": "JWT", "kid": key_id}
    payload = {"iat": iat, "exp": iat + 300, "aud": "/admin/"}

    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    assinatura = hmac.new(secret, f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url(assinatura)}"


def criar_membro_free_ghost(email: str, nome: str = "") -> bool:
    """
    Cria (ou não faz nada, se já existir) um membro FREE no Ghost. Best-effort:
    nunca lança exceção, sempre retorna bool -- falha aqui não pode quebrar a
    experiência do usuário que só queria entrar na lista de espera.
    """
    if not GHOST_ADMIN_API_URL:
        print("[GHOST ADMIN] GHOST_ADMIN_API_URL não configurada -- membro não criado.")
        return False

    token = _gerar_jwt_ghost()
    if not token:
        print("[GHOST ADMIN] Não foi possível gerar o JWT -- confira GHOST_ADMIN_API_KEY.")
        return False

    url = f"{GHOST_ADMIN_API_URL.rstrip('/')}/ghost/api/admin/members/"
    corpo = json.dumps({"members": [{"email": email.strip().lower(), "name": nome}]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=corpo,
        headers={
            "Authorization": f"Ghost {token}",
            "Content-Type": "application/json",
            "Accept-Version": "v5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
        return True
    except urllib.error.HTTPError as e:
        corpo_erro = e.read().decode("utf-8", errors="replace")
        if e.code == 422 and "already exists" in corpo_erro.lower():
            # membro já existe no Ghost -- não é erro de verdade, é um no-op
            return True
        print(f"[GHOST ADMIN] Erro ao criar membro '{email}': {e.code} {corpo_erro}")
        return False
    except Exception as e:
        print(f"[GHOST ADMIN] Erro ao criar membro '{email}': {e}")
        return False
