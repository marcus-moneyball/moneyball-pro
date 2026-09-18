# api/eventos.py
import json

def registrar_evento(conn, email, tipo_evento, metadata=None):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO eventos_usuario (email, tipo_evento, metadata) VALUES (%s, %s, %s)",
            (email, tipo_evento, json.dumps(metadata or {}, ensure_ascii=False)),
        )
    conn.commit()
