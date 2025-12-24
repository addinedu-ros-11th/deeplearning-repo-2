import argparse
import json
import socket
from datetime import datetime

import mysql.connector


def get_db(args):
    return mysql.connector.connect(
        host=args.db_host,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
        autocommit=True,
    )


def init_table(args):
    conn = mysql.connector.connect(
        host=args.db_host,
        user=args.db_user,
        password=args.db_password,
        autocommit=True,
    )
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {args.db_name}")
    cur.execute(f"USE {args.db_name}")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_event_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            task VARCHAR(32) NOT NULL,
            label VARCHAR(128),
            score FLOAT,
            source_id VARCHAR(64),
            payload_json TEXT,
            seen_at DATETIME NOT NULL
        )
        """
    )
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="UDP event logger (AI inference results)")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", default="1234")
    parser.add_argument("--db-name", default="face_id")
    args = parser.parse_args()

    init_table(args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))

    try:
        while True:
            data, _addr = sock.recvfrom(65535)
            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            task = payload.get("task", "")
            source_id = payload.get("source_id") or None
            events = payload.get("events") or []
            if not events:
                continue

            now = datetime.now()
            with get_db(args) as conn:
                cur = conn.cursor()
                for event in events:
                    label = event.get("label")
                    score = event.get("score")
                    cur.execute(
                        """
                        INSERT INTO ai_event_logs (task, label, score, source_id, payload_json, seen_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            task,
                            label,
                            score,
                            source_id,
                            json.dumps(event, ensure_ascii=True),
                            now,
                        ),
                    )
                cur.close()
    finally:
        sock.close()


if __name__ == "__main__":
    main()
