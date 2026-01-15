"""데이터베이스 유틸리티"""
import threading
import mysql.connector
from config import DB_CONFIG, DB_BOOTSTRAP_CONFIG

_db_lock = threading.Lock()


def get_db():
    """데이터베이스 연결 반환"""
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    """데이터베이스 및 테이블 초기화"""
    with _db_lock:
        conn = mysql.connector.connect(**DB_BOOTSTRAP_CONFIG)
        cur = conn.cursor()
        cur.execute("CREATE DATABASE IF NOT EXISTS face_id")
        cur.execute("USE face_id")
        
        # persons 테이블
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                serial_number VARCHAR(100) NOT NULL,
                role ENUM('employee', 'patient') NOT NULL,
                registered_at DATETIME NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                embedding LONGBLOB NOT NULL
            )
            """
        )
        
        # ai_event_logs 테이블
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_event_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                task VARCHAR(32) NOT NULL,
                label VARCHAR(128),
                score FLOAT,
                source_id VARCHAR(64),
                payload_json TEXT,
                video_path VARCHAR(255),
                seen_at DATETIME NOT NULL
            )
            """
        )
        
        # video_path 컬럼 추가 (이미 있으면 무시)
        try:
            cur.execute("ALTER TABLE ai_event_logs ADD COLUMN video_path VARCHAR(255)")
        except mysql.connector.Error:
            pass
        
        cur.close()
        conn.close()


def get_db_lock():
    """DB 락 반환"""
    return _db_lock
