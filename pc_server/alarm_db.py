"""
alarm_db.py
报警事件持久化模块 —— SQLite 数据库操作

表结构 alarm_events：
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp       TEXT    ISO 8601 格式时间戳
  intrusion_count INTEGER 入侵人数
  snapshot_path   TEXT    抓拍图片的本地路径

所有写操作在后台线程执行，通过回调函数通知调用方写入结果。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# 数据库文件路径：与本文件同目录
DB_PATH = Path(__file__).parent / "alarm_log.db"

# 抓拍图片保存目录
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

# 模块级锁，保证多线程写入安全
_db_lock = threading.Lock()


def init_db() -> None:
    """初始化数据库：建表（IF NOT EXISTS），同时创建 snapshots 目录。"""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alarm_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                intrusion_count INTEGER NOT NULL,
                snapshot_path   TEXT    NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    print(f"[AlarmDB] 数据库已初始化: {DB_PATH}")
    print(f"[AlarmDB] 抓拍目录: {SNAPSHOT_DIR}")


def insert_alarm(
    timestamp: str,
    count: int,
    snapshot_path: str,
    on_success: Optional[Callable[[], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
) -> None:
    """
    插入一条报警记录（在后台线程中执行，不阻塞调用线程）。

    参数：
      timestamp     ISO 8601 格式时间字符串
      count         入侵人数
      snapshot_path 抓拍图片本地路径
      on_success    写入成功后的回调（在后台线程中调用）
      on_error      写入失败后的回调，参数为错误信息（在后台线程中调用）
    """
    def _write():
        try:
            with _db_lock:
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute(
                    "INSERT INTO alarm_events (timestamp, intrusion_count, snapshot_path) "
                    "VALUES (?, ?, ?)",
                    (timestamp, count, snapshot_path),
                )
                conn.commit()
                conn.close()
            print(f"[AlarmDB] 已记录报警: {timestamp}, {count} 人, {snapshot_path}")
            if on_success is not None:
                on_success()
        except Exception as e:
            err_msg = f"[AlarmDB] 写入失败: {e}"
            print(err_msg)
            if on_error is not None:
                on_error(str(e))

    threading.Thread(target=_write, daemon=True, name="AlarmDBWriter").start()


def query_alarms(limit: int = 100, offset: int = 0) -> List[Tuple]:
    """
    查询报警历史记录（按时间倒序）。

    返回列表，每项为 (id, timestamp, intrusion_count, snapshot_path)。
    """
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute(
            "SELECT id, timestamp, intrusion_count, snapshot_path "
            "FROM alarm_events ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()
        conn.close()
    return rows


def get_alarm_count() -> int:
    """获取报警记录总数。"""
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute("SELECT COUNT(*) FROM alarm_events")
        count = cursor.fetchone()[0]
        conn.close()
    return count
