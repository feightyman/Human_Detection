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


def delete_alarms(ids: List[int], remove_snapshots: bool = True) -> Tuple[bool, str, int]:
    """
    批量删除指定 id 的报警记录（同步执行，调用方建议放到后台线程）。
    返回 (是否成功, 错误信息, 实际删除条数)。
    """
    if not ids:
        return True, "", 0
    try:
        snapshot_paths: List[str] = []
        with _db_lock:
            conn = sqlite3.connect(str(DB_PATH))
            # 用 ? 占位符防注入；构造 (?, ?, ?, ...)
            placeholders = ",".join("?" * len(ids))
            if remove_snapshots:
                cursor = conn.execute(
                    f"SELECT snapshot_path FROM alarm_events WHERE id IN ({placeholders})",
                    ids,
                )
                snapshot_paths = [r[0] for r in cursor.fetchall() if r and r[0]]
            cursor = conn.execute(
                f"DELETE FROM alarm_events WHERE id IN ({placeholders})", ids,
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
        # 数据库锁释放后再做文件 IO，避免阻塞写入
        for p in snapshot_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"[AlarmDB] 抓拍文件删除失败（忽略） {p}: {e}")
        print(f"[AlarmDB] 已删除报警记录 {deleted} 条（请求 {len(ids)} 条）")
        return True, "", deleted
    except Exception as e:
        err = f"[AlarmDB] 批量删除失败: {e}"
        print(err)
        return False, str(e), 0


def delete_all_alarms(remove_snapshots: bool = True) -> Tuple[bool, str, int]:
    """
    清空全部报警记录（同步执行）。
    返回 (是否成功, 错误信息, 删除条数)。
    """
    try:
        snapshot_paths: List[str] = []
        with _db_lock:
            conn = sqlite3.connect(str(DB_PATH))
            if remove_snapshots:
                cursor = conn.execute("SELECT snapshot_path FROM alarm_events")
                snapshot_paths = [r[0] for r in cursor.fetchall() if r and r[0]]
            cursor = conn.execute("DELETE FROM alarm_events")
            deleted = cursor.rowcount
            # 重置自增 id，使下次插入序号从 1 开始
            conn.execute("DELETE FROM sqlite_sequence WHERE name='alarm_events'")
            conn.commit()
            conn.close()
        for p in snapshot_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"[AlarmDB] 抓拍文件删除失败（忽略） {p}: {e}")
        print(f"[AlarmDB] 已清空所有报警记录，共 {deleted} 条")
        return True, "", deleted
    except Exception as e:
        err = f"[AlarmDB] 清空失败: {e}"
        print(err)
        return False, str(e), 0
