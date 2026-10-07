import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from urllib.request import pathname2url

ROUTER_WAIT_SECONDS = 1.5
ROUTER_POLL_SECONDS = 0.05
SQLITE_TIMEOUT = 0.5
HELD_CODE = "HELD_MODEL"
ROUTER_TABLES = ("router_decisions", "router_usage")
DECISION_COLUMNS = (
    "decision_id",
    "timestamp",
    "session_hint",
    "request_index_in_session",
    "requested_model",
    "chosen_model",
    "chosen_effort",
    "mode",
    "reason_codes",
    "signal_values",
    "action",
    "applied",
)
USAGE_COLUMNS = (
    "decision_id",
    "model_reported",
    "input_tokens",
    "output_tokens",
    "status",
    "cost_usd",
    "baseline_cost_usd",
)


def router_db_path():
    configured = os.environ.get("ROUTER_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".tamias" / "router.sqlite3"


def timestamp_key(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed.astimezone().timestamp()


def connect_ro(path):
    uri = "file:" + pathname2url(str(path)) + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT)


def columns_of(conn, table):
    if table not in ROUTER_TABLES:
        return set()
    try:
        rows = conn.execute("PRAGMA table_info(%s)" % table).fetchall()
    except Exception:
        return set()
    return {str(row[1]) for row in rows}


def select_rows(conn, table, columns, where="", params=(), order=""):
    available = columns_of(conn, table)
    chosen = [name for name in columns if name in available]
    if not chosen:
        return []
    sql = "SELECT %s FROM %s" % (", ".join(chosen), table)
    if where:
        sql += " WHERE " + where
    if order:
        sql += " ORDER BY " + order
    cursor = conn.execute(sql, params)
    names = [column[0] for column in cursor.description or []]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def newest_decision(conn, since=None):
    fresh = []
    for order in ("timestamp DESC LIMIT 200", "rowid DESC LIMIT 200", "LIMIT 200", ""):
        try:
            rows = select_rows(conn, "router_decisions", DECISION_COLUMNS, order=order)
        except Exception:
            continue
        for row in rows:
            key = timestamp_key(row.get("timestamp"))
            if key is None:
                continue
            if since is not None and key <= since:
                continue
            fresh.append((key, row))
        if fresh:
            break
    if not fresh:
        return None
    return max(fresh, key=lambda pair: pair[0])[1]


def usage_for(conn, decision_id):
    if decision_id is None:
        return None
    for order in ("rowid DESC", ""):
        try:
            rows = select_rows(
                conn, "router_usage", USAGE_COLUMNS, "decision_id = ?", (decision_id,), order
            )
        except Exception:
            continue
        if rows:
            return rows[0]
    return None


def lookup_router(since):
    path = router_db_path()
    try:
        if not path.exists():
            return None, None
        conn = connect_ro(path)
    except Exception:
        return None, None
    try:
        decision = newest_decision(conn, since)
        usage = usage_for(conn, decision.get("decision_id") if decision else None)
    except Exception:
        return None, None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return decision, usage


def wait_for_router(since, timeout=None, interval=None, sleep=None, clock=None):
    if timeout is None:
        timeout = ROUTER_WAIT_SECONDS
    if interval is None:
        interval = ROUTER_POLL_SECONDS
    if sleep is None:
        sleep = time.sleep
    if clock is None:
        clock = time.monotonic
    deadline = clock() + timeout
    while True:
        decision, usage = lookup_router(since)
        if decision is not None and usage is not None:
            return decision, usage
        if clock() >= deadline:
            return decision, usage
        sleep(interval)


def latest_any():
    decision, _ = lookup_router(None)
    return decision


def router_mode():
    try:
        mode = str((latest_any() or {}).get("mode") or "").strip().lower()
    except Exception:
        return "n/a"
    return mode if mode in {"shadow", "active"} else "n/a"


def json_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except (TypeError, ValueError):
            return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    items = []
    for entry in raw:
        if isinstance(entry, dict):
            label = entry.get("code") or entry.get("name") or entry.get("reason")
            if label:
                items.append(str(label))
        elif entry is not None:
            items.append(str(entry))
    return items


def json_object(value):
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def flag(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "applied"}


def reason_codes(decision):
    if decision is None:
        return []
    return json_list(decision.get("reason_codes"))


def signals_of(decision):
    if decision is None:
        return {}
    return json_object(decision.get("signal_values"))


def routing_label(decision):
    if decision is None:
        return "n/a"
    codes = reason_codes(decision)
    if flag(decision.get("applied")) and HELD_CODE not in codes:
        return "SWITCHED"
    if HELD_CODE in codes:
        return "HELD"
    return "STAY"
