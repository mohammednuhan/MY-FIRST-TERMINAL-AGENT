import json
import os
import socket
import sqlite3
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import pathname2url

from dotenv import load_dotenv
from openai import OpenAI
from rich import box
from rich.align import Align
from rich.box import Box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

APP_VERSION = "v0.1"
APP_NAME = "T A M I A S"
APP_SUBTITLE = "MODEL ROUTING TERMINAL"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "poolside/laguna-s-2.1:free"
PROBE_TIMEOUT = 0.3
ROUTER_WAIT_SECONDS = 1.5
ROUTER_POLL_SECONDS = 0.05
SQLITE_TIMEOUT = 0.5
ERROR_LIMIT = 200
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
PROMPT = "[bold green]tamias> [/bold green] "
BLOCK = "\u2588"
SOLID_ROWS = "\n".join(
    [BLOCK * 4, BLOCK + " " + BLOCK * 2, BLOCK * 4, BLOCK + " " + BLOCK * 2,
     BLOCK + " " + BLOCK * 2, BLOCK * 4, BLOCK + " " + BLOCK * 2, BLOCK * 4]
)


def solid_box():
    try:
        return Box(SOLID_ROWS)
    except TypeError:
        return Box(*(BLOCK,) * 8)


TAMIAS_BOX = solid_box()


def base_url():
    return os.environ.get("AGENT_BASE_URL", "").strip() or DEFAULT_BASE_URL


def model_name():
    return os.environ.get("AGENT_MODEL", "").strip() or DEFAULT_MODEL


def short_error(exc, secret=None, limit=ERROR_LIMIT):
    message = " ".join(str(exc).split())
    if secret:
        message = message.replace(secret, "[redacted]")
    if len(message) > limit:
        message = message[:limit].rstrip() + "..."
    kind = type(exc).__name__
    return "%s: %s" % (kind, message) if message else kind


def answer_text(response):
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def attr_value(source, name):
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def number_of(source, *names):
    for name in names:
        value = attr_value(source, name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def tokens_of(response):
    usage = attr_value(response, "usage")
    return number_of(usage, "input_tokens", "prompt_tokens"), number_of(
        usage, "output_tokens", "completion_tokens"
    )


def display_number(value):
    if value is None:
        return "n/a"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return ("%.4f" % number).rstrip("0").rstrip(".")


def count_text(value):
    if value is None:
        return "n/a"
    return "{:,.0f}".format(float(value))


def format_cost(value):
    if value is None:
        return "UNKNOWN"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if number == 0.0:
        return "FREE"
    return "$%.6f" % number


def field(label, value):
    return "%-8s: %s" % (label, value)


def flag(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "applied"}


class CostSum:
    def __init__(self):
        self.known = 0.0
        self.entries = 0
        self.unknown = 0

    def add(self, value):
        if value is None:
            self.unknown += 1
            return
        try:
            self.known += float(value)
        except (TypeError, ValueError):
            self.unknown += 1
            return
        self.entries += 1

    def text(self):
        if not self.entries and not self.unknown:
            return "n/a"
        if not self.entries:
            return "UNKNOWN (known part)"
        base = format_cost(self.known)
        if self.unknown:
            return "%s (known part)" % base
        return base


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
        usage = usage_for(conn, attr_value(decision, "decision_id"))
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


def reason_codes(decision):
    return json_list(attr_value(decision, "reason_codes"))


def signals_of(decision):
    return json_object(attr_value(decision, "signal_values"))


def routing_label(decision):
    if HELD_CODE in reason_codes(decision):
        return "HELD"
    if flag(attr_value(decision, "applied")):
        return "SWITCHED"
    return "STAY"


def task_text(decision):
    if decision is None:
        return "n/a"
    signals = signals_of(decision)
    tier = signals.get("classifier_tier")
    score = signals.get("classifier_score")
    if tier in (None, "") and score is None:
        return "n/a"
    if tier in (None, ""):
        text = "score %s" % display_number(score)
    elif score is None:
        text = str(tier)
    else:
        text = "%s (score %s)" % (tier, display_number(score))
    codes = reason_codes(decision)
    if codes:
        text += "  " + " ".join(codes)
    return text


def router_mode():
    try:
        mode = str(attr_value(latest_any(), "mode") or "").strip().lower()
    except Exception:
        return "n/a"
    return mode if mode in {"shadow", "active"} else "n/a"


def probe_router(url, timeout=PROBE_TIMEOUT):
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
        if not host:
            return "offline"
        if port is None:
            port = 443 if parts.scheme == "https" else 80
        with socket.create_connection((host, port), timeout=timeout):
            return "online"
    except Exception:
        return "offline"


def print_banner(console, url, mode):
    body = Align.center("\n".join(["", APP_NAME, APP_SUBTITLE, ""]), vertical="middle")
    console.print(Panel(body, box=TAMIAS_BOX, style="bold bright_cyan", padding=(0, 1)))
    state = probe_router(url)
    colour = "green" if state == "online" else "red"
    console.print(
        Align.center(
            "[dim]%s | router: [%s]%s[/%s] | mode: %s[/dim]"
            % (APP_VERSION, colour, state, colour, mode)
        )
    )
    console.print()


class Session:
    def __init__(self):
        self.history = []
        self.turns = 0
        self.tokens_in = 0.0
        self.tokens_out = 0.0
        self.routed_cost = CostSum()
        self.baseline_cost = CostSum()
        self.switches = 0
        self.models = Counter()
        self.last_decision = None

    def clear(self):
        self.history = []

    def record(self, model, decision, usage, tokens_in, tokens_out):
        self.turns += 1
        if tokens_in is None and usage is not None:
            tokens_in = number_of(usage, "input_tokens", "prompt_tokens")
        if tokens_out is None and usage is not None:
            tokens_out = number_of(usage, "output_tokens", "completion_tokens")
        self.tokens_in += tokens_in or 0.0
        self.tokens_out += tokens_out or 0.0
        if model:
            self.models[model] += 1
        if decision is not None:
            self.last_decision = decision
            if routing_label(decision) == "SWITCHED":
                self.switches += 1
        if usage is not None:
            self.routed_cost.add(number_of(usage, "cost_usd"))
            self.baseline_cost.add(number_of(usage, "baseline_cost_usd"))

    def session_text(self):
        turns = "%d turn%s" % (self.turns, "" if self.turns == 1 else "s")
        switches = "%d switch%s" % (self.switches, "" if self.switches == 1 else "es")
        return "%s | in %s out %s | routed %s | baseline %s | %s" % (
            turns,
            count_text(self.tokens_in),
            count_text(self.tokens_out),
            self.routed_cost.text(),
            self.baseline_cost.text(),
            switches,
        )


def routing_panel(session, requested, used, decision, usage, tokens_in, tokens_out):
    if decision is None:
        model_text = "requested %s / used %s  n/a" % (requested, used)
        log_check = "n/a"
    else:
        model_text = "requested %s / used %s  %s" % (requested, used, routing_label(decision))
        logged = attr_value(decision, "chosen_model")
        if not logged:
            log_check = "n/a"
        else:
            log_check = "MISMATCH" if logged != used else "OK"
    if tokens_in is None and tokens_out is None:
        token_text = "n/a"
    else:
        token_text = "in %s  out %s" % (
            count_text(tokens_in) if tokens_in is not None else "n/a",
            count_text(tokens_out) if tokens_out is not None else "n/a",
        )
    if decision is None:
        cost_text = "n/a"
        baseline_text = "n/a"
    else:
        cost_text = format_cost(number_of(usage, "cost_usd") if usage else None)
        baseline_text = format_cost(number_of(usage, "baseline_cost_usd") if usage else None)
    lines = [
        field("task", task_text(decision)),
        field("model", model_text),
        field("log check", log_check),
        field("tokens", token_text),
        field("cost", "%-9s baseline: %s" % (cost_text, baseline_text)),
        field("session", session.session_text()),
    ]
    return Panel(
        Text("\n".join(lines)),
        title="ROUTING",
        title_align="left",
        box=box.ASCII,
        expand=True,
        padding=(0, 1),
    )


def print_why(console, session):
    decision = session.last_decision
    signals = signals_of(decision) if decision is not None else {}
    codes = reason_codes(decision) if decision is not None else []
    numbers = []
    for key in sorted(signals):
        if key in {"classifier_tier", "classifier_score"}:
            continue
        value = signals[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numbers.append("%s=%s" % (key, display_number(value)))
    lines = [
        field("tier", str(signals.get("classifier_tier") or "n/a")),
        field("score", display_number(signals.get("classifier_score"))),
        field("reasons", " ".join(codes) if codes else "n/a"),
        field("signals", " ".join(numbers) if numbers else "n/a"),
    ]
    for line in lines:
        console.print(Text(line))


def print_stats(console, session):
    console.print(Text(field("session", session.session_text())))
    if not session.models:
        console.print(Text(field("models", "n/a")))
        return
    for name, count in sorted(session.models.items()):
        console.print(Text(field("model", "%s  x%d" % (name, count))))


def print_tokens(console, tokens_in, tokens_out):
    if tokens_in is None and tokens_out is None:
        console.print(Text(field("tokens", "n/a")))
        return
    console.print(
        Text(
            field(
                "tokens",
                "in %s  out %s"
                % (
                    count_text(tokens_in) if tokens_in is not None else "n/a",
                    count_text(tokens_out) if tokens_out is not None else "n/a",
                ),
            )
        )
    )


def handle_command(console, session, line, state):
    parts = line.split()
    name = parts[0]
    if name == "/new":
        session.clear()
        console.print("[dim]history cleared[/dim]")
    elif name == "/stats":
        print_stats(console, session)
    elif name == "/why":
        print_why(console, session)
    elif name == "/routing":
        if len(parts) == 1:
            console.print("[dim]routing panel: %s[/dim]" % ("on" if state["routing"] else "off"))
        elif parts[1] in {"on", "off"}:
            state["routing"] = parts[1] == "on"
            console.print("[dim]routing panel: %s[/dim]" % parts[1])
        else:
            console.print("[red]usage: /routing on|off[/red]")
    else:
        console.print("[red]unknown command: %s[/red]" % name)


def chat_loop(console, client, session, requested, secret=None, input_fn=None):
    state = {"routing": True}
    if input_fn is None:
        input_fn = console.input
    while True:
        try:
            user_input = input_fn(PROMPT)
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user_input.strip():
            continue
        if user_input == "/exit":
            break
        if user_input.startswith("/"):
            handle_command(console, session, user_input, state)
            continue
        session.history.append({"role": "user", "content": user_input})
        sent_at = time.time()
        try:
            response = client.chat.completions.create(
                model=requested, messages=session.history
            )
        except Exception as exc:
            console.print("[bold red]error:[/bold red]", Text(short_error(exc, secret)))
            if session.history and session.history[-1].get("role") == "user":
                session.history.pop()
            continue
        answer = answer_text(response)
        if not answer or not answer.strip():
            answer = "(empty reply)"
        used = str(attr_value(response, "model") or requested)
        tokens_in, tokens_out = tokens_of(response)
        console.print("[bold cyan]Agent:[/bold cyan]")
        console.print(Markdown(answer))
        decision, usage = (None, None)
        if state["routing"]:
            decision, usage = wait_for_router(sent_at)
        session.record(used, decision, usage, tokens_in, tokens_out)
        session.history.append({"role": "assistant", "content": answer})
        if state["routing"]:
            console.print(
                routing_panel(session, requested, used, decision, usage, tokens_in, tokens_out)
            )
        else:
            print_tokens(console, tokens_in, tokens_out)


def main():
    load_dotenv()
    console = Console()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        console.print("[bold red]config:[/bold red] OPENROUTER_API_KEY is not set (see .env.example)")
        return 1
    requested = model_name()
    url = base_url()
    try:
        client = OpenAI(base_url=url, api_key=api_key)
    except Exception as exc:
        console.print("[bold red]config:[/bold red]", Text(short_error(exc, api_key)))
        return 1
    print_banner(console, url, router_mode())
    session = Session()
    chat_loop(console, client, session, requested, secret=api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
