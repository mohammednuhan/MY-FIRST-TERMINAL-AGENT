import os
import socket
import time
from collections import Counter
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

import router_log
import ui

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "poolside/laguna-s-2.1:free"
PROBE_TIMEOUT = 0.3
ERROR_LIMIT = 200


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


def answer_text(response):
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def tokens_of(response):
    usage = attr_value(response, "usage")
    return number_of(usage, "input_tokens", "prompt_tokens"), number_of(
        usage, "output_tokens", "completion_tokens"
    )


def probe_router(url, timeout=PROBE_TIMEOUT):
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if not host:
            return False
        port = parts.port
        if port is None:
            port = 443 if parts.scheme == "https" else 80
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def to_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def usage_has(usage, key):
    return usage is not None and key in usage


class CostSum:
    def __init__(self):
        self.total = 0.0
        self.known = 0
        self.unknown = 0

    def add(self, value):
        number = to_number(value)
        if number is None:
            self.unknown += 1
            return
        self.total += number
        self.known += 1


class Session:
    def __init__(self):
        self.history = []
        self.turns = 0
        self.switches = 0
        self.models = Counter()
        self.last_decision = None
        self.routed = CostSum()
        self.baseline = CostSum()
        self.pair_cost = 0.0
        self.pair_baseline = 0.0
        self.pairs = 0

    def clear(self):
        self.history = []

    def record(self, model, decision, usage):
        self.turns += 1
        if model:
            self.models[model] += 1
        if decision is not None:
            self.last_decision = decision
            if router_log.routing_label(decision) == "SWITCHED":
                self.switches += 1
        if usage is None:
            return
        cost = usage.get("cost_usd") if "cost_usd" in usage else None
        baseline = usage.get("baseline_cost_usd") if "baseline_cost_usd" in usage else None
        if "cost_usd" in usage:
            self.routed.add(cost)
        if "baseline_cost_usd" in usage:
            self.baseline.add(baseline)
        if "cost_usd" in usage and "baseline_cost_usd" in usage:
            cost_number = to_number(cost)
            baseline_number = to_number(baseline)
            if cost_number is not None and baseline_number is not None:
                self.pairs += 1
                self.pair_cost += cost_number
                self.pair_baseline += baseline_number


def handle_command(console, session, line, state):
    parts = line.split()
    name = parts[0]
    if name == "/new":
        session.clear()
        ui.message(console, "history cleared")
    elif name == "/stats":
        ui.print_stats(console, session)
    elif name == "/why":
        ui.print_why(console, session)
    elif name == "/routing":
        if len(parts) == 1:
            ui.message(console, "routing panel: %s" % ("on" if state["routing"] else "off"))
        elif parts[1] in {"on", "off"}:
            state["routing"] = parts[1] == "on"
            ui.message(console, "routing panel: %s" % parts[1])
        else:
            ui.error(console, "usage: /routing on|off")
    else:
        ui.error(console, "unknown command: " + name)


def chat_loop(console, client, session, requested, secret=None, input_fn=None):
    state = {"routing": True}
    if input_fn is None:
        input_fn = console.input
    while True:
        try:
            user_input = input_fn(ui.PROMPT)
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
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=requested, messages=session.history
            )
        except Exception as exc:
            ui.error(console, short_error(exc, secret), headline="error")
            if session.history and session.history[-1].get("role") == "user":
                session.history.pop()
            continue
        latency = time.perf_counter() - started
        answer = answer_text(response)
        if not answer or not answer.strip():
            answer = "(empty reply)"
        used = str(attr_value(response, "model") or requested)
        tokens_in, tokens_out = tokens_of(response)
        decision, usage = (None, None)
        if state["routing"]:
            try:
                decision, usage = router_log.wait_for_router(sent_at)
            except Exception:
                decision, usage = (None, None)
        if usage is not None:
            if tokens_in is None:
                tokens_in = number_of(usage, "input_tokens")
            if tokens_out is None:
                tokens_out = number_of(usage, "output_tokens")
        session.record(used, decision, usage)
        session.history.append({"role": "assistant", "content": answer})
        ui.render_turn(
            console,
            session,
            user_input,
            answer,
            requested,
            used,
            decision,
            usage,
            tokens_in,
            tokens_out,
            latency,
            routing=state["routing"],
        )


def main():
    load_dotenv()
    console = Console()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        ui.error(
            console,
            "OPENROUTER_API_KEY is not set (see .env.example)",
            headline="config",
        )
        return 1
    requested = model_name()
    url = base_url()
    try:
        client = OpenAI(base_url=url, api_key=api_key)
    except Exception as exc:
        ui.error(console, short_error(exc, api_key), headline="config")
        return 1
    ui.print_banner(console, online=probe_router(url), mode=router_log.router_mode())
    session = Session()
    try:
        chat_loop(console, client, session, requested, secret=api_key)
    except KeyboardInterrupt:
        console.print()
    except Exception as exc:
        ui.error(console, short_error(exc, api_key), headline="stopped")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
