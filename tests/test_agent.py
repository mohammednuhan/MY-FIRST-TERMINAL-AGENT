import io
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import agent
import router_log
import ui

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 110
ART_FIRST_LINE = "████████╗ █████╗ ███╗   ███╗██╗ █████╗ ███████╗"
BANNED_MARKERS = ("confidence", "intent", "context")


class Cp1252Stream(io.StringIO):
    encoding = "cp1252"


def make_console(width=WIDTH, stream=None):
    return Console(
        file=stream if stream is not None else io.StringIO(),
        width=width,
        no_color=True,
        legacy_windows=False,
        highlight=False,
    )


def render(renderable, width=WIDTH):
    console = make_console(width)
    console.print(renderable)
    return console.file.getvalue()


def call(func, *args, width=WIDTH, **kwargs):
    console = make_console(width)
    func(console, *args, **kwargs)
    return console.file.getvalue()


def make_response(content="ok", model="poolside/laguna-s-2.1:free", tokens_in=214, tokens_out=530):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, model, messages):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages]})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_client(responses):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(responses)))


def decision_row(
    decision_id="d1",
    timestamp=None,
    requested_model="poolside/laguna-s-2.1:free",
    chosen_model="poolside/laguna-s-2.1:free",
    mode="shadow",
    reason_codes=("STRONG_KW",),
    tier="HIGH",
    score=100,
    applied=1,
):
    signals = {"classifier_tier": tier, "classifier_score": score, "api_format": 1}
    return (
        decision_id,
        time.time() if timestamp is None else timestamp,
        "sess",
        1,
        requested_model,
        chosen_model,
        "high",
        mode,
        json.dumps(list(reason_codes)),
        json.dumps(signals),
        "rewrite",
        applied,
        None,
    )


def usage_row(decision_id="d1", cost=0.0, baseline=0.0, tokens_in=214, tokens_out=530):
    return (decision_id, "reported", tokens_in, tokens_out, 0, 0, "ok", cost, baseline, "")


DECISION_DDL = """
CREATE TABLE router_decisions (
    decision_id TEXT PRIMARY KEY,
    timestamp REAL,
    session_hint TEXT,
    request_index_in_session INTEGER,
    requested_model TEXT,
    chosen_model TEXT,
    chosen_effort TEXT,
    mode TEXT,
    reason_codes TEXT,
    signal_values TEXT,
    action TEXT,
    applied INTEGER,
    error TEXT
)
"""

USAGE_DDL = """
CREATE TABLE router_usage (
    decision_id TEXT,
    model_reported TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    status TEXT,
    cost_usd REAL,
    baseline_cost_usd REAL,
    notes TEXT
)
"""


def write_db(path, decisions=(), usages=(), decision_ddl=DECISION_DDL, usage_ddl=USAGE_DDL):
    conn = sqlite3.connect(str(path))
    try:
        if decision_ddl:
            conn.execute(decision_ddl)
            if decisions:
                conn.executemany(
                    "INSERT INTO router_decisions VALUES (%s)"
                    % ",".join("?" * len(decisions[0])),
                    decisions,
                )
        if usage_ddl:
            conn.execute(usage_ddl)
            if usages:
                conn.executemany(
                    "INSERT INTO router_usage VALUES (%s)" % ",".join("?" * len(usages[0])),
                    usages,
                )
        conn.commit()
    finally:
        conn.close()


def echoing_input(console, lines):
    source = iter(lines)

    def read(prompt):
        value = next(source)
        console.print("%s %s" % (prompt.rstrip(), value))
        return value

    return read


@pytest.fixture
def no_router_wait(monkeypatch):
    monkeypatch.setattr(router_log, "ROUTER_WAIT_SECONDS", 0.0)


def routing_output(decision, requested="poolside/laguna-s-2.1:free", used=None, width=WIDTH):
    used = used if used is not None else (
        decision.get("chosen_model") if decision else requested
    )
    return call(ui.print_routing, requested, used, decision, width=width)


def full_turn(prompt, reply, decision, usage, session=None, routing=True, width=WIDTH,
              requested="poolside/laguna-s-2.1:free", used=None, tokens_in=214.0,
              tokens_out=530.0, latency=1.5):
    session = session if session is not None else agent.Session()
    used = used if used is not None else (
        decision.get("chosen_model") if decision else requested
    )
    return call(
        ui.render_turn,
        session,
        prompt,
        reply,
        requested,
        used,
        decision,
        usage,
        tokens_in,
        tokens_out,
        latency,
        routing=routing,
        width=width,
    )


def test_env_defaults(monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    assert agent.base_url() == "https://openrouter.ai/api/v1"
    assert agent.model_name() == "poolside/laguna-s-2.1:free"
    monkeypatch.setenv("AGENT_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("AGENT_MODEL", "vendor/cheap:free")
    assert agent.base_url() == "http://127.0.0.1:1234/v1"
    assert agent.model_name() == "vendor/cheap:free"


def test_probe_router_offline_when_connect_fails(monkeypatch):
    def refuse(address, timeout=None):
        raise OSError("no route")

    monkeypatch.setattr(agent.socket, "create_connection", refuse)
    assert agent.probe_router("http://example.invalid:8123/v1") is False
    assert agent.probe_router("not-a-url") is False
    assert agent.probe_router("") is False


def test_probe_router_online_uses_host_and_port(monkeypatch):
    seen = {}

    @contextmanager
    def fake_connect(address, timeout=None):
        seen["address"] = address
        seen["timeout"] = timeout
        yield None

    monkeypatch.setattr(agent.socket, "create_connection", fake_connect)
    assert agent.probe_router("http://example.invalid:8123/v1") is True
    assert seen == {"address": ("example.invalid", 8123), "timeout": 0.3}
    agent.probe_router("https://example.invalid/v1")
    assert seen["address"] == ("example.invalid", 443)
    agent.probe_router("http://example.invalid/v1")
    assert seen["address"] == ("example.invalid", 80)


def test_banner_block_art_and_status():
    out = call(ui.print_banner, True, "shadow")
    assert ART_FIRST_LINE in out
    assert "MODEL ROUTING TERMINAL" in out
    assert "v0.1.0" in out
    assert "● ROUTER ONLINE   ● MODE SHADOW" in out
    assert "Traceback" not in out
    for line in out.splitlines():
        assert len(line) <= WIDTH


def test_banner_offline_and_unknown_mode():
    out = call(ui.print_banner, False, "weird")
    assert "● ROUTER OFFLINE   ● MODE N/A" in out


def test_banner_narrow_terminal_drops_block_art():
    out = call(ui.print_banner, True, "active", width=52)
    assert "█" not in out
    assert "MODEL ROUTING TERMINAL" in out
    assert "v0.1.0" in out
    assert "● ROUTER ONLINE   ● MODE ACTIVE" in out


def test_banner_ascii_fallback_when_console_cannot_encode():
    console = make_console(WIDTH, Cp1252Stream())
    ui.print_banner(console, online=True, mode="shadow")
    out = console.file.getvalue()
    out.encode("cp1252")
    assert "█" not in out
    assert "●" not in out
    assert "########" in out
    assert "* ROUTER ONLINE" in out
    assert out.splitlines()[0].startswith("+")
    assert "MODEL ROUTING TERMINAL" in out


def test_db_connection_is_read_only(tmp_path):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()], [usage_row()])
    conn = router_log.connect_ro(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM router_decisions").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO router_decisions VALUES (%s)" % ",".join("?" * 13),
                decision_row("d2"),
            )
    finally:
        conn.close()


def test_router_db_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "custom.sqlite3"))
    assert router_log.router_db_path() == tmp_path / "custom.sqlite3"
    monkeypatch.delenv("ROUTER_DB")
    assert router_log.router_db_path() == Path.home() / ".tamias" / "router.sqlite3"


def test_wait_polls_until_usage_row_appears(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    conn = sqlite3.connect(str(path))

    def late_usage(interval):
        conn.execute(
            "INSERT INTO router_usage VALUES (?,?,?,?,?,?,?,?,?,?)",
            usage_row(cost=0.25),
        )
        conn.commit()

    decision, usage = router_log.wait_for_router(
        time.time() - 5, timeout=5.0, interval=0.01, sleep=late_usage
    )
    conn.close()
    assert decision["chosen_model"] == "poolside/laguna-s-2.1:free"
    assert usage["cost_usd"] == 0.25


def test_wait_stops_at_timeout_without_usage(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    assert decision is not None
    assert usage is None


def test_wait_ignores_rows_older_than_request(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row(timestamp=time.time() - 600)], [usage_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    assert router_log.wait_for_router(time.time(), timeout=0.0) == (None, None)


def test_missing_db_gives_no_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "nope.sqlite3"))
    assert router_log.lookup_router(time.time() - 5) == (None, None)
    assert router_log.router_mode() == "n/a"


def test_broken_db_file_gives_no_data(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    path.write_text("this is not a database")
    monkeypatch.setenv("ROUTER_DB", str(path))
    assert router_log.lookup_router(time.time() - 5) == (None, None)
    assert router_log.router_mode() == "n/a"


def test_mode_read_from_log(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row(mode="active")], [usage_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    assert router_log.router_mode() == "active"
    other = tmp_path / "other.sqlite3"
    write_db(other, [decision_row(mode="weird")], [usage_row()])
    monkeypatch.setenv("ROUTER_DB", str(other))
    assert router_log.router_mode() == "n/a"


def test_missing_columns_do_not_crash(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(
        path,
        [("d1", time.time(), "poolside/laguna-s-2.1:free", 0.25)],
        [("d1", 11, 22, 0.125)],
        decision_ddl=(
            "CREATE TABLE router_decisions (decision_id TEXT, timestamp REAL,"
            " requested_model TEXT, cost_usd REAL)"
        ),
        usage_ddl=(
            "CREATE TABLE router_usage (decision_id TEXT, input_tokens INTEGER,"
            " output_tokens INTEGER, cost_usd REAL)"
        ),
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    assert decision is not None
    session = agent.Session()
    session.record("m", decision, usage)
    out = full_turn("a prompt", "a reply", decision, usage, session=session,
                    tokens_in=None, tokens_out=None)
    assert "complexity: n/a" in out
    assert "score: n/a" in out
    assert "decision: n/a" in out
    assert "laguna-s-2.1" in out
    assert "input tokens : n/a" in out
    assert "model cost   : $0.125000" in out
    assert "baseline cost: n/a" in out
    assert "savings      : n/a" in out
    assert "Traceback" not in out


def test_format_cost_rules():
    assert ui.format_cost(0.0) == "FREE"
    assert ui.format_cost(None) == "UNKNOWN"
    assert ui.format_cost("") == "UNKNOWN"
    assert ui.format_cost(0.25) == "$0.250000"


def test_savings_rules():
    assert ui.savings_percent(0.0, 0.0) == "0.0%"
    assert ui.savings_percent(0.0042, 0.011) == "61.8%"
    assert ui.savings_percent(None, 0.5) == "n/a"
    assert ui.savings_percent(0.5, None) == "n/a"
    assert ui.savings_percent(0.5, 0.0) == "n/a"
    assert ui.savings_percent("x", 0.5) == "n/a"


def test_cost_sum_never_counts_unknown_as_zero():
    total = agent.CostSum()
    assert ui.cost_sum_text(total) == "n/a"
    total.add(None)
    assert ui.cost_sum_text(total) == "UNKNOWN"
    assert total.total == 0.0
    total.add(0.0)
    assert ui.cost_sum_text(total) == "FREE (known part)"
    total.add(0.25)
    assert ui.cost_sum_text(total) == "$0.250000 (known part)"
    clean = agent.CostSum()
    clean.add(0.0)
    assert ui.cost_sum_text(clean) == "FREE"


def test_short_model_names():
    assert ui.short_model("poolside/laguna-s-2.1:free") == "laguna-s-2.1"
    assert ui.short_model("vendor/big:free") == "big"
    assert ui.short_model("solo:free") == "solo"
    assert ui.short_model("just/model") == "model"
    assert ui.short_model(None) == "n/a"
    assert ui.short_model("") == "n/a"


def test_score_bar_is_twenty_characters():
    console = make_console()
    assert ui.score_bar(console, None) is None
    assert ui.score_bar(console, "bad") is None
    full = ui.score_bar(console, 100)
    empty = ui.score_bar(console, 0)
    partial = ui.score_bar(console, 78)
    assert len(full) == 20 and full == "█" * 20
    assert len(empty) == 20 and empty == "░" * 20
    assert len(partial) == 20
    assert partial.count("█") == 16
    ascii_bar = ui.score_bar(make_console(stream=Cp1252Stream()), 50)
    assert ascii_bar == "#" * 10 + "." * 10


def test_routing_label_rules():
    assert router_log.routing_label(None) == "n/a"
    assert router_log.routing_label({"applied": 1, "reason_codes": '["A"]'}) == "SWITCHED"
    assert router_log.routing_label({"applied": 0, "reason_codes": '["A"]'}) == "STAY"
    assert router_log.routing_label(
        {"applied": 1, "reason_codes": '["HELD_MODEL"]'}
    ) == "HELD"
    assert router_log.routing_label(
        {"applied": 0, "reason_codes": '["HELD_MODEL", "A"]'}
    ) == "HELD"
    assert router_log.routing_label({}) == "STAY"


def test_analyzing_tree_uses_only_real_data(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    signals = {
        "classifier_tier": "MID",
        "classifier_score": 42,
        "confidence": 0.91,
        "intent": "chat",
        "context": "private user text",
    }
    write_db(
        path,
        [("d1", time.time(), "sess", 3, "req", "req", "mid", "active",
          json.dumps(["LOW_COST"]), json.dumps(signals), "keep", 0, None)],
        [usage_row()],
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    out = call(ui.print_analyzing, decision)
    assert "complexity: MID" in out
    assert "score: 42" in out
    assert "reason codes: LOW_COST" in out
    assert "decision: MID tier model" in out
    for marker in BANNED_MARKERS:
        assert marker not in out
    assert "private user text" not in out


def test_analyzing_tree_without_decision_is_all_na():
    out = call(ui.print_analyzing, None)
    assert "complexity: n/a" in out
    assert "score: n/a" in out
    assert "reason codes: n/a" in out
    assert "decision: n/a" in out


def test_routing_panel_switched_with_bar_boxes_and_mismatch():
    decision = {
        "chosen_model": "vendor/small:free",
        "reason_codes": json.dumps(["STRONG_KW", "HISTORY_TRIGGER"]),
        "signal_values": json.dumps({"classifier_tier": "HIGH", "classifier_score": 78}),
        "applied": 1,
    }
    out = routing_output(decision, used="vendor/big:free")
    assert re.search(r"█{16}░{4} 78", out)
    assert "ROUTE STRONG_KW HISTORY_TRIGGER" in out
    assert "laguna-s-2.1" in out
    assert "big" in out
    assert "->" in out
    assert "MODEL SWITCHED" in out
    assert "log check: MISMATCH" in out
    assert out.splitlines()[0].startswith("╭─ ROUTING DECISION")


def test_routing_panel_stay_without_mismatch():
    decision = {
        "chosen_model": "poolside/laguna-s-2.1:free",
        "reason_codes": json.dumps(["LOW_COST"]),
        "signal_values": json.dumps({"classifier_score": 12}),
        "applied": 0,
    }
    out = routing_output(decision, used="poolside/laguna-s-2.1:free")
    assert "MODEL STAY" in out
    assert "MODEL SWITCHED" not in out
    assert "MISMATCH" not in out
    assert "░" in out


def test_routing_panel_held_model():
    decision = {
        "chosen_model": "poolside/laguna-s-2.1:free",
        "reason_codes": json.dumps(["HELD_MODEL"]),
        "signal_values": json.dumps({"classifier_score": 90}),
        "applied": 1,
    }
    out = routing_output(decision)
    assert "MODEL HELD" in out
    assert "MODEL SWITCHED" not in out


def test_routing_panel_without_row_prints_na():
    out = routing_output(None, used="vendor/big:free")
    assert "n/a" in out
    assert "laguna-s-2.1" in out
    assert "big" in out
    assert "MISMATCH" not in out
    assert "Traceback" not in out


def test_execution_panel_cost_rules():
    free = {"cost_usd": 0.0, "baseline_cost_usd": 0.0}
    out = call(ui.print_execution, 214.0, 530.0, 1.2345, free)
    assert "input tokens : 214" in out
    assert "output tokens: 530" in out
    assert "latency      : 1.234 s" in out
    assert "model cost   : FREE" in out
    assert "baseline cost: FREE" in out
    assert "savings      : 0.0%" in out

    unknown = {"cost_usd": None, "baseline_cost_usd": None}
    out = call(ui.print_execution, 214.0, 530.0, None, unknown)
    assert "model cost   : UNKNOWN" in out
    assert "baseline cost: UNKNOWN" in out
    assert "savings      : n/a" in out
    assert "latency      : n/a" in out

    out = call(ui.print_execution, None, None, None, None)
    assert "input tokens : n/a" in out
    assert "model cost   : n/a" in out
    assert "baseline cost: n/a" in out
    assert "savings      : n/a" in out


def test_session_block_fields(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(
        path,
        [decision_row(reason_codes=("STRONG_KW",), applied=1)],
        [usage_row(cost=0.25, baseline=0.5)],
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("poolside/laguna-s-2.1:free", decision, usage)
    session.record("poolside/laguna-s-2.1:free", decision, usage)
    out = call(ui.print_session, session)
    assert "turns   : 2" in out
    assert "switches: 2" in out
    assert "routed  : $0.500000" in out
    assert "baseline: $1.000000" in out
    assert "saved   : $0.500000 (50.0%)" in out
    assert out.splitlines()[0].startswith("─")


def test_session_block_without_any_data():
    session = agent.Session()
    out = call(ui.print_session, session)
    assert "turns   : 0" in out
    assert "routed  : n/a" in out
    assert "baseline: n/a" in out
    assert "saved   : n/a" in out


def test_turn_output_order(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()], [usage_row(cost=0.0, baseline=0.0)])
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    out = full_turn("what is routing", "it picks a model", decision, usage)
    marks = ["REQUEST", "ANALYZING", "ROUTING DECISION", "EXECUTION", "Agent", "SESSION"]
    positions = [out.index(mark) for mark in marks]
    assert positions == sorted(positions)
    assert "what is routing" in out
    assert "it picks a model" in out
    assert "Traceback" not in out


def test_turn_without_router_still_shows_model_and_latency(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("poolside/laguna-s-2.1:free", decision, usage)
    out = full_turn("a prompt", "a reply", decision, usage, session=session, latency=0.42)
    assert "complexity: n/a" in out
    assert "laguna-s-2.1" in out
    assert "latency      : 0.420 s" in out
    assert "model cost   : n/a" in out
    assert "input tokens : 214" in out
    assert "Traceback" not in out


def test_panels_respect_terminal_width(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    for width in (60, 80, 120):
        out = full_turn("a prompt for width", "a reply", decision, usage, width=width)
        for line in out.splitlines():
            assert len(line) <= width


def test_why_prints_no_confidence_intent_or_context(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    signals = {
        "classifier_tier": "HIGH",
        "classifier_score": 100,
        "confidence": 0.91,
        "intent": "secret intent",
        "context": "private note",
    }
    write_db(
        path,
        [("d1", time.time(), "sess", 3, "req", "req", "high", "active",
          json.dumps(["STRONG_KW", "HISTORY_TRIGGER"]), json.dumps(signals),
          "rewrite", 1, None)],
        [usage_row()],
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("req", decision, usage)
    out = call(ui.print_why, session)
    assert "tier   : HIGH" in out
    assert "score  : 100" in out
    assert "reasons: STRONG_KW HISTORY_TRIGGER" in out
    assert "signals" not in out
    for marker in BANNED_MARKERS:
        assert marker not in out
    assert "private note" not in out
    assert "req" not in out


def test_why_without_decision_prints_na():
    out = call(ui.print_why, agent.Session())
    assert "tier   : n/a" in out
    assert "score  : n/a" in out
    assert "reasons: n/a" in out


def test_stats_reports_models_and_session():
    session = agent.Session()
    session.record("vendor/big:free", None, None)
    session.record("vendor/big:free", None, None)
    session.record("vendor/small:free", None, None)
    out = call(ui.print_stats, session)
    assert "turns   : 3" in out
    assert "model   : vendor/big:free  x2" in out
    assert "model   : vendor/small:free  x1" in out


def test_chat_loop_commands_and_history(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    client = fake_client([make_response("first"), make_response(""), make_response("third")])
    console = make_console()
    session = agent.Session()
    agent.chat_loop(
        console,
        client,
        session,
        "poolside/laguna-s-2.1:free",
        input_fn=echoing_input(
            console,
            [
                "hello there",
                "/stats",
                "/why",
                "/routing off",
                "still here?",
                "/routing on",
                "/new",
                "after reset",
                "/stats",
                "/exit",
            ],
        ),
    )
    out = console.file.getvalue()

    assert out.count("(empty reply)") == 1
    assert "first" in out and "third" in out
    assert "REQUEST" in out
    assert "ROUTING DECISION" in out
    assert "routing panel: off" in out and "routing panel: on" in out
    assert "model   : poolside/laguna-s-2.1:free  x3" in out
    assert "turns   : 3" in out
    assert len(client.chat.completions.calls) == 3
    assert [(m["role"], m["content"]) for m in session.history] == [
        ("user", "after reset"),
        ("assistant", "third"),
    ]
    assert session.turns == 3
    assert "Traceback" not in out


def test_chat_loop_sends_full_history(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    client = fake_client([make_response("one"), make_response("two")])
    console = make_console()
    agent.chat_loop(
        console,
        client,
        agent.Session(),
        "vendor/model:free",
        input_fn=echoing_input(console, ["first question", "second question", "/exit"]),
    )
    calls = client.chat.completions.calls
    assert calls[0]["model"] == "vendor/model:free"
    assert [m["content"] for m in calls[0]["messages"]] == ["first question"]
    assert [m["content"] for m in calls[1]["messages"]] == [
        "first question",
        "one",
        "second question",
    ]


def test_routing_off_hides_routing_panels(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    client = fake_client([make_response("quiet")])
    console = make_console()
    agent.chat_loop(
        console,
        client,
        agent.Session(),
        "m",
        input_fn=echoing_input(console, ["/routing off", "hi", "/exit"]),
    )
    out = console.file.getvalue()
    assert "ROUTING DECISION" not in out
    assert "ANALYZING" not in out
    assert "REQUEST" in out
    assert "EXECUTION" in out
    assert "input tokens : 214" in out
    assert "quiet" in out


def test_chat_loop_exits_cleanly_on_eof(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))

    def read(prompt):
        raise EOFError

    console = make_console()
    agent.chat_loop(console, fake_client([]), agent.Session(), "m", input_fn=read)
    assert "Traceback" not in console.file.getvalue()


def test_main_without_key_reports_config(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(agent, "load_dotenv", lambda *args, **kwargs: None)
    assert agent.main() == 1
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().out


def test_short_error_truncates_and_redacts():
    secret = "sk-secret-value"
    message = "failed\nwith key %s and %s" % (secret, "y" * 500)
    text = agent.short_error(RuntimeError(message), secret)
    assert text.startswith("RuntimeError: failed with key [redacted] and y")
    assert secret not in text
    assert "\n" not in text
    assert text.endswith("...")
    assert len(text) <= len("RuntimeError: ") + agent.ERROR_LIMIT + 3
    assert agent.short_error(ValueError("")) == "ValueError"


def test_error_line_is_short_and_pops_history(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    secret = "sk-secret-value"
    boom = RuntimeError("failed with key %s and %s" % (secret, "x" * 400))
    client = fake_client([boom, make_response("recovered")])
    console = make_console()
    session = agent.Session()
    agent.chat_loop(
        console,
        client,
        session,
        "m",
        secret=secret,
        input_fn=echoing_input(console, ["secret prompt text", "try again", "/exit"]),
    )
    out = console.file.getvalue()
    assert secret not in out
    assert "error: RuntimeError: failed with key [redacted] and" in out
    assert "Traceback" not in out
    assert [line for line in out.splitlines() if line.startswith("error:")]
    assert [m["content"] for m in session.history] == ["try again", "recovered"]
    assert session.turns == 1


def test_no_prompt_text_written_to_disk(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    client = fake_client([make_response("private reply")])
    console = make_console()
    before = sorted(p.name for p in ROOT.iterdir())
    agent.chat_loop(
        console,
        client,
        agent.Session(),
        "m",
        input_fn=echoing_input(console, ["private prompt", "/exit"]),
    )
    after = sorted(p.name for p in ROOT.iterdir())
    assert before == after
    for name in after:
        path = ROOT / name
        if path.is_file() and path.suffix in {".py", ".txt", ".json"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "private prompt" not in text
            assert "private reply" not in text


def test_turn_rendering_never_prints_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    decision, usage = router_log.wait_for_router(time.time() - 5, timeout=0.0)
    weird = {
        "signal_values": "not json at all",
        "reason_codes": {"unexpected": "shape"},
        "applied": object(),
    }
    out = full_turn("prompt", "reply", weird, usage, tokens_in="bad", latency=None)
    assert "Traceback" not in out
    assert "prompt" in out


def test_timestamp_key_variants():
    assert router_log.timestamp_key(None) is None
    assert router_log.timestamp_key("") is None
    assert router_log.timestamp_key("nonsense") is None
    assert router_log.timestamp_key(1700000000) == 1700000000
    assert router_log.timestamp_key("1700000000") == 1700000000
    parsed = router_log.timestamp_key("2026-01-01T00:00:00Z")
    assert parsed == pytest.approx(1767225600.0)


def test_reason_codes_tolerate_shapes():
    assert router_log.json_list('["A", "B"]') == ["A", "B"]
    assert router_log.json_list("A, B") == ["A", "B"]
    assert router_log.json_list([{"code": "A"}, {"name": "B"}, "C"]) == ["A", "B", "C"]
    assert router_log.json_list(None) == []
    assert router_log.json_object("not json") == {}
    assert router_log.json_object('{"a": 1}') == {"a": 1}
