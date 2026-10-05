import io
import json
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

import agent

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 110


def render(renderable, width=WIDTH):
    console = Console(
        file=io.StringIO(),
        width=width,
        no_color=True,
        legacy_windows=False,
        highlight=False,
    )
    console.print(renderable)
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
        console.print(Text("%s %s" % (prompt.rstrip(), value)))
        return value

    return read


@pytest.fixture
def no_router_wait(monkeypatch):
    monkeypatch.setattr(agent, "ROUTER_WAIT_SECONDS", 0.0)


def test_env_defaults(monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    assert agent.base_url() == "https://openrouter.ai/api/v1"
    assert agent.model_name() == "poolside/laguna-s-2.1:free"
    monkeypatch.setenv("AGENT_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("AGENT_MODEL", "vendor/cheap:free")
    assert agent.base_url() == "http://127.0.0.1:1234/v1"
    assert agent.model_name() == "vendor/cheap:free"


def test_banner_layout_and_single_width(monkeypatch):
    monkeypatch.setattr(agent, "probe_router", lambda url, timeout=0.3: "offline")
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
    agent.print_banner(console, "http://127.0.0.1:1", "n/a")
    out = console.file.getvalue()

    assert "T A M I A S" in out
    assert "MODEL ROUTING TERMINAL" in out
    assert "v0.1 | router: offline | mode: n/a" in out

    box_lines = [line for line in out.splitlines() if line.startswith(agent.BLOCK)]
    assert box_lines
    assert len({len(line) for line in box_lines}) == 1
    for line in out.splitlines():
        for char in line:
            assert unicodedata.east_asian_width(char) not in ("W", "F")
            assert not 0x1F300 <= ord(char) <= 0x1FAFF


def test_banner_mode_active_and_online(monkeypatch):
    monkeypatch.setattr(agent, "probe_router", lambda url, timeout=0.3: "online")
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
    agent.print_banner(console, "https://openrouter.ai/api/v1", "active")
    out = console.file.getvalue()
    assert "v0.1 | router: online | mode: active" in out
    assert "MODEL ROUTING TERMINAL" in out


def test_probe_router_offline_on_bad_host():
    assert agent.probe_router("http://127.0.0.1:1", timeout=0.3) == "offline"
    assert agent.probe_router("not-a-url") == "offline"
    assert agent.probe_router("") == "offline"


def test_probe_router_online_uses_host_and_port(monkeypatch):
    seen = {}

    @contextmanager
    def fake_connect(address, timeout=None):
        seen["address"] = address
        seen["timeout"] = timeout
        yield None

    monkeypatch.setattr(agent.socket, "create_connection", fake_connect)
    assert agent.probe_router("http://example.invalid:8123/v1") == "online"
    assert seen == {"address": ("example.invalid", 8123), "timeout": 0.3}
    assert agent.probe_router("https://example.invalid/v1") == "online"
    assert seen["address"] == ("example.invalid", 443)
    assert agent.probe_router("http://example.invalid/v1") == "online"
    assert seen["address"] == ("example.invalid", 80)


def test_db_connection_is_read_only(tmp_path):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()], [usage_row()])
    conn = agent.connect_ro(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM router_decisions").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO router_decisions VALUES (%s)" % ",".join("?" * 13), decision_row("d2")
            )
    finally:
        conn.close()


def test_router_db_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "custom.sqlite3"))
    assert agent.router_db_path() == tmp_path / "custom.sqlite3"
    monkeypatch.delenv("ROUTER_DB")
    assert agent.router_db_path() == Path.home() / ".tamias" / "router.sqlite3"


def test_wait_polls_until_usage_row_appears(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    conn = sqlite3.connect(str(path))

    def late_usage(interval):
        conn.execute("INSERT INTO router_usage VALUES (?,?,?,?,?,?,?,?,?,?)", usage_row(cost=0.25))
        conn.commit()

    decision, usage = agent.wait_for_router(
        time.time() - 5, timeout=5.0, interval=0.01, sleep=late_usage
    )
    conn.close()
    assert decision["chosen_model"] == "poolside/laguna-s-2.1:free"
    assert usage["cost_usd"] == 0.25


def test_wait_stops_at_timeout_without_usage(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    assert decision is not None
    assert usage is None


def test_ignores_rows_older_than_request(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row(timestamp=time.time() - 600)], [usage_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    assert agent.wait_for_router(time.time(), timeout=0.0) == (None, None)


def test_missing_db_gives_no_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "nope.sqlite3"))
    assert agent.lookup_router(time.time() - 5) == (None, None)
    assert agent.router_mode() == "n/a"


def test_broken_db_file_gives_no_data(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    path.write_text("this is not a database")
    monkeypatch.setenv("ROUTER_DB", str(path))
    assert agent.lookup_router(time.time() - 5) == (None, None)
    assert agent.router_mode() == "n/a"


def test_mode_read_from_log(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row(mode="active")], [usage_row()])
    monkeypatch.setenv("ROUTER_DB", str(path))
    assert agent.router_mode() == "active"
    other = tmp_path / "other.sqlite3"
    write_db(other, [decision_row(mode="weird")], [usage_row()])
    monkeypatch.setenv("ROUTER_DB", str(other))
    assert agent.router_mode() == "n/a"


def panel_text(session, used, decision, usage, tokens_in=214, tokens_out=530,
               requested="poolside/laguna-s-2.1:free", width=WIDTH):
    return render(
        agent.routing_panel(session, requested, used, decision, usage, tokens_in, tokens_out),
        width=width,
    )


def assert_ascii_panel(out):
    lines = out.splitlines()
    assert lines[0].startswith("+") and " ROUTING " in lines[0]
    assert lines[-1].startswith("+") and lines[-1].endswith("+")
    for line in lines:
        if line.startswith("+"):
            assert line.endswith("+")
        else:
            assert line.startswith("|") and line.endswith("|")
    for banned in ("\u2502", "\u2500", "\u250c", "\u2510", "\u2514", "\u2518", "\u2588"):
        assert banned not in out


def test_panel_stay_free_and_log_ok(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(
        path,
        [
            decision_row(
                chosen_model="poolside/laguna-s-2.1:free", reason_codes=("LOW_COST",), applied=0
            )
        ],
        [usage_row(cost=0.0, baseline=0.0)],
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("poolside/laguna-s-2.1:free", decision, usage, 214, 530)
    out = panel_text(session, "poolside/laguna-s-2.1:free", decision, usage)

    assert_ascii_panel(out)
    assert "task    : HIGH (score 100)  LOW_COST" in out
    assert "model   : requested poolside/laguna-s-2.1:free / used poolside/laguna-s-2.1:free  STAY" in out
    assert "log check: OK" in out
    assert "tokens  : in 214  out 530" in out
    assert "cost    : FREE      baseline: FREE" in out
    assert "session : 1 turn | in 214 out 530 | routed FREE | baseline FREE | 0 switches" in out


def test_panel_switched_and_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(
        path,
        [decision_row(chosen_model="vendor/small:free", reason_codes=("STRONG_KW", "STRONG_KW"))],
        [usage_row(cost=0.0123456, baseline=0.5)],
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("vendor/big:free", decision, usage, 214, 530)
    out = panel_text(session, "vendor/big:free", decision, usage)

    assert "model   : requested poolside/laguna-s-2.1:free / used vendor/big:free  SWITCHED" in out
    assert "log check: MISMATCH" in out
    assert "$0.012346" in out
    assert "$0.500000" in out
    assert "1 switch" in session.session_text()


def test_panel_held_label(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(
        path,
        [decision_row(reason_codes=("HELD_MODEL",), applied=1)],
        [usage_row()],
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    out = panel_text(agent.Session(), "poolside/laguna-s-2.1:free", decision, usage)
    assert "HELD" in out
    assert "SWITCHED" not in out


def test_panel_unknown_cost_is_not_zero(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()], [usage_row(cost=None, baseline=None)])
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("poolside/laguna-s-2.1:free", decision, usage, 214, 530)
    out = panel_text(session, "poolside/laguna-s-2.1:free", decision, usage)
    assert "cost    : UNKNOWN   baseline: UNKNOWN" in out
    assert "routed UNKNOWN (known part)" in session.session_text()
    assert "log check: OK" in out


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
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    assert decision is not None
    session = agent.Session()
    out = panel_text(session, "poolside/laguna-s-2.1:free", decision, usage, tokens_in=None, tokens_out=None)
    assert "task    : n/a" in out
    assert "n/a" in out
    assert "used poolside/laguna-s-2.1:free" in out


def test_no_router_still_shows_model_and_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    assert decision is None and usage is None
    session = agent.Session()
    session.record("poolside/laguna-s-2.1:free", decision, usage, 214, 530)
    out = panel_text(session, "poolside/laguna-s-2.1:free", decision, usage)
    assert_ascii_panel(out)
    assert "task    : n/a" in out
    assert "used poolside/laguna-s-2.1:free  n/a" in out
    assert "log check: n/a" in out
    assert "tokens  : in 214  out 530" in out
    assert "cost    : n/a       baseline: n/a" in out
    assert "routed n/a | baseline n/a" in out
    assert "Traceback" not in out


def test_usage_tokens_used_when_response_missing(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    write_db(path, [decision_row()], [usage_row(tokens_in=11, tokens_out=22)])
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("m", decision, usage, None, None)
    assert "in 11 out 22" in session.session_text()


def test_cost_sum_never_counts_unknown_as_zero():
    total = agent.CostSum()
    assert total.text() == "n/a"
    total.add(None)
    assert total.text() == "UNKNOWN (known part)"
    assert total.known == 0.0
    total.add(0.0)
    assert total.text() == "FREE (known part)"
    total.add(0.25)
    assert total.text() == "$0.250000 (known part)"
    clean = agent.CostSum()
    clean.add(0.0)
    assert clean.text() == "FREE"


def test_format_cost_rules():
    assert agent.format_cost(0.0) == "FREE"
    assert agent.format_cost(None) == "UNKNOWN"
    assert agent.format_cost("") == "UNKNOWN"
    assert agent.format_cost(0.25) == "$0.250000"


def test_why_has_no_text_fields(tmp_path, monkeypatch):
    path = tmp_path / "router.sqlite3"
    signals = {"classifier_tier": "HIGH", "classifier_score": 100, "api_format": 1,
               "confidence": 0.91, "note": "user asked something private"}
    write_db(
        path,
        [("d1", time.time(), "sess", 3, "req", "req", "high", "active",
          json.dumps(["STRONG_KW", "HISTORY_TRIGGER"]),
          json.dumps(signals), "rewrite", 1, None)],
        [usage_row()],
    )
    monkeypatch.setenv("ROUTER_DB", str(path))
    decision, usage = agent.wait_for_router(time.time() - 5, timeout=0.0)
    session = agent.Session()
    session.record("req", decision, usage, 1, 2)
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
    agent.print_why(console, session)
    out = console.file.getvalue()
    assert "tier    : HIGH" in out
    assert "score   : 100" in out
    assert "reasons : STRONG_KW HISTORY_TRIGGER" in out
    assert "signals : api_format=1 confidence=0.91" in out
    assert "private" not in out
    assert "req" not in out


def test_why_without_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
    agent.print_why(console, agent.Session())
    out = console.file.getvalue()
    assert "tier    : n/a" in out
    assert "score   : n/a" in out
    assert "reasons : n/a" in out
    assert "signals : n/a" in out


def test_chat_loop_commands_and_history(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    client = fake_client([make_response("first"), make_response(""), make_response("third")])
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
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
    assert "ROUTING" in out
    assert "routing panel: off" in out and "routing panel: on" in out
    assert "model   : poolside/laguna-s-2.1:free  x3" in out
    assert "session : 3 turns | in 642 out 1,590 | routed n/a | baseline n/a | 0 switches" in out
    assert len(client.chat.completions.calls) == 3
    assert [(m["role"], m["content"]) for m in session.history] == [
        ("user", "after reset"),
        ("assistant", "third"),
    ]
    assert session.turns == 3


def test_chat_loop_sends_full_history(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    client = fake_client([make_response("one"), make_response("two")])
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
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


def test_routing_off_hides_panel(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))
    client = fake_client([make_response("quiet")])
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
    agent.chat_loop(
        console,
        client,
        agent.Session(),
        "m",
        input_fn=echoing_input(console, ["/routing off", "hi", "/exit"]),
    )
    out = console.file.getvalue()
    assert "ROUTING" not in out
    assert "tokens  : in 214  out 530" in out


def test_chat_loop_exits_cleanly_on_eof(no_router_wait, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB", str(tmp_path / "missing.sqlite3"))

    def read(prompt):
        raise EOFError

    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
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
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
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
    console = Console(file=io.StringIO(), width=WIDTH, no_color=True, legacy_windows=False)
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


def test_panel_respects_terminal_width():
    session = agent.Session()
    for width in (60, 80, 120):
        out = panel_text(session, "poolside/laguna-s-2.1:free", None, None, width=width)
        for line in out.splitlines():
            assert len(line) <= width


def test_timestamp_key_variants():
    assert agent.timestamp_key(None) is None
    assert agent.timestamp_key("") is None
    assert agent.timestamp_key("nonsense") is None
    assert agent.timestamp_key(1700000000) == 1700000000
    assert agent.timestamp_key("1700000000") == 1700000000
    parsed = agent.timestamp_key("2026-01-01T00:00:00Z")
    assert parsed == pytest.approx(1767225600.0)


def test_reason_codes_tolerates_shapes():
    assert agent.json_list('["A", "B"]') == ["A", "B"]
    assert agent.json_list("A, B") == ["A", "B"]
    assert agent.json_list([{"code": "A"}, {"name": "B"}, "C"]) == ["A", "B", "C"]
    assert agent.json_list(None) == []
    assert agent.json_object("not json") == {}
    assert agent.json_object('{"a": 1}') == {"a": 1}
