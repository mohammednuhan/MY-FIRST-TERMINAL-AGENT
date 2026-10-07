from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.tree import Tree

import router_log

APP_VERSION = "v0.1.0"
APP_SUBTITLE = "MODEL ROUTING TERMINAL"
PROMPT = "[bold green]tamias> [/bold green] "
MIN_ART_WIDTH = 60
BAR_WIDTH = 20
BLOCK_ART = "\n".join(
    (
        "████████╗ █████╗ ███╗   ███╗██╗ █████╗ ███████╗",
        "╚══██╔══╝██╔══██╗████╗ ████║██║██╔══██╗██╔════╝",
        "   ██║   ███████║██╔████╔██║██║███████║███████╗",
        "   ██║   ██╔══██║██║╚██╔╝██║██║██╔══██║╚════██║",
        "   ██║   ██║  ██║██║ ╚═╝ ██║██║██║  ██║███████║",
        "   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚══════╝",
    )
)
ART_TO_ASCII = str.maketrans(
    {
        "\u2588": "#",
        "\u2550": "-",
        "\u2551": "|",
        "\u2554": "+",
        "\u2557": "+",
        "\u255a": "+",
        "\u255d": "+",
        "\u25cf": "*",
        "\u2591": ".",
    }
)
TIER_STYLES = {"LOW": "green", "MID": "yellow", "HIGH": "red"}
SWITCH_STYLES = {"SWITCHED": "cyan", "HELD": "yellow", "STAY": "dim", "n/a": "dim"}
SWITCH_WORDS = frozenset(("SWITCHED", "HELD", "STAY"))
MODE_STYLES = {"SHADOW": "cyan", "ACTIVE": "green"}
MODE_WORDS = frozenset(MODE_STYLES)


def ascii_only(console):
    try:
        encoding = (console.encoding or "utf-8").lower()
    except Exception:
        return False
    return not encoding.startswith("utf")


def fit(console, value):
    text = value if isinstance(value, str) else str(value)
    try:
        encoding = console.encoding or "utf-8"
    except Exception:
        encoding = "utf-8"
    try:
        text.encode(encoding)
        return text
    except Exception:
        pass
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii")


def styled(console, value, style=None):
    return Text(fit(console, value), style=style)


def label_line(console, label, value, style=None, width=None):
    head = str(label) if width is None else str(label).ljust(width)
    line = Text()
    line.append(head + ": ", style="dim")
    line.append(fit(console, value), style=style)
    return line


def rule_line(console):
    return Rule(style="dim", characters="-" if ascii_only(console) else "\u2500")


def display_number(value):
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number.is_integer():
        return str(int(number))
    return ("%.4f" % number).rstrip("0").rstrip(".")


def count_text(value):
    if value is None:
        return "n/a"
    try:
        return "{:,.0f}".format(float(value))
    except (TypeError, ValueError):
        return "n/a"


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


def latency_text(value):
    if value is None:
        return "n/a"
    try:
        return "%.3f s" % float(value)
    except (TypeError, ValueError):
        return "n/a"


def savings_percent(cost, baseline):
    if cost is None or baseline is None:
        return "n/a"
    try:
        routed = float(cost)
        base = float(baseline)
    except (TypeError, ValueError):
        return "n/a"
    if base > 0:
        return "%.1f%%" % ((base - routed) / base * 100.0)
    if routed == 0.0 and base == 0.0:
        return "0.0%"
    return "n/a"


def cost_sum_text(total):
    known = int(getattr(total, "known", 0) or 0)
    unknown = int(getattr(total, "unknown", 0) or 0)
    if not known and not unknown:
        return "n/a"
    if not known:
        return "UNKNOWN"
    text = format_cost(getattr(total, "total", None))
    if unknown:
        return "%s (known part)" % text
    return text


def saved_text(session):
    pairs = int(getattr(session, "pairs", 0) or 0)
    if not pairs:
        return "n/a"
    cost = float(getattr(session, "pair_cost", 0.0) or 0.0)
    baseline = float(getattr(session, "pair_baseline", 0.0) or 0.0)
    return "%s (%s)" % (format_cost(baseline - cost), savings_percent(cost, baseline))


def usage_text(usage, key):
    if usage is None or key not in usage:
        return "n/a"
    return format_cost(usage.get(key))


def savings_of(usage):
    if usage is None or "cost_usd" not in usage or "baseline_cost_usd" not in usage:
        return "n/a"
    return savings_percent(usage.get("cost_usd"), usage.get("baseline_cost_usd"))


def short_model(name):
    if name is None:
        return "n/a"
    text = str(name).strip()
    if not text:
        return "n/a"
    if "/" in text:
        text = text.split("/", 1)[1]
    if text.endswith(":free"):
        text = text[: -len(":free")]
    return text or "n/a"


def tier_word(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def tier_style(tier):
    return TIER_STYLES.get(tier)


def decision_word(tier):
    if tier in TIER_STYLES:
        return "%s tier model" % tier
    return "n/a"


def score_bar(console, score):
    if score is None:
        return None
    try:
        number = float(score)
    except (TypeError, ValueError):
        return None
    filled = int(number / 100.0 * BAR_WIDTH + 0.5)
    filled = max(0, min(BAR_WIDTH, filled))
    if ascii_only(console):
        return "#" * filled + "." * (BAR_WIDTH - filled)
    return "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)


def print_banner(console, online=False, mode="n/a"):
    ascii_mode = ascii_only(console)
    parts = []
    if console.width >= MIN_ART_WIDTH:
        art = BLOCK_ART if not ascii_mode else BLOCK_ART.translate(ART_TO_ASCII)
        parts.append(Align.center(Text(art, style="bold cyan")))
    parts.append(
        Columns(
            [
                Align.left(Text(APP_SUBTITLE, style="bold")),
                Align.right(Text(APP_VERSION, style="dim")),
            ],
            padding=0,
            expand=True,
            equal=True,
        )
    )
    parts.append(rule_line(console))
    parts.append(status_line(console, online, mode))
    console.print(Panel(Group(*parts), box=box.ROUNDED, padding=(0, 1)))


def status_line(console, online, mode):
    dot = "*" if ascii_only(console) else "\u25cf"
    state = "ONLINE" if online else "OFFLINE"
    state_style = "green" if online else "red"
    word = str(mode or "").strip().upper()
    if word not in MODE_WORDS:
        word = "N/A"
    mode_style = MODE_STYLES.get(word, "dim")
    line = Text()
    line.append(dot + " ", style=state_style)
    line.append("ROUTER ", style="dim")
    line.append(state, style=state_style)
    line.append("   ")
    line.append(dot + " ", style=mode_style)
    line.append("MODE ", style="dim")
    line.append(word, style=mode_style)
    return line


def print_request(console, prompt):
    body = styled(console, str(prompt).rstrip("\r\n"))
    console.print(
        Panel(body, title="REQUEST", title_align="left", box=box.ROUNDED, padding=(0, 1))
    )


def print_analyzing(console, decision):
    signals = router_log.signals_of(decision)
    codes = router_log.reason_codes(decision)
    tier = tier_word(signals.get("classifier_tier"))
    rows = (
        ("complexity", tier or "n/a", tier_style(tier) or "dim"),
        ("score", display_number(signals.get("classifier_score")), None),
        ("reason codes", " ".join(codes) if codes else "n/a", None),
        ("decision", decision_word(tier), tier_style(tier) or "dim"),
    )
    tree = Tree(Text("ANALYZING", style="bold"), guide_style="dim")
    for label, value, style in rows:
        node = Text()
        node.append(label + ": ", style="dim")
        node.append(fit(console, value), style=style)
        tree.add(node)
    console.print(tree)


def print_routing(console, requested, used, decision):
    signals = router_log.signals_of(decision)
    codes = router_log.reason_codes(decision)
    label = router_log.routing_label(decision)
    caption = "MODEL " + label if label in SWITCH_WORDS else label
    body = Group(
        bar_line(console, signals.get("classifier_score")),
        route_line(console, codes),
        models_row(console, requested, used),
        Align.center(styled(console, caption, SWITCH_STYLES.get(label))),
    )
    console.print(
        Panel(
            body,
            title="ROUTING DECISION",
            title_align="left",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    print_log_check(console, decision, used)


def bar_line(console, score):
    line = Text()
    bar = score_bar(console, score)
    if bar is None:
        line.append("n/a", style="dim")
        return line
    line.append(bar)
    line.append(" " + display_number(score))
    return line


def route_line(console, codes):
    line = Text()
    line.append("ROUTE ", style="dim")
    line.append(fit(console, " ".join(codes)) if codes else "n/a", style=None if codes else "dim")
    return line


def model_box(console, name):
    return Panel(
        Text(fit(console, short_model(name)), justify="center"),
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def models_row(console, requested, used):
    arrow = Text("\n->\n", justify="center", style="dim")
    row = Columns(
        [model_box(console, requested), arrow, model_box(console, used)],
        padding=(0, 1),
        expand=False,
    )
    return Align.center(row)


def print_log_check(console, decision, used):
    if decision is None:
        return
    chosen = decision.get("chosen_model")
    if not chosen or not used:
        return
    if str(chosen) == str(used):
        return
    line = Text()
    line.append("log check: ", style="dim")
    line.append("MISMATCH", style="red")
    console.print(line)


def print_execution(console, tokens_in, tokens_out, latency, usage):
    rows = (
        label_line(console, "input tokens", count_text(tokens_in), width=13),
        label_line(console, "output tokens", count_text(tokens_out), width=13),
        label_line(console, "latency", latency_text(latency), width=13),
        label_line(console, "model cost", usage_text(usage, "cost_usd"), width=13),
        label_line(console, "baseline cost", usage_text(usage, "baseline_cost_usd"), width=13),
        label_line(console, "savings", savings_of(usage), width=13),
    )
    console.print(
        Panel(
            Group(*rows),
            title="EXECUTION",
            title_align="left",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def print_reply(console, reply):
    console.print(Text("Agent", style="bold cyan"))
    text = fit(console, reply)
    if ascii_only(console):
        console.print(Text(text))
    else:
        console.print(Markdown(text))


def session_panel(console, session):
    rows = (
        label_line(console, "turns", str(getattr(session, "turns", 0)), width=8),
        label_line(console, "switches", str(getattr(session, "switches", 0)), width=8),
        label_line(console, "routed", cost_sum_text(getattr(session, "routed", None)), width=8),
        label_line(console, "baseline", cost_sum_text(getattr(session, "baseline", None)), width=8),
        label_line(console, "saved", saved_text(session), width=8),
    )
    return Panel(
        Group(*rows),
        title="SESSION",
        title_align="left",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def print_session(console, session):
    console.print(rule_line(console))
    console.print(session_panel(console, session))


def print_stats(console, session):
    console.print(session_panel(console, session))
    models = getattr(session, "models", None)
    if not models:
        console.print(label_line(console, "models", "n/a", width=8))
        return
    for name, count in sorted(models.items()):
        console.print(label_line(console, "model", "%s  x%d" % (name, count), width=8))


def print_why(console, session):
    decision = getattr(session, "last_decision", None)
    signals = router_log.signals_of(decision)
    codes = router_log.reason_codes(decision)
    tier = tier_word(signals.get("classifier_tier"))
    rows = (
        label_line(console, "tier", tier or "n/a", style=tier_style(tier) or "dim", width=7),
        label_line(
            console, "score", display_number(signals.get("classifier_score")), width=7
        ),
        label_line(console, "reasons", " ".join(codes) if codes else "n/a", width=7),
    )
    for row in rows:
        console.print(row)


def render_turn(
    console,
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
    routing=True,
):
    try:
        print_request(console, prompt)
        if routing:
            print_analyzing(console, decision)
            print_routing(console, requested, used, decision)
        print_execution(console, tokens_in, tokens_out, latency, usage if routing else None)
    except Exception:
        message(console, "display unavailable")
    print_reply(console, reply)
    print_session(console, session)


def message(console, text):
    console.print(Text(fit(console, text), style="dim"))


def error(console, text, headline=None):
    line = Text()
    if headline:
        line.append(str(headline) + ": ", style="bold red")
    line.append(fit(console, text), style="red")
    console.print(line)
