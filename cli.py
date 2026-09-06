"""
cli.py
Every knob in `config.py`, reachable from the command line.

`main.py` still works the way it always did -- run it with no arguments and it
shows the menu. This module is the other door: it reads `RunConfig` itself and
builds one flag per field, so there is no list of arguments here to drift out of
sync with the configuration. Add a field to `RunConfig` and it becomes a flag,
with its docstring as the help text.

    python main.py                          the menu
    python main.py train --ticker AAPL      one mode, no prompts
    python main.py sweep --epochs 800       a capacity study
    python main.py --help                   every flag, generated

Type handling follows the default value: `bool` becomes `--x / --no-x`, a tuple
becomes a comma-separated list, `Path` becomes a path. `hidden_layers` also
accepts the compact `256x4` form for "four layers of 256".
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from config import CONFIG, RunConfig

MODES = ("train", "price", "compare", "sweep")


# -- value parsing ------------------------------------------------------------

def parse_layers(text: str) -> tuple[int, ...]:
    """``"256x4"`` or ``"64,64,64"`` -> ``(256, 256, 256, 256)`` / ``(64, 64, 64)``.

    The ``NxM`` form exists because the interesting axis in a capacity sweep is
    "wider or deeper", and typing ``512,512,512,512,512,512`` to ask that
    question invites a typo that silently changes the experiment.
    """
    text = text.strip().lower()
    if not text:
        raise argparse.ArgumentTypeError("empty architecture")
    if "x" in text and "," not in text:
        width, _, depth = text.partition("x")
        try:
            return (int(width),) * int(depth)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{text!r} is not WIDTHxDEPTH, e.g. 256x4") from None
    try:
        return tuple(int(p) for p in text.replace(" ", "").split(",") if p)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of widths") from None


def _tuple_parser(example):
    """Comma-separated list parser, element type taken from the current default."""
    elem = type(example[0]) if example else float

    def parse(text: str):
        text = text.strip()
        if not text:
            return ()
        return tuple(elem(p) for p in text.replace(" ", "").split(",") if p)

    return parse


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _field_docs() -> dict[str, str]:
    """The docstring written under each `RunConfig` field, read from the source.

    Python throws those away at import time -- they are documentation for the
    reader, not runtime data -- so `--help` would otherwise be a list of bare
    flag names while the actual explanations sat in `config.py`. Parsing the
    source keeps one description per knob instead of two that drift apart.
    """
    import ast
    import inspect

    docs: dict[str, str] = {}
    try:
        tree = ast.parse(inspect.getsource(sys.modules[RunConfig.__module__]))
    except (OSError, TypeError, SyntaxError):                # pragma: no cover
        return docs

    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "RunConfig"):
            continue
        pending = None
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                pending = stmt.target.id
            elif (pending and isinstance(stmt, ast.Expr)
                  and isinstance(stmt.value, ast.Constant)
                  and isinstance(stmt.value.value, str)):
                docs[pending] = " ".join(stmt.value.value.split())
                pending = None
            else:
                pending = None
    return docs


def _help_for(docs: dict[str, str], name: str, default) -> str:
    """The field's own first sentence, then its default."""
    text = docs.get(name, "")
    if text:
        first = text.split(". ")[0].rstrip(".")
        text = (first[:180] + ". ") if first else ""
    # argparse runs help strings through %-formatting; a literal "25%" in a
    # field's own docstring would otherwise blow up --help.
    return f"{text}(default: {default})".replace("%", "%%")


# -- parser -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """One flag per `RunConfig` field, generated from the dataclass."""
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="Neural implied-volatility surface. "
                    "Run with no arguments for the interactive menu.",
        epilog="Full reference, with worked examples: docs/COMMANDS.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "mode", nargs="?", choices=MODES, default=None,
        help="train: fit a chain and score it. price: query the last surface. "
             "compare: rank archived runs. sweep: fit several architectures on "
             "one chain and table the result. Omit for the menu.",
    )

    docs = _field_docs()
    for field in dataclasses.fields(RunConfig):
        if field.name == "mode":
            continue                      # the positional above owns it
        default = getattr(CONFIG, field.name)
        flag = _flag(field.name)
        help_text = _help_for(docs, field.name, default)

        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction,
                           default=None, help=help_text)
        elif field.name == "hidden_layers":
            p.add_argument(flag, "--hidden", type=parse_layers, default=None,
                           metavar="WIDTHxDEPTH",
                           help="hidden widths, e.g. 256x4 or 64,64,64 "
                                f"(default: {','.join(map(str, default))})")
        elif isinstance(default, Path):
            p.add_argument(flag, type=Path, default=None, metavar="PATH",
                           help=help_text)
        elif isinstance(default, tuple):
            p.add_argument(flag, type=_tuple_parser(default), default=None,
                           metavar="A,B,C",
                           help=_help_for(docs, field.name,
                                          ",".join(map(str, default)) or "empty"))
        elif default is None:
            p.add_argument(flag, default=None, metavar="PATH",
                           help=_help_for(docs, field.name, "none"))
        else:
            p.add_argument(flag, type=type(default), default=None,
                           help=help_text)

    return p


def config_from_args(argv: list[str] | None = None) -> tuple[RunConfig, bool]:
    """Apply command-line overrides to a copy of `CONFIG`.

    Returns ``(config, had_arguments)``. With no arguments nothing is touched and
    `main.py` behaves exactly as it did before this file existed -- the menu is
    the default interface, not a fallback.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return dataclasses.replace(CONFIG), False

    args = build_parser().parse_args(argv)
    cfg = dataclasses.replace(CONFIG)

    for name, value in vars(args).items():
        if value is None:
            continue
        setattr(cfg, name, value)

    cfg.__post_init__()                   # re-coerce the Path fields
    return cfg, True
