"""The command line: one flag per config field, and no drift between them."""

from __future__ import annotations

import dataclasses

import pytest

from cli import MODES, build_parser, config_from_args, parse_layers
from config import CONFIG, RunConfig


def test_every_config_field_is_reachable_from_the_command_line():
    """The point of generating the parser: adding a knob cannot forget the flag."""
    flags = {a.dest for a in build_parser()._actions}
    for field in dataclasses.fields(RunConfig):
        assert field.name in flags, f"{field.name} has no command-line flag"


def test_no_arguments_changes_nothing():
    """`python main.py` must behave exactly as it did before the CLI existed."""
    cfg, had_args = config_from_args([])
    assert not had_args
    assert cfg == CONFIG


@pytest.mark.parametrize("mode", MODES)
def test_mode_is_positional(mode):
    cfg, had_args = config_from_args([mode])
    assert had_args and cfg.mode == mode


def test_overrides_apply_and_the_rest_is_untouched():
    cfg, _ = config_from_args(["train", "--ticker", "aapl", "--epochs", "42"])
    assert cfg.epochs == 42
    assert cfg.ticker == "aapl"
    assert cfg.alpha == CONFIG.alpha           # everything else stays put


def test_boolean_flags_go_both_ways():
    assert config_from_args(["--no-run-baselines"])[0].run_baselines is False
    assert config_from_args(["--run-baselines"])[0].run_baselines is True


def test_tuple_flags_keep_their_element_type():
    cfg, _ = config_from_args(["--price-maturities", "0.25,1.0",
                               "--sweep-architectures", "32x2,64x3"])
    assert cfg.price_maturities == (0.25, 1.0)
    assert cfg.sweep_architectures == ("32x2", "64x3")


def test_paths_arrive_as_paths():
    from pathlib import Path

    cfg, _ = config_from_args(["--chain-file", "results/spy_chain.npz",
                               "--output-dir", "out"])
    assert isinstance(cfg.chain_file, Path)
    assert isinstance(cfg.output_dir, Path)


@pytest.mark.parametrize("text,expected", [
    ("256x4", (256, 256, 256, 256)),
    ("64,64,64", (64, 64, 64)),
    (" 128 x 2 ".replace(" ", ""), (128, 128)),
    ("32", (32,)),
])
def test_architecture_shorthand(text, expected):
    assert parse_layers(text) == expected


@pytest.mark.parametrize("bad", ["", "x4", "256x", "wide,er"])
def test_bad_architecture_is_rejected(bad):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_layers(bad)


def test_help_renders():
    """A `%` in a field's docstring used to crash --help; keep it rendering."""
    text = build_parser().format_help()
    assert "--hidden" in text and "--chain-file" in text
    assert "25%" in text or "25% of its own mid" in text
