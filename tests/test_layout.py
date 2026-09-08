"""The package layout, enforced.

A structure nobody checks decays back into a flat namespace one convenient
import at a time, and the failure is silent: the code keeps working, it just
stops being findable. These tests read the import graph out of the source and
fail when the layering breaks, which is the only way the layout in the README
stays true.

The rule is one-directional dependencies:

    pricing  ->  surfaces  ->  data  ->  evaluation  ->  quote

`pricing` is primitives and depends on nothing else here. `data` needs pricing
(cleaning inverts IVs and de-Americanises) and surfaces (the synthetic
generator samples a known SSVI surface). `evaluation` needs a chain and a
surface to judge. `quote` -- pricing one named option off a fitted surface --
consumes all four, which is exactly why it sits above them rather than inside
`pricing/`.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "volsurface"

#: Later may import earlier; earlier may never import later.
LAYERS = ["pricing", "surfaces", "data", "evaluation", "quote"]
RANK = {name: i for i, name in enumerate(LAYERS)}


def _layer_of(path: pathlib.Path) -> str | None:
    """Which layer a file belongs to, or None for the package root."""
    parts = path.relative_to(PKG).parts
    if parts[0] == "quote.py":
        return "quote"
    return parts[0] if parts[0] in RANK else None


def _internal_imports(path: pathlib.Path) -> set[str]:
    """Every `volsurface.*` module this file imports, however it spells it."""
    parts = path.relative_to(PKG).parts
    here = list(parts[:-1])
    found: set[str] = set()

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[: len(here) - (node.level - 1)] if node.level > 1 else here
                found.add(".".join(base + ([node.module] if node.module else [])))
            elif node.module and node.module.startswith("volsurface"):
                found.add(node.module[len("volsurface.") :])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("volsurface."):
                    found.add(alias.name[len("volsurface.") :])
    return {m for m in found if m}


SOURCES = sorted(PKG.rglob("*.py"))


def test_the_layers_exist():
    """A renamed or deleted package must break this file, not just the README."""
    for name in ("data", "pricing", "surfaces", "evaluation"):
        assert (PKG / name / "__init__.py").is_file(), f"volsurface/{name}/ is gone"
    assert (PKG / "quote.py").is_file()
    assert (PKG / "surfaces" / "neural" / "__init__.py").is_file(), \
        "the network moved; the README says it is in surfaces/neural/"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_module_imports_a_later_layer(path):
    """The dependency arrow only points one way.

    A violation here is not a style complaint -- it is the beginning of an
    import cycle, and the last one (`price_option` filed under `pricing/`,
    reaching into `data` and `evaluation`) was found this way.
    """
    src = _layer_of(path)
    if src is None:                      # volsurface/__init__.py re-exports everything
        return

    for module in _internal_imports(path):
        dst = "quote" if module == "quote" else module.split(".")[0]
        if dst not in RANK or dst == src:
            continue
        assert RANK[dst] < RANK[src], (
            f"{path.relative_to(PKG)} imports volsurface.{module}: "
            f"'{src}' may not depend on '{dst}' "
            f"(order is {' -> '.join(LAYERS)})"
        )


def test_the_library_never_imports_the_application():
    """`config.py`, `cli.py` and `main.py` are the app; the library is reusable
    without them. An import in this direction is how a library stops being one."""
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for banned in ("import config", "from config", "import cli", "from cli",
                       "import main", "from main"):
            assert banned not in text, \
                f"{path.relative_to(PKG)} reaches back into the application ({banned})"


def test_every_public_name_is_reachable_from_the_top():
    """The layout is for reading the code, not a tax on using it: every name in
    a subpackage's `__all__` must still be `from volsurface import ...`-able."""
    import volsurface
    from volsurface import data, evaluation, pricing, surfaces

    exported = set(volsurface.__all__)
    for sub in (data, pricing, surfaces, evaluation):
        for name in sub.__all__:
            assert hasattr(volsurface, name), \
                f"{sub.__name__}.{name} is not re-exported from volsurface"
            assert name in exported, \
                f"{sub.__name__}.{name} is importable but missing from __all__"


def test_all_of___all___actually_exists():
    import volsurface

    missing = [n for n in volsurface.__all__ if not hasattr(volsurface, n)]
    assert not missing, f"__all__ names nothing: {missing}"


def test_the_network_is_where_the_docs_say_it_is():
    """The complaint that started the restructure was not being able to find
    the network. Pin the path the README advertises."""
    from volsurface.surfaces.neural.model import NeuralVolSurface

    assert NeuralVolSurface.__module__ == "volsurface.surfaces.neural.model"


def test_the_data_generator_is_where_the_docs_say_it_is():
    from volsurface.data.synthetic import synthetic_snapshot

    assert synthetic_snapshot.__module__ == "volsurface.data.synthetic"
