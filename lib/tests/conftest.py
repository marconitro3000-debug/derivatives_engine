"""
lib/tests/conftest.py
Adds both the project root and lib/ to sys.path so the lib modules
are importable with their original names (e.g. `from exotics.barrier import`).
"""
import sys
import os

_root = os.path.join(os.path.dirname(__file__), "..", "..")
_lib  = os.path.join(os.path.dirname(__file__), "..")

for p in (_root, _lib):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
