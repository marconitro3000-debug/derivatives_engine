"""Shared fixtures.

The synthetic chain is generated once per session: it costs a few hundred
implied-vol inversions, and every test that needs a chain needs the same one.
"""

from __future__ import annotations

import pytest

from volsurface.data import synthetic_snapshot


@pytest.fixture(scope="session")
def clean_chain():
    """A noiseless chain from a known arbitrage-free SSVI surface."""
    return synthetic_snapshot(noise_bps=0.0)


@pytest.fixture(scope="session")
def noisy_chain():
    """The same surface with 25bp of IV noise -- a stand-in for real quotes."""
    return synthetic_snapshot(noise_bps=25.0, seed=1)
