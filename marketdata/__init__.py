"""
marketdata/
Turns a raw broker option chain into the clean (k, T, IV) point cloud the
surface models are fitted to.
"""

from .chain import ChainSnapshot, fetch_chain, build_snapshot, synthetic_snapshot

__all__ = ["ChainSnapshot", "fetch_chain", "build_snapshot", "synthetic_snapshot"]
