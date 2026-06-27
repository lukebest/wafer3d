"""DRAM trace coalescing via XOR match keys (Figure 5, Voxel paper)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if TYPE_CHECKING:
    from voxelsim.backends.ramulator_backend import DramRequest, RamulatorBackend

DEFAULT_QUEUE_SIZE = 32


def compute_match_keys(addrs: np.ndarray) -> np.ndarray:
    """Bit-wise XOR between consecutive addresses."""
    if len(addrs) == 0:
        return np.array([], dtype=np.uint64)
    keys = np.zeros(len(addrs), dtype=np.uint64)
    keys[0] = addrs[0]
    for i in range(1, len(addrs)):
        keys[i] = addrs[i] ^ addrs[i - 1]
    return keys


if HAS_NUMBA:

    @njit(cache=True)
    def _xor_keys_numba(addrs: np.ndarray) -> np.ndarray:
        n = len(addrs)
        keys = np.empty(n, dtype=np.uint64)
        if n == 0:
            return keys
        keys[0] = addrs[0]
        for i in range(1, n):
            keys[i] = addrs[i] ^ addrs[i - 1]
        return keys


class TraceCoalescer:
    """Cache DRAM timing results for structurally equivalent access patterns."""

    def __init__(self, tCL: int = 14, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self.queue_size = queue_size
        self._latency_cache: dict[tuple, list[int]] = {}
        self.hits = 0
        self.misses = 0

    def _keys_tuple(self, addrs: list[int]) -> tuple:
        arr = np.array(addrs, dtype=np.uint64)
        if HAS_NUMBA:
            keys = _xor_keys_numba(arr)
        else:
            keys = compute_match_keys(arr)
        return tuple(int(k) for k in keys)

    def simulate_channel(
        self,
        channel_id: int,
        requests: list["DramRequest"],
        backend: "RamulatorBackend",
    ) -> list[int]:
        if not requests:
            return []

        addrs = [r.addr for r in requests]
        key = self._keys_tuple(addrs)

        if key in self._latency_cache:
            self.hits += 1
            cached = self._latency_cache[key]
            if len(cached) == len(requests):
                return cached

        self.misses += 1
        tagged = self._tag_divergent(addrs, key)
        results_obj = backend.simulate_trace(requests, channel_id)
        latencies = [r.latency_cycles for r in results_obj]

        # Warm-up: first N in each tagged block
        self._latency_cache[key] = latencies
        return latencies

    def _tag_divergent(self, addrs: list[int], key: tuple) -> set[int]:
        """Mark divergent requests and surrounding window (N=32)."""
        tagged: set[int] = set()
        # On cache miss entire trace is tagged for simulation
        for i in range(len(addrs)):
            tagged.add(i)
            for j in range(max(0, i - self.queue_size), min(len(addrs), i + self.queue_size + 1)):
                tagged.add(j)
        return tagged

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
