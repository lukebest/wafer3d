"""NoC simulation: Tier-A analytic model + Tier-B BookSim with pattern cache."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from voxelsim.chip.config import ChipConfig, NoCBackend, NoCTopology
from voxelsim.chip.topology import ChipTopology


@dataclass
class NoCTransfer:
    src_core: int
    dst_core: int
    byte_size: int
    inject_cycle: int = 0
    event_id: int = 0


@dataclass
class NoCLatencyResult:
    transfer: NoCTransfer
    latency_cycles: int
    noc_overhead_cycles: int
    backend: str


class NoCBackendProtocol(Protocol):
    def estimate_batch(self, transfers: list[NoCTransfer]) -> list[NoCLatencyResult]:
        ...


@dataclass
class AnalyticNoCModel:
    """Tier-A: hop count + link bandwidth sharing + contention."""

    config: ChipConfig
    topology: ChipTopology = field(init=False)

    def __post_init__(self) -> None:
        self.topology = ChipTopology(self.config)
        self.link_bw = self.config.noc.link_bandwidth_bytes_per_cycle
        # NoC bandwidth strictly lower than SRAM read bandwidth (paper §5.4)
        sram_read_bw = self.config.per_core_sram_bytes  # bytes/cycle upper bound proxy
        self.effective_link_bw = min(self.link_bw, sram_read_bw // 4)

    def _single_latency(self, t: NoCTransfer) -> int:
        if t.src_core == t.dst_core:
            return 1
        hops = self.topology.hops(t.src_core, t.dst_core)
        if self.config.noc_topology == NoCTopology.ALL_TO_ALL:
            hops = 1
        flit_bytes = max(1, self.effective_link_bw)
        transfer_cycles = max(1, (t.byte_size + flit_bytes - 1) // flit_bytes)
        return hops * transfer_cycles + hops  # +1 cycle per hop pipeline

    def estimate_batch(self, transfers: list[NoCTransfer]) -> list[NoCLatencyResult]:
        if not transfers:
            return []

        # Link contention: count overlapping transfers per link (not byte volume)
        link_count: dict[tuple[int, int], int] = {}
        for t in transfers:
            for a, b in self.topology.route_links(t.src_core, t.dst_core):
                key = (min(a, b), max(a, b))
                link_count[key] = link_count.get(key, 0) + 1

        max_contention = max(link_count.values()) if link_count else 1
        contention_factor = min(8.0, max(1.0, float(max_contention)))

        results: list[NoCLatencyResult] = []
        for t in transfers:
            base = self._single_latency(t)
            lat = int(base * contention_factor)
            overhead = max(0, lat - base)
            results.append(
                NoCLatencyResult(
                    transfer=t,
                    latency_cycles=lat,
                    noc_overhead_cycles=overhead,
                    backend="analytic",
                )
            )
        return results


@dataclass
class PatternCache:
    """Cache BookSim/analytic results by structured traffic pattern key."""

    _store: dict[str, int] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def pattern_key(
        self,
        transfers: list[NoCTransfer],
        topology: str,
        routing: str,
    ) -> str:
        parts: list[str] = [topology, routing]
        base_src = transfers[0].src_core if transfers else 0
        base_dst = transfers[0].dst_core if transfers else 0
        for t in sorted(transfers, key=lambda x: (x.inject_cycle, x.event_id)):
            ds = t.dst_core - base_dst
            ss = t.src_core - base_src
            rel = t.inject_cycle - (transfers[0].inject_cycle if transfers else 0)
            vol_bucket = (t.byte_size // 128) * 128
            parts.append(f"{ss}:{ds}:{vol_bucket}:{rel}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> int | None:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, latency: int) -> None:
        self._store[key] = latency


class NoCSimulator:
    """Unified NoC interface with Tier-A/Tier-B backends and pattern caching."""

    def __init__(self, config: ChipConfig) -> None:
        self.config = config
        self.topology = ChipTopology(config)
        self.analytic = AnalyticNoCModel(config)
        self.cache = PatternCache()
        self._booksim = None
        if config.noc.backend == NoCBackend.BOOKSIM:
            self._init_booksim()

    def _init_booksim(self) -> None:
        try:
            from voxelsim.backends.booksim_backend import BookSimBackend

            self._booksim = BookSimBackend(self.config)
        except Exception:
            self._booksim = None

    def estimate_transfers(self, transfers: list[NoCTransfer]) -> list[NoCLatencyResult]:
        if not transfers:
            return []

        key = self.cache.pattern_key(
            transfers,
            self.config.noc_topology.value,
            self.config.noc.routing,
        )
        cached = self.cache.get(key)
        if cached is not None:
            per = max(1, cached // len(transfers))
            return [
                NoCLatencyResult(
                    transfer=t,
                    latency_cycles=per,
                    noc_overhead_cycles=0,
                    backend="cached",
                )
                for t in transfers
            ]

        if self.config.noc.backend == NoCBackend.BOOKSIM and self._booksim is not None:
            results = self._booksim.estimate_batch(transfers)
            total = sum(r.latency_cycles for r in results)
            self.cache.put(key, total)
            return results

        results = self.analytic.estimate_batch(transfers)
        total = sum(r.latency_cycles for r in results)
        self.cache.put(key, total)
        return results
