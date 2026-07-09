# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)

_TAG = "NIXL-PROF"


@dataclass
class _Timeline:
    kind: str
    t0: float
    ctx: str
    events: list[tuple[str, float]] = field(default_factory=list)


class NixlXferProfiler:
    def __init__(self, role: str, enabled: bool) -> None:
        self.enabled = enabled
        self.role = role
        self._lock = threading.Lock()
        self._timelines: dict[str, _Timeline] = {}

    def begin(self, key: str, kind: str, **ctx: object) -> None:
        """Start a timeline for *key*; overwrites any stale timeline."""
        if not self.enabled:
            return
        now = time.perf_counter()
        ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
        with self._lock:
            self._timelines[key] = _Timeline(kind, now, ctx_str, [("begin", now)])
        logger.info("%s %s %s START %s %s", _TAG, self.role, kind, key, ctx_str)

    def step(self, key: str, event: str) -> None:
        """Record a milestone; no-op if the timeline was never begun."""
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            tl = self._timelines.get(key)
            if tl is None:
                return
            prev = tl.events[-1][1]
            t0 = tl.t0
            kind = tl.kind
            tl.events.append((event, now))
        logger.info(
            "%s %s %s %s %s +%.1fms (d%.1fms)",
            _TAG,
            self.role,
            kind,
            key,
            event,
            (now - t0) * 1000.0,
            (now - prev) * 1000.0,
        )

    def end(self, key: str, event: str = "done") -> None:
        """Close a timeline and log the full per-stage breakdown."""
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            tl = self._timelines.pop(key, None)
            if tl is None:
                return
            tl.events.append((event, now))
            breakdown = " ".join(
                f"{name}=+{(t - tl.t0) * 1000.0:.1f}" for name, t in tl.events
            )
            kind = tl.kind
            ctx = tl.ctx
            total_ms = (now - tl.t0) * 1000.0
        logger.info(
            "%s %s %s DONE %s %s total=%.1fms | %s",
            _TAG,
            self.role,
            kind,
            key,
            ctx,
            total_ms,
            breakdown,
        )
