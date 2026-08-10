"""
proxy_manager.py
----------------
Stateful proxy pool for high-throughput Playwright scraping.

Implements M2 (health-managed proxy rotation):
- Each proxy is a ProxySlot with a health state:
  HEALTHY / IN_USE / COOLING / QUARANTINED.
- ``rent()`` hands out the next usable proxy (round-robin, skipping any slot
  that is busy, cooling, or quarantined).
- ``release(ok=...)`` routes failures through an error taxonomy (auth /
  rate-limit / handshake / network-latency / packet-loss) and applies
  exponential backoff, quarantine, and a simple circuit breaker.
- Rotation is surfaced to the caller naturally: after a failed ``release()``,
  the next ``rent()`` returns a different proxy (or None when none is usable);
  the caller opens a fresh Playwright context, since proxies are bound at
  context level in Playwright.

Usage
-----
    mgr = ProxyManager([cfg1, cfg2, None])      # None = direct / no proxy
    proxy = await mgr.rent()
    ...
    await mgr.release(proxy, ok=False, error=exc)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from proxy_loader import ProxyConfig


class ProxyErrorKind(str, Enum):
    """Normalised failure categories used for error-taxonomy routing."""
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    PROXY_HANDSHAKE = "proxy_handshake"
    NETWORK_LATENCY = "network_latency"
    PACKET_LOSS = "packet_loss"
    OTHER = "other"


class ProxyHealth(str, Enum):
    HEALTHY = "healthy"
    IN_USE = "in_use"
    COOLING = "cooling"
    QUARANTINED = "quarantined"


# Recognised substrings, checked in priority order.  These cover Playwright
# navigation errors (`net::ERR_*`), socket errors, and HTTP status rulings.
_AUTH_SIGNS = ("407", "proxy authentication", "authentication required",
               "credentials")
_RATE_SIGNS = ("429", "403", "rate limit", "too many requests")
_HANDSHAKE_SIGNS = ("proxy_connection_failed", "tunnel_connection_failed",
                    "proxy handshake", "err_proxy", "bad gateway", "502",
                    "503", "504")
_LATENCY_SIGNS = ("timed out", "timeout", "network idle", "max retries")
_LOSS_SIGNS = ("connection reset", "reset by peer", "internet disconnected",
               "network is unreachable", "connection refused",
               "socket hang up", "no route to host", "err_connection",
               "connection closed while reading", "target page, context or browser has been closed",
               "browser has been closed", "browser.close")



def classify_proxy_error(exc: BaseException) -> ProxyErrorKind:
    """Map a Playwright/socket exception onto the proxy error taxonomy."""
    msg = " ".join(str(a) for a in getattr(exc, "args", ()))
    msg = (msg + " " + type(exc).__name__).lower().strip()
    if not msg:
        return ProxyErrorKind.OTHER
    if any(s in msg for s in _AUTH_SIGNS):
        return ProxyErrorKind.AUTH
    if any(s in msg for s in _RATE_SIGNS):
        return ProxyErrorKind.RATE_LIMIT
    if any(s in msg for s in _HANDSHAKE_SIGNS):
        return ProxyErrorKind.PROXY_HANDSHAKE
    if any(s in msg for s in _LATENCY_SIGNS):
        return ProxyErrorKind.NETWORK_LATENCY
    if any(s in msg for s in _LOSS_SIGNS):
        return ProxyErrorKind.PACKET_LOSS
    return ProxyErrorKind.OTHER

# ──── Proxy slot & manager ───────────────────────────────────────────────────


@dataclass
class ProxySlot:
    config: ProxyConfig | None
    status: ProxyHealth = ProxyHealth.HEALTHY
    fail_count: int = 0
    ok_count: int = 0
    requests: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""

    @property
    def label(self) -> str:
        return "direct" if self.config is None else self.config.label

    @property
    def is_direct(self) -> bool:
        return self.config is None


@dataclass
class RentalResult:
    """Explicit result returned by ProxyManager.rent_result()."""
    acquired: bool
    config: ProxyConfig | None
    slot: ProxySlot | None = None


class ProxyManager:
    """
    Asyncio-safe pool of proxies with health tracking, exponential backoff,
    and a circuit breaker that quarantines persistently failing proxies.
    """

    def __init__(
        self,
        configs: list[ProxyConfig | None],
        *,
        circuit_threshold: int = 3,
        quarantine_seconds: float = 300.0,
        cooldown_base: float = 10.0,
        cooldown_max: float = 120.0,
    ) -> None:
        self._slots = [ProxySlot(cfg) for cfg in configs]
        self._circuit_threshold = circuit_threshold
        self._quarantine_seconds = quarantine_seconds
        self._cooldown_base = cooldown_base
        self._cooldown_max = cooldown_max
        self._cursor = 0
        self._lock = asyncio.Lock()

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def slots(self) -> list[ProxySlot]:
        return list(self._slots)

    def state_of(self, config: ProxyConfig | None) -> str:
        slot = self._find(config)
        return "direct" if slot is None or slot.is_direct else slot.status.value

    def summary(self) -> str:
        counts = {h.value: 0 for h in ProxyHealth}
        for s in self._slots:
            counts[s.status.value] += 1
        return (
            f"proxies {len(self._slots)} | "
            f"healthy {counts['healthy']} in-use {counts['in_use']} "
            f"cooling {counts['cooling']} quarantined {counts['quarantined']}"
        )

    # ── scheduling ────────────────────────────────────────────────────────────

    async def rent_result(self) -> RentalResult:
        """
        Attempt to rent the next usable proxy slot.
        Returns a RentalResult with `acquired=True` and the `ProxyConfig` (or None for direct),
        or `acquired=False` when every slot is busy, cooling, or quarantined.
        """
        async with self._lock:
            n = len(self._slots)
            now = time.monotonic()
            for i in range(n):
                idx = (self._cursor + i) % n
                slot = self._slots[idx]

                if slot.status == ProxyHealth.IN_USE:
                    continue

                if slot.status == ProxyHealth.QUARANTINED:
                    if now < slot.cooldown_until:
                        continue
                    # Quarantine expired -> auto-recover circuit breaker
                    logger.info("ProxyManager | [{}] quarantine expired; auto-recovering to HEALTHY", slot.label)
                    slot.status = ProxyHealth.HEALTHY
                    slot.fail_count = 0

                if slot.status == ProxyHealth.COOLING:
                    if now < slot.cooldown_until:
                        continue
                    # Cooldown expired -> auto-recover
                    slot.status = ProxyHealth.HEALTHY

                self._cursor = (idx + 1) % n
                slot.status = ProxyHealth.IN_USE
                slot.requests += 1
                return RentalResult(acquired=True, config=slot.config, slot=slot)

            return RentalResult(acquired=False, config=None, slot=None)

    async def rent(self) -> ProxyConfig | None:
        """
        Return the next usable proxy config, or None when every slot is busy/blocked
        or when renting a direct connection. Use `rent_result()` to distinguish
        between pool exhaustion and direct connection acquisition.
        """
        res = await self.rent_result()
        if not res.acquired:
            return None
        return res.config

    async def release(
        self,
        config: ProxyConfig | None,
        ok: bool,
        error: BaseException | None = None,
    ) -> None:
        """Return a rented proxy and apply health / backoff rules."""
        async with self._lock:
            slot = self._find(config)
            if slot is None:
                return
            if ok:
                slot.ok_count += 1
                slot.fail_count = 0
                slot.last_error = ""
                slot.status = ProxyHealth.HEALTHY
                return

            kind = classify_proxy_error(error) if error else ProxyErrorKind.OTHER
            slot.fail_count += 1
            backoff = min(self._cooldown_base * (2 ** (slot.fail_count - 1)),
                          self._cooldown_max)

            # Auth failures and handshake failures quarantine directly (they
            # will not recover on their own); hitting the circuit threshold
            # also trips the breaker.
            blocked = (
                kind in (ProxyErrorKind.AUTH, ProxyErrorKind.PROXY_HANDSHAKE)
                or slot.fail_count >= self._circuit_threshold
            )
            if blocked:
                slot.status = ProxyHealth.QUARANTINED
                slot.cooldown_until = time.monotonic() + self._quarantine_seconds
            else:
                slot.status = ProxyHealth.COOLING
                slot.cooldown_until = time.monotonic() + backoff
            slot.last_error = f"{kind.value}: {str(error)[:140]}"
            logger.warning(
                "ProxyManager | [{}] -> {} (fail #{}) [{}]",
                slot.label, slot.status.value, slot.fail_count, slot.last_error,
            )

    # ── internals ─────────────────────────────────────────────────────────────

    def _find(self, config: ProxyConfig | None) -> ProxySlot | None:
        for slot in self._slots:
            if slot.config is None and config is None:
                return slot
            if slot.config is not None and slot.config is config:
                return slot
        return None