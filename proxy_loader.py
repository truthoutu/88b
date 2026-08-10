"""
proxy_loader.py
---------------
Loads, validates, and deduplicates SOCKS5/SOCKS4/HTTP proxies from a plain-
text file and returns them as ProxyConfig objects ready for Playwright.

Supported URI formats
─────────────────────
    socks5://user:pass@host:port   (authenticated)
    socks5://host:port             (anonymous)
    socks4://host:port
    http://user:pass@host:port
    http://host:port

Usage
─────
    from proxy_loader import load_proxies

    proxies = load_proxies("proxies.txt")   # returns [] if file absent
    for p in proxies:
        print(p.server, p.label)            # "socks5://host:port", "proxy-0"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

class ProxyConfig(NamedTuple):
    """
    Playwright-ready proxy configuration.

    ``server`` is the proxy URI without credentials, as Playwright expects
    credentials in separate ``username`` / ``password`` fields.

    ``label`` is a short human-readable tag used in log messages and CSV
    filenames (e.g. "p00", "p01", …).
    """
    server: str           # e.g. "socks5://192.168.1.10:1080"
    username: str | None  # None if proxy is anonymous
    password: str | None  # None if proxy is anonymous
    label: str            # e.g. "p00"

    def as_playwright_proxy(self) -> dict:
        """Return a dict suitable for Playwright's ``proxy=`` parameter."""
        d: dict = {"server": self.server}
        if self.username:
            d["username"] = self.username
        if self.password:
            d["password"] = self.password
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_SCHEMES = {"socks5", "socks4", "http", "https"}

# Quick sanity-check: host must look like an IP or hostname, port 1–65535
_HOST_RE = re.compile(
    r"^[a-zA-Z0-9._\-]+"          # hostname or IP
    r"$"
)
_PORT_RANGE = range(1, 65536)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_proxy_line(line: str, line_no: int) -> ProxyConfig | None:
    """
    Parse one raw proxy line.  Returns None if the line is invalid and logs
    a warning so the caller can skip it gracefully.
    """
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None

    # urllib.parse requires a scheme; add a dummy one if missing so we can
    # at least give a meaningful error message.
    if "://" not in raw:
        logger.warning(
            "proxy_loader | line {:d}: missing scheme (e.g. socks5://). "
            "Skipping: {!r}", line_no, raw
        )
        return None

    try:
        parsed = urlparse(raw)
    except Exception as exc:
        logger.warning(
            "proxy_loader | line {:d}: parse error – {}. Skipping: {!r}",
            line_no, exc, raw
        )
        return None

    scheme = (parsed.scheme or "").lower()
    if scheme not in _SUPPORTED_SCHEMES:
        logger.warning(
            "proxy_loader | line {:d}: unsupported scheme '{}'. "
            "Supported: {}. Skipping.",
            line_no, scheme, ", ".join(sorted(_SUPPORTED_SCHEMES))
        )
        return None

    host = parsed.hostname or ""
    if not host or not _HOST_RE.match(host):
        logger.warning(
            "proxy_loader | line {:d}: invalid or missing host. Skipping: {!r}",
            line_no, raw
        )
        return None

    port = parsed.port
    if port is None or port not in _PORT_RANGE:
        logger.warning(
            "proxy_loader | line {:d}: missing or out-of-range port. Skipping: {!r}",
            line_no, raw
        )
        return None

    username = parsed.username or None
    password = parsed.password or None

    # Build server URI *without* credentials (Playwright takes them separately)
    server = f"{scheme}://{host}:{port}"

    # Placeholder label – caller assigns the final index-based label
    return ProxyConfig(
        server=server,
        username=username,
        password=password,
        label="",  # filled in by load_proxies()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_proxies(path: str | Path) -> list[ProxyConfig]:
    """
    Load and validate proxies from *path*.

    Returns a deduplicated list of :class:`ProxyConfig` objects.  If the
    file does not exist, logs a warning and returns an empty list (caller
    should then run in no-proxy / single-context mode).

    Parameters
    ----------
    path:
        Path to the proxy list text file (absolute or relative).
    """
    proxy_path = Path(path)

    if not proxy_path.exists():
        logger.warning(
            "proxy_loader | Proxy file not found: {}. "
            "Running in single-context (no proxy) mode.", proxy_path.resolve()
        )
        return []

    raw_lines = proxy_path.read_text(encoding="utf-8").splitlines()
    logger.info(
        "proxy_loader | Reading {} lines from {}", len(raw_lines), proxy_path.resolve()
    )

    parsed: list[ProxyConfig] = []
    seen_servers: set[str] = set()

    for line_no, line in enumerate(raw_lines, start=1):
        cfg = _parse_proxy_line(line, line_no)
        if cfg is None:
            continue

        if cfg.server in seen_servers:
            logger.debug(
                "proxy_loader | line {:d}: duplicate server '{}' – skipping.",
                line_no, cfg.server
            )
            continue

        seen_servers.add(cfg.server)
        parsed.append(cfg)

    # Assign short labels now that we know the full count
    width = max(len(str(len(parsed) - 1)), 2)  # zero-pad width
    labelled: list[ProxyConfig] = [
        ProxyConfig(
            server=p.server,
            username=p.username,
            password=p.password,
            label=f"p{str(i).zfill(width)}",
        )
        for i, p in enumerate(parsed)
    ]

    if labelled:
        logger.success(
            "proxy_loader | Loaded {:d} valid proxy/proxies.", len(labelled)
        )
        for p in labelled:
            creds = f" (auth: {p.username})" if p.username else " (anonymous)"
            logger.debug("proxy_loader |   [{}] {}{}", p.label, p.server, creds)
    else:
        logger.warning(
            "proxy_loader | No valid proxies found in {}. "
            "Running in single-context (no proxy) mode.", proxy_path
        )

    return labelled


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (python proxy_loader.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    path_arg = sys.argv[1] if len(sys.argv) > 1 else "proxies.txt"
    proxies = load_proxies(path_arg)

    if not proxies:
        print("No proxies loaded.")
        sys.exit(0)

    print(f"\nLoaded {len(proxies)} proxy/proxies:\n")
    print(f"  {'Label':<8} {'Server':<40} {'Username':<20} {'Auth?'}")
    print(f"  {'-'*8} {'-'*40} {'-'*20} {'-'*5}")
    for p in proxies:
        print(
            f"  {p.label:<8} {p.server:<40} "
            f"{p.username or '—':<20} {'yes' if p.username else 'no'}"
        )
    print()
