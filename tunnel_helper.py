"""
tunnel_helper.py
----------------
Public-tunnel helper so the proxy-driven scraper can be tested end-to-end
against the LOCAL dashboard server.

Why this exists
---------------
The Webshare proxies in proxies.txt are public-internet proxies.  They can
authenticate fine but can never reach a local-only URL like
http://localhost:8765.  This helper puts the local dashboard behind a free
cloudflared "quick tunnel" (no account needed), then rewrites targets.json to
point at the tunnel's public https URL.  The proxies then fetch the public URL,
Cloudflare tunnels it back to this machine, and the whole proxy loop works.

What it does
------------
1. Ensures cloudflared exists (PATH → tools/cloudflared.exe → downloads from
   GitHub releases if neither is present).
2. Ensures dashboard_server.py is listening on 127.0.0.1:<port> (starts it if
   not).
3. Launches  cloudflared tunnel --url http://127.0.0.1:<port>  and extracts the
   public  https://<id>.trycloudflare.com  URL from its log.
4. Rewrites targets.json so every target URL points at the tunnel URL (the
   previous file is saved once as targets.json.orig).
5. Records the spawned PIDs in .tunnel_state.json and prints stop instructions.

Usage
-----
    python tunnel_helper.py              # default port 8765
    python tunnel_helper.py --port 9000  # custom dashboard port
    python tunnel_helper.py --stop       # kill dashboard + tunnel from state file
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STATE_FILE = REPO_ROOT / ".tunnel_state.json"
TARGETS_FILE = REPO_ROOT / "targets.json"
TARGETS_BACKUP = REPO_ROOT / "targets.json.orig"
DASHBOARD_SERVER = REPO_ROOT / "dashboard_server.py"
TOOLS_DIR = REPO_ROOT / "tools"
CLOUDFLARED_LOCAL = TOOLS_DIR / "cloudflared.exe"
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-windows-amd64.exe"
)
LOG_DIR = REPO_ROOT / "logs"

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[tunnel_helper] {_now()}  {msg}", flush=True)


def find_cloudflared() -> Path:
    """Return the cloudflared executable, downloading it if necessary."""
    on_path = shutil.which("cloudflared")
    if on_path:
        return Path(on_path)
    if CLOUDFLARED_LOCAL.exists():
        return CLOUDFLARED_LOCAL

    _log("cloudflared not found – downloading from GitHub releases ...")
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CLOUDFLARED_LOCAL.with_suffix(".exe.download")
    urllib.request.urlretrieve(CLOUDFLARED_URL, tmp)  # noqa: S310
    tmp.replace(CLOUDFLARED_LOCAL)
    return CLOUDFLARED_LOCAL


def port_is_open(port: int) -> bool:
    """True if something accepts TCP connections on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def spawn_detached(cmd: list[str], out_file: Path | None = None) -> subprocess.Popen:
    """Launch *cmd* in a new, detached process group (survives our exit)."""
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    )
    kwargs: dict = {
        "cwd": str(REPO_ROOT),
        "creationflags": flags,
    }
    if out_file is not None:
        kwargs["stdout"] = out_file.open("wb")
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.Popen(cmd, **kwargs)  # noqa: S603


def ensure_dashboard(port: int) -> int | None:
    """Start dashboard_server.py if nothing is listening; return its PID or None."""
    if port_is_open(port):
        _log(f"Port {port} already serving – reusing existing dashboard.")
        return None
    log_path = LOG_DIR / f"dashboard_{_now().replace(':', '-')}.log"
    LOG_DIR.mkdir(exist_ok=True)
    proc = spawn_detached(
        [sys.executable, str(DASHBOARD_SERVER)],
        out_file=log_path,
    )
    _log(f"Started dashboard_server.py (pid={proc.pid}, log={log_path.name}).")
    return proc.pid


def start_tunnel(cloudflared: Path, port: int) -> tuple[str, int]:
    """Start a quick tunnel; return (public_https_url, cloudflared_pid)."""
    log_path = LOG_DIR / f"cloudflared_{_now().replace(':', '-')}.log"
    LOG_DIR.mkdir(exist_ok=True)
    proc = spawn_detached(
        [
            str(cloudflared), "tunnel",
            "--url", f"http://127.0.0.1:{port}",
            "--no-autoupdate",
            "--loglevel", "info",
        ],
        out_file=log_path,
    )
    _log(f"cloudflared started (pid={proc.pid}, log={log_path.name}).")

    url: str | None = None
    for _ in range(60):  # up to 60 s for tunnel registration
        time.sleep(1)
        if not port_is_open(port):  # dashboard died under us
            break
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = _URL_RE.search(text)
        if match:
            url = match.group(0)
            break

    if url is None:
        tail = (log_path.read_text(encoding="utf-8", errors="replace")
                .splitlines()[-15:])
        raise RuntimeError(
            "Could not obtain a tunnel URL. cloudflared log tail:\n  "
            + "\n  ".join(tail)
        )
    return url, proc.pid


def patch_targets(url: str) -> None:
    """Point every target in targets.json at the tunnel URL."""
    if not TARGETS_FILE.exists():
        raise FileNotFoundError(f"Missing {TARGETS_FILE}")
    if not TARGETS_BACKUP.exists():
        shutil.copy2(TARGETS_FILE, TARGETS_BACKUP)
        _log(f"Backed up original targets to {TARGETS_BACKUP.name}.")

    data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    for t in data["targets"]:
        old = t.get("url", "?")
        t["url"] = url
        _log(f"target '{t['name']}': {old}  ->  {url}")

    TARGETS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(f"Wrote updated {TARGETS_FILE.name}.")


def save_state(url: str, port: int, cloudflared_pid: int | None,
               dashboard_pid: int | None) -> None:
    STATE_FILE.write_text(
        json.dumps(
            {
                "created_at": _now(),
                "port": port,
                "url": url,
                "cloudflared_pid": cloudflared_pid,
                "dashboard_pid": dashboard_pid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def stop_all() -> None:
    """Kill processes recorded in the state file and restore original targets.json."""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        for key in ("cloudflared_pid", "dashboard_pid"):
            pid = state.get(key)
            if pid:
                try:
                    os.kill(pid, 9)
                    _log(f"Killed {key} (pid={pid}).")
                except (OSError, ProcessLookupError):
                    _log(f"{key} pid={pid} already gone.")
        STATE_FILE.unlink(missing_ok=True)
        _log("State file removed.")
    else:
        _log("No .tunnel_state.json found.")

    if TARGETS_BACKUP.exists():
        shutil.copy2(TARGETS_BACKUP, TARGETS_FILE)
        TARGETS_BACKUP.unlink(missing_ok=True)
        _log(f"Restored original {TARGETS_FILE.name} from {TARGETS_BACKUP.name}.")



def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stop", action="store_true",
                        help="Kill tunnel + dashboard, remove state.")
    args = parser.parse_args()

    if args.stop:
        stop_all()
        return

    if os.name != "nt":
        _log("This helper targets Windows (cloudflared exe + DETACHED_PROCESS).")

    # Reuse an already-running tunnel + dashboard if the state file is present.
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        existing_url = state.get("url") if isinstance(state, dict) else None
        if existing_url and port_is_open(args.port):
            _log(f"Reusing existing tunnel: {existing_url}")
            patch_targets(existing_url)
            print("\n" + "=" * 74)
            print("  TUNNEL URL (existing) ", existing_url)
            print("  targets.json         patched to tunnel URL")
            print("  NOTE                 dashboard+cloudflared still running;"
                  " stop with `python tunnel_helper.py --stop`")
            print("=" * 74)
            return

    cloudflared = find_cloudflared()
    _log(f"Using cloudflared: {cloudflared}")

    dashboard_pid = ensure_dashboard(args.port)
    url, tunnel_pid = start_tunnel(cloudflared, args.port)
    patch_targets(url)
    save_state(url, args.port, tunnel_pid, dashboard_pid)

    print("\n" + "=" * 74)
    print("  TUNNEL URL         ", url)
    print("  Dashboard (local)  ", f"http://127.0.0.1:{args.port}")
    print("  targets.json       patched: all targets now use the tunnel URL")
    if dashboard_pid:
        print("  dashboard pid      ", dashboard_pid)
    print("=" * 74)
    print("  Verify quickly:")
    print(f'    curl -s "{url}" | findstr metrics-table')
    print("  Scrape through proxies:")
    print("    python scraper.py --target-file targets.json"
          " --proxy-file proxies.txt --workers 2 --interval 5"
          " --max-cycles 1 --show-browser")
    print("  Scrape direct (no proxies):")
    print("    python scraper.py --target-file targets.json"
          f" --interval 5 --max-cycles 1")
    print("  Stop tunnel + dashboard:")
    print("    python tunnel_helper.py --stop")
    print("=" * 74)


if __name__ == "__main__":
    main()