"""
dashboard_server.py
--------------------
A minimal standalone HTTP server that serves a realistic, real-time updating
dashboard with a table containing: Timestamp | Event Name | Numeric Result

Run this first so the scraper has a target to work against:
    python dashboard_server.py

Then open http://localhost:8765 in a browser to verify it before running the scraper.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import json
import random
import math

# ---------------------------------------------------------------------------
# Event catalogue – used to generate realistic synthetic data
# ---------------------------------------------------------------------------
EVENT_CATALOGUE = [
    "CPU_SPIKE",
    "MEMORY_PRESSURE",
    "DISK_IO_BURST",
    "NETWORK_LATENCY",
    "API_RESPONSE_TIME",
    "DB_QUERY_DURATION",
    "CACHE_HIT_RATE",
    "ERROR_RATE",
    "THROUGHPUT_OPS",
    "ACTIVE_SESSIONS",
    "QUEUE_DEPTH",
    "GC_PAUSE_MS",
    "PACKET_LOSS_PCT",
    "CPU_TEMPERATURE",
    "HEAP_UTILIZATION",
]

# ---------------------------------------------------------------------------
# In-memory store – the server mutates this every second
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_rows: list[dict] = []
_MAX_ROWS = 200

def _generate_row() -> dict:
    """Produce one realistic metric row."""
    event = random.choice(EVENT_CATALOGUE)
    # Each event has a plausible numeric range
    ranges = {
        "CPU_SPIKE": (45.0, 99.9),
        "MEMORY_PRESSURE": (30.0, 95.0),
        "DISK_IO_BURST": (0.5, 800.0),
        "NETWORK_LATENCY": (0.2, 350.0),
        "API_RESPONSE_TIME": (5.0, 2000.0),
        "DB_QUERY_DURATION": (1.0, 5000.0),
        "CACHE_HIT_RATE": (60.0, 99.5),
        "ERROR_RATE": (0.0, 8.0),
        "THROUGHPUT_OPS": (100.0, 50000.0),
        "ACTIVE_SESSIONS": (1.0, 5000.0),
        "QUEUE_DEPTH": (0.0, 10000.0),
        "GC_PAUSE_MS": (0.5, 300.0),
        "PACKET_LOSS_PCT": (0.0, 5.0),
        "CPU_TEMPERATURE": (35.0, 90.0),
        "HEAP_UTILIZATION": (20.0, 95.0),
    }
    lo, hi = ranges.get(event, (0.0, 100.0))
    # Add a sine-wave oscillation to make time-series patterns visible
    t = time.time()
    base = lo + (hi - lo) * (0.5 + 0.5 * math.sin(t / 30.0 + hash(event) % 10))
    noise = random.gauss(0, (hi - lo) * 0.05)
    value = max(lo, min(hi, base + noise))

    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
    return {"timestamp": ts, "event_name": event, "numeric_result": round(value, 4)}


def _background_generator():
    """Push a new row every ~1 second indefinitely."""
    while True:
        row = _generate_row()
        with _lock:
            _rows.append(row)
            if len(_rows) > _MAX_ROWS:
                _rows.pop(0)
        time.sleep(random.uniform(0.8, 1.5))


# ---------------------------------------------------------------------------
# HTML dashboard template
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Real-Time Metrics Dashboard</title>
  <style>
    :root {
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --accent: #58a6ff; --success: #3fb950; --warn: #d29922;
      --danger: #f85149; --text: #c9d1d9; --muted: #8b949e;
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; }
    header { padding:1.2rem 2rem; background:var(--surface);
             border-bottom:1px solid var(--border); display:flex;
             align-items:center; gap:1rem; }
    header h1 { font-size:1.2rem; font-weight:600; color:var(--accent); }
    #status { font-size:.8rem; color:var(--success); margin-left:auto; }
    #status.stale { color:var(--warn); }
    .container { padding:1.5rem 2rem; }
    .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
    .stat-card { background:var(--surface); border:1px solid var(--border);
                 border-radius:8px; padding:1rem 1.2rem; }
    .stat-card .label { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
    .stat-card .value { font-size:1.8rem; font-weight:700; margin-top:.25rem; }
    .table-wrapper { background:var(--surface); border:1px solid var(--border);
                     border-radius:8px; overflow:hidden; }
    .table-header { padding:.75rem 1rem; border-bottom:1px solid var(--border);
                    font-size:.8rem; color:var(--muted); display:flex; justify-content:space-between; }
    table { width:100%; border-collapse:collapse; }
    thead th { padding:.6rem 1rem; text-align:left; font-size:.75rem;
               text-transform:uppercase; letter-spacing:.08em;
               color:var(--muted); border-bottom:1px solid var(--border);
               position:sticky; top:0; background:var(--surface); }
    tbody tr { border-bottom:1px solid var(--border); transition:background .15s; }
    tbody tr:hover { background:rgba(88,166,255,.06); }
    tbody tr.new-row { animation:flash .6s ease; }
    tbody td { padding:.55rem 1rem; font-size:.85rem; font-family:'Courier New',monospace; }
    td.ts { color:var(--muted); font-size:.78rem; }
    td.ev { color:var(--accent); font-weight:600; font-family:inherit; }
    td.num { text-align:right; }
    td.num.high { color:var(--danger); }
    td.num.mid  { color:var(--warn); }
    td.num.low  { color:var(--success); }
    @keyframes flash { from{background:rgba(88,166,255,.25)} to{background:transparent} }
    #metrics-table-container { max-height:520px; overflow-y:auto; }
  </style>
</head>
<body>
  <header>
    <h1>⚡ Real-Time Metrics Dashboard</h1>
    <span id="status">● LIVE</span>
  </header>

  <div class="container">
    <div class="stats">
      <div class="stat-card">
        <div class="label">Total Events</div>
        <div class="value" id="stat-total">0</div>
      </div>
      <div class="stat-card">
        <div class="label">Last Update</div>
        <div class="value" id="stat-last" style="font-size:1rem;margin-top:.5rem">—</div>
      </div>
      <div class="stat-card">
        <div class="label">Avg Numeric Result</div>
        <div class="value" id="stat-avg">—</div>
      </div>
    </div>

    <div class="table-wrapper">
      <div class="table-header">
        <span>Live Event Stream</span>
        <span id="row-count">0 rows</span>
      </div>
      <div id="metrics-table-container">
        <table id="metrics-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Event Name</th>
              <th class="num">Numeric Result</th>
            </tr>
          </thead>
          <tbody id="metrics-tbody">
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let prevCount = 0;
    let lastFetch = Date.now();

    function classify(v, lo, hi) {
      const pct = (v - lo) / (hi - lo);
      if (pct > 0.75) return 'high';
      if (pct > 0.40) return 'mid';
      return 'low';
    }

    async function fetchData() {
      try {
        const res = await fetch('/api/data');
        const data = await res.json();
        lastFetch = Date.now();

        // Stats
        document.getElementById('stat-total').textContent = data.total;
        document.getElementById('row-count').textContent = data.rows.length + ' rows';
        if (data.rows.length) {
          document.getElementById('stat-last').textContent =
            data.rows[data.rows.length - 1].timestamp.split('T')[1];
          const avg = data.rows.reduce((s, r) => s + r.numeric_result, 0) / data.rows.length;
          document.getElementById('stat-avg').textContent = avg.toFixed(2);
        }

        // Table diff
        const tbody = document.getElementById('metrics-tbody');
        const newCount = data.rows.length;
        const added = newCount - prevCount;

        if (added > 0) {
          // Insert new rows at the top
          const newRows = data.rows.slice(prevCount);
          newRows.reverse().forEach(row => {
            const tr = document.createElement('tr');
            tr.className = 'new-row';
            const cls = 'mid'; // simplified classification
            tr.innerHTML = `
              <td class="ts">${row.timestamp}</td>
              <td class="ev">${row.event_name}</td>
              <td class="num ${cls}">${row.numeric_result.toFixed(4)}</td>`;
            tbody.prepend(tr);
          });
          prevCount = newCount;
        }

        document.getElementById('status').textContent = '● LIVE';
        document.getElementById('status').className = '';
      } catch(e) {
        document.getElementById('status').textContent = '○ STALE';
        document.getElementById('status').className = 'stale';
      }
    }

    // Poll every 1.2 s for near-real-time feel
    fetchData();
    setInterval(fetchData, 1200);
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log spam
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/data":
            self._serve_json()
        else:
            self.send_error(404)

    def _serve_html(self):
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self):
        with _lock:
            payload = json.dumps({
                "total": len(_rows),
                "rows": list(_rows),
            }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PORT = 8765
    # Start background data generator
    t = threading.Thread(target=_background_generator, daemon=True)
    t.start()

    server = HTTPServer(("localhost", PORT), DashboardHandler)
    print(f"[dashboard_server] Serving on http://localhost:{PORT}")
    print("[dashboard_server] Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard_server] Stopped.")
