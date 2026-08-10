"""
app.py
------
FastAPI Web UI wrapper and WebSocket telemetry server for Playwright scraper.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import shutil
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from scraper import run_scraper_async, get_status_snapshot


# ── Global State ─────────────────────────────────────────────────────────────
_active_task: asyncio.Task | None = None
_start_time: float | None = None
_job_config: dict[str, Any] = {}
_latest_snapshot: dict[str, Any] = {}


# ── WebSocket Manager ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        payload = json.dumps(message)
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                to_remove.append(connection)
        for dead in to_remove:
            self.disconnect(dead)


manager = ConnectionManager()


# ── Telemetry Callback Hook ──────────────────────────────────────────────────
async def _ws_telemetry_hook(snapshot: dict):
    global _latest_snapshot
    _latest_snapshot = snapshot
    state = "running" if (_active_task and not _active_task.done()) else "idle"
    elapsed = round(time.monotonic() - _start_time, 1) if _start_time and state == "running" else 0

    broadcast_data = {
        "type": "status_update",
        "engine_state": state,
        "elapsed_seconds": elapsed,
        "job_config": _job_config,
        "snapshot": snapshot,
    }
    await manager.broadcast(broadcast_data)


# ── Request / Response Models ────────────────────────────────────────────────
class StartRequest(BaseModel):
    target_file: str = "targets.json"
    proxy_file: str = "proxies.txt"
    workers: int = 9
    interval: int = 10
    max_cycles: int = 0
    show_browser: bool = False



class ProbeRequest(BaseModel):
    target: str = "default"
    stake: float = 100.0
    predicted_odds: float = 2.0
    strategy: str = "VARIANCE_ARBITRAGE"



# ── Background Runner ────────────────────────────────────────────────────────
async def _run_scraper_task(req: StartRequest):
    global _active_task, _start_time, _job_config
    _start_time = time.monotonic()
    _job_config = req.model_dump()

    target_file_path = req.target_file if Path(req.target_file).exists() else None
    proxy_file_path = req.proxy_file if (req.proxy_file and Path(req.proxy_file).exists()) else None

    args = argparse.Namespace(
        command="run",
        url="http://localhost:8765",
        table_id="table",
        targets=[],
        target_file=target_file_path,
        output_dir="scraped_data",
        interval=req.interval,
        max_cycles=req.max_cycles,
        show_browser=req.show_browser,
        no_console=True,
        log_level="INFO",
        proxy_file=proxy_file_path,
        workers=req.workers,
        worker_timeout=30,
    )

    try:
        logger.info("Starting background Playwright scraper engine via Web UI...")
        await run_scraper_async(args, status_callback=_ws_telemetry_hook)
    except asyncio.CancelledError:
        logger.warning("Scraper task cancelled by user request.")
    except Exception as exc:
        logger.error("Scraper engine task failed: {}", exc)
    finally:
        _start_time = None
        # Send final idle state
        await manager.broadcast({
            "type": "status_update",
            "engine_state": "idle",
            "elapsed_seconds": 0,
            "job_config": _job_config,
            "snapshot": await get_status_snapshot(),
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db import db
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(title="Real-Time Telemetry Dashboard Scraper", version="2.0.0", lifespan=lifespan)


# ── API Routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)

async def get_index():
    index_path = Path("templates") / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="templates/index.html not found.")
    return index_path.read_text(encoding="utf-8")


@app.post("/api/start")
@app.post("/start")
async def start_scraper(req: StartRequest):
    global _active_task
    if _active_task and not _active_task.done():
        raise HTTPException(status_code=400, detail="Scraper engine is already running.")

    _active_task = asyncio.create_task(_run_scraper_task(req))
    return {"status": "success", "message": "Scraper engine started.", "config": req.model_dump()}


@app.post("/api/stop")
@app.post("/stop")
async def stop_scraper():
    global _active_task
    if not _active_task or _active_task.done():
        return {"status": "info", "message": "Scraper engine is not currently running."}

    _active_task.cancel()
    try:
        await _active_task
    except asyncio.CancelledError:
        pass
    _active_task = None
    return {"status": "success", "message": "Scraper engine stopped successfully."}








@app.get("/api/telemetry")
async def get_telemetry_history(limit: int = 50):
    from db import db
    records = await db.get_recent_scrapes(limit=limit)
    return {"status": "success", "count": len(records), "data": records}


@app.get("/api/drift-features")
async def get_drift_features(limit: int = 50):
    from db import db
    features = await db.get_aggregated_drift_features(limit=limit)
    return {"status": "success", "count": len(features), "features": features}


@app.get("/api/oracle-cards")
async def get_oracle_cards(limit: int = 20):
    try:
        from oracle_alerts import oracle_engine
        cards = oracle_engine.get_active_cards(limit=limit)
    except Exception:
        cards = []
    return {"status": "success", "count": len(cards), "cards": cards}



class PredictRequest(BaseModel):
    rolling_rtp_10: float = 1.0
    rtp_z_score: float = 0.0
    sequence_entropy: float = 1.0
    cluster_index: float = 1.0
    seed_volatility: float = 1.0


@app.post("/api/audit-predict")
async def predict_auditor_state(req: PredictRequest):
    from train_auditor import auditor_predictor
    res = auditor_predictor.predict_correction_probability(req.model_dump())
    return {"status": "success", "prediction": res}


@app.post("/api/audit-train")
async def train_auditor_model():
    from db import db
    from train_auditor import PRNGAuditorTrainer
    records = await db.get_recent_scrapes(limit=500)
    trainer = PRNGAuditorTrainer()
    res = trainer.train_from_telemetry_rows(records)
    return res




@app.get("/api/db-status")
async def get_db_status():
    from db import db
    return {
        "db_connected": db.is_connected,
        "mode": "supabase_postgresql" if db.is_connected else "local_fallback",
    }


@app.get("/api/status")
async def get_status():
    from db import db
    is_running = _active_task is not None and not _active_task.done()
    snapshot = await get_status_snapshot()
    elapsed = round(time.monotonic() - _start_time, 1) if (_start_time and is_running) else 0
    return {
        "engine_state": "running" if is_running else "idle",
        "elapsed_seconds": elapsed,
        "db_connected": db.is_connected,
        "job_config": _job_config,
        "snapshot": snapshot,
    }


@app.post("/api/probe")

async def execute_micro_probe(req: ProbeRequest):
    from rtp_engine import rtp_calculator
    stats = rtp_calculator.get_rtp_stats(req.target)

    delta = stats["divergence_delta"]
    confidence = round(min(1.0, max(0.5, 0.75 + (delta * 0.5))), 4)
    simulated_payout = round(req.stake * req.predicted_odds if delta < 0.15 else 0.0, 2)

    result = {
        "probe_id": f"PRB-{int(time.time()*1000)}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": req.target,
        "stake_ngn": req.stake,
        "predicted_odds": req.predicted_odds,
        "strategy": req.strategy,
        "rtp_horizons": {
            "short_10": stats["short_rtp"],
            "med_100": stats["med_rtp"],
            "macro_1000": stats["macro_rtp"],
            "divergence_delta": delta,
        },
        "variance_confidence": confidence,
        "execution_status": "EXECUTED_SUCCESS",
        "simulated_payout_ngn": simulated_payout,
        "net_yield_ngn": round(simulated_payout - req.stake, 2),
    }
    logger.info("[MICRO-PROBE] Executed validation probe for target={}: stake={} NGN, confidence={}", req.target, req.stake, confidence)
    return result




@app.get("/api/download")
async def download_data():
    scraped_dir = Path("scraped_data")
    if not scraped_dir.exists() or not any(scraped_dir.iterdir()):
        raise HTTPException(status_code=404, detail="No scraped data files found.")

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in scraped_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(scraped_dir)
                zf.write(file, arcname=str(arcname))

    mem_zip.seek(0)
    filename = f"scraped_data_{int(time.time())}.zip"
    return StreamingResponse(
        mem_zip,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status snapshot immediately on connect
        snapshot = await get_status_snapshot()
        is_running = _active_task is not None and not _active_task.done()
        elapsed = round(time.monotonic() - _start_time, 1) if (_start_time and is_running) else 0
        await websocket.send_text(json.dumps({
            "type": "status_update",
            "engine_state": "running" if is_running else "idle",
            "elapsed_seconds": elapsed,
            "job_config": _job_config,
            "snapshot": snapshot,
        }))

        while True:
            # Maintain heartbeat / receive client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket exception: {}", exc)
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

