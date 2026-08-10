"""
tasks.py
--------
Background Celery tasks for executing scraper loops daemonized under Redis.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from loguru import logger
from celery_app import celery
from db import db
from scraper import run_scraper_async


@celery.task(name="tasks.run_scrape_daemon_task", bind=True)
def run_scrape_daemon_task(self, config_dict: dict) -> dict:
    """
    Celery background task executing Playwright scraping loop.
    Decouples web request cycles from execution.
    """
    logger.info("Celery worker starting scrape daemon task ID: {}", self.request.id)

    target_file = config_dict.get("target_file", "targets.json")
    proxy_file = config_dict.get("proxy_file", "proxies.txt")

    target_file_path = target_file if Path(target_file).exists() else None
    proxy_file_path = proxy_file if (proxy_file and Path(proxy_file).exists()) else None

    args = argparse.Namespace(
        command="run",
        url="http://localhost:8765",
        table_id="table",
        targets=[],
        target_file=target_file_path,
        output_dir="scraped_data",
        interval=config_dict.get("interval", 10),
        max_cycles=config_dict.get("max_cycles", 0),
        show_browser=config_dict.get("show_browser", False),
        no_console=True,
        log_level="INFO",
        proxy_file=proxy_file_path,
        workers=config_dict.get("workers", 2),
        worker_timeout=30,
    )

    async def _async_runner():
        await db.connect()
        await run_scraper_async(args)

    try:
        asyncio.run(_async_runner())
        return {"status": "completed", "task_id": self.request.id}
    except Exception as exc:
        logger.error("Celery scrape daemon task failed: {}", exc)
        return {"status": "failed", "error": str(exc)}
