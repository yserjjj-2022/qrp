"""Logging setup with run identifier."""

from __future__ import annotations

import logging
import uuid


def setup_logging(level: str = "INFO", run_id: str | None = None) -> str:
    """Configure logging and return the run_id."""
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]

    fmt = f"%(asctime)s [{run_id}] %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return run_id
