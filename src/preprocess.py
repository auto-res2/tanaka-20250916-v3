"""src/preprocess.py
Data-loading and pre-processing logic.  As with the other stubs, this
module exists only so that relative imports resolve.  Any call into it
will raise a `RuntimeError` describing the problem.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def preprocess(cfg: Dict[str, Any]) -> None:  # pragma: no cover – stub
    """Stubbed preprocessing entry point."""
    LOGGER.error("No Experiment Code was provided – aborting preprocessing.")
    raise RuntimeError(
        "preprocess() cannot execute because the source Experiment Code is "
        "missing.  Please supply the original script so that it can be "
        "refactored into this project structure."
    )
