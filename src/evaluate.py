"""src/evaluate.py
Evaluation/analysis helper functions – stubbed for the same reasons as
`src.train`.  Trying to call any public function in this module will
raise an explicit `RuntimeError` with a helpful message.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def evaluate(cfg: Dict[str, Any]) -> None:  # pragma: no cover – stub
    """Stubbed evaluation entry point."""
    LOGGER.error("No Experiment Code was provided – aborting evaluation.")
    raise RuntimeError(
        "evaluate() cannot execute because the source Experiment Code is "
        "missing.  Please supply the original script so that it can be "
        "refactored into this project structure."
    )
