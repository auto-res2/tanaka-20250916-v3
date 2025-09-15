"""src/train.py
This module would normally contain the model definition and the training
loop extracted from the original single-file experiment.  Unfortunately,
no runnable Experiment Code was provided – the field literally contains
only the sentence "I’m sorry, but I can’t comply with that."  

To keep the project importable we expose a `train` stub that explicitly
fails with a clear message so that downstream scripts terminate in a
controlled fashion rather than with obscure `AttributeError`s.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def train(cfg: Dict[str, Any]) -> None:  # pragma: no cover – stub
    """Stubbed training entry point.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed YAML configuration.
    """
    LOGGER.error("No Experiment Code was provided – aborting training.")
    raise RuntimeError(
        "train() cannot execute because the source Experiment Code is "
        "missing.  Please supply the original script so that it can be "
        "refactored into this project structure."
    )
