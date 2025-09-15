"""src/main.py
CLI entry point that would orchestrate smoke-test and full-experiment
runs.  Because the core logic is absent we simply parse the flags and
exit with a descriptive error so that users understand why nothing runs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Relative imports – these resolve to the stubbed modules above
from .preprocess import preprocess
from .train import train
from .evaluate import evaluate

LOGGER = logging.getLogger("dynamic_flash_gat")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SMOKE_FILE = CONFIG_DIR / "smoke_test.yaml"
FULL_FILE = CONFIG_DIR / "full_experiment.yaml"


def load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} not found.")
    try:
        with path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)
    except yaml.YAMLError as exc:
        LOGGER.error("Failed to parse YAML config %s: %s", path, exc)
        raise


def run_phase(cfg_path: Path) -> None:
    """Run a single experiment phase given a YAML file."""
    cfg = load_cfg(cfg_path)

    # The following calls will immediately raise because the modules are stubs.
    preprocess(cfg)
    train(cfg)
    evaluate(cfg)


def main() -> None:  # pragma: no cover – CLI only
    parser = argparse.ArgumentParser(description="Dynamic-FlashGAT experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run only the smoke-test configuration")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment configuration")

    args = parser.parse_args()

    if args.smoke_test:
        cfg_path = SMOKE_FILE
    elif args.full_experiment:
        cfg_path = FULL_FILE
    else:  # This branch should be unreachable due to mutually-exclusive group.
        parser.error("Must specify either --smoke-test or --full-experiment")
        sys.exit(1)

    LOGGER.info("Selected configuration: %s", cfg_path)

    # Immediately inform the user that execution will fail because the core
    # experiment code is missing, but do so in a controlled and explicit
    # manner rather than letting obscure errors surface later.
    try:
        run_phase(cfg_path)
    except RuntimeError as err:
        LOGGER.critical(str(err))
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
