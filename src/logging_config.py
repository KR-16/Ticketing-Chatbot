"""One logging setup shared by the training pipeline, CLI, and API."""

import logging
import os

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: str = None):
    """Configure root logging once; level comes from the LOG_LEVEL env var
    (default INFO) unless passed explicitly."""
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=resolved, format=LOG_FORMAT)
