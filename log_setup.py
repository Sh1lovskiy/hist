# log_setup.py
from loguru import logger
import sys


def setup_logger() -> None:
    """Configure loguru sink and format."""
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{message}"
    )
    logger.add(sys.stderr, format=fmt, level="INFO", diagnose=False)
