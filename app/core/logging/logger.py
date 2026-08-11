from pathlib import Path
import sys

from loguru import logger

from app.core.config.settings import settings

# ---------------------------------------------------------
# Create logs directory
# ---------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Remove default logger
# ---------------------------------------------------------

logger.remove()

# ---------------------------------------------------------
# Console Logger
# ---------------------------------------------------------

logger.add(
    sys.stdout,
    level=settings.log_level,
    colorize=True,
    backtrace=True,
    diagnose=True,
    enqueue=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:"
        "<cyan>{function}</cyan>:"
        "<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)

# ---------------------------------------------------------
# Application Log
# ---------------------------------------------------------

logger.add(
    LOG_DIR / "application.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=False,
)

# ---------------------------------------------------------
# Error Log
# ---------------------------------------------------------

logger.add(
    LOG_DIR / "error.log",
    rotation="00:00",
    retention="60 days",
    compression="zip",
    level="ERROR",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

__all__ = ["logger"]