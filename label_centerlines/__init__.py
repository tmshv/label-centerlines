import logging

from ._src import get_centerline


__version__ = "2026.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
