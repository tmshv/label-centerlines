import logging

from .centerline import get_centerline
from .exceptions import CenterlineError

__all__ = ["get_centerline", "CenterlineError", "__version__"]

__version__ = "2026.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
