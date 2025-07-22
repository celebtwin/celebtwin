from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

__version__: str
logger: 'logging.Logger'


def _initialize() -> None:
    """Initialize module-level variables."""
    import importlib.metadata
    import logging

    global __version__, logger
    __version__ = importlib.metadata.version(__name__)
    logger = logging.getLogger("celebtwin")
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

_initialize()


def preload() -> None:
    """Eagerly import all modules and subpackages."""
    import importlib
    import pkgutil

    for module_info in pkgutil.walk_packages(__path__, __name__ + "."):
        if not module_info.name.endswith('.__main__'):
            importlib.import_module(module_info.name)
