import logging
from pathlib import Path

version_path = Path(__file__).parent / "version.txt"

if version_path.is_file():
    with open(version_path) as version_file:
        __version__ = version_file.read().strip()

del Path, version_path

logger = logging.getLogger("celebtwin")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
del logging


def preload() -> None:
    """Eagerly import all modules and subpackages."""
    import importlib
    import pkgutil

    for module_info in pkgutil.walk_packages(__path__, __name__ + "."):
        if not module_info.name.endswith('.__main__'):
            importlib.import_module(module_info.name)
