from pathlib import Path

version_path = Path(__file__).parent / "version.txt"

if version_path.is_file():
    with open(version_path) as version_file:
        __version__ = version_file.read().strip()

del Path, version_path
