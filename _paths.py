"""Load the released source trees before unrelated installed research forks."""
from pathlib import Path
import sys


def configure_vendor_paths():
    root = Path(__file__).resolve().parent
    for relative in ("vendor/dppo", "vendor/stable-baselines3"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)
