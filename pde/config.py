"""Load config.yaml and resolve paths relative to the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def p(rel):
    """Project-relative path -> absolute Path."""
    rel = Path(rel)
    return rel if rel.is_absolute() else ROOT / rel
