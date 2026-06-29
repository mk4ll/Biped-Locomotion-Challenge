"""Load the single source of parameters (config/params.yaml)."""
from pathlib import Path
import yaml

# Project root = two levels up from this file (src/utils/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = PROJECT_ROOT / "config" / "params.yaml"


def load_params(path: str | Path = DEFAULT_PARAMS) -> dict:
    """Read params.yaml into a dict. All tunables live here, never hardcoded."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str | Path) -> Path:
    """Resolve a path from params.yaml relative to the project root."""
    p = Path(rel_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
