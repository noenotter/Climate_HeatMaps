from __future__ import annotations
from pathlib import Path
import os
import yaml

def project_root() -> Path:
    """
    Returns the project root (folder that contains this 'lib' dir).
    Works from app.py, notebooks, or scripts in subfolders.
    """
    return Path(__file__).resolve().parents[1]

def load_config() -> dict:
    cfg_path = project_root() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()

# Allow environment overrides (optional)
DATA_ROOT = Path(os.getenv("DATA_ROOT", CFG.get("data_root", ".")))
OUT_ROOT  = Path(os.getenv("OUTPUTS_ROOT", CFG.get("outputs_root", "outputs")))

# Resolve relative to project root if given as relative path
if not DATA_ROOT.is_absolute():
    DATA = project_root() / DATA_ROOT
else:
    DATA = DATA_ROOT

if not OUT_ROOT.is_absolute():
    OUT = project_root() / OUT_ROOT
else:
    OUT = OUT_ROOT

def in_data(rel: str | Path) -> Path:
    """Path to an input inside data_root (here: project root)."""
    return (DATA / rel)

def in_out(rel: str | Path) -> Path:
    """Path to an output inside outputs_root."""
    return (OUT / rel)

def ensure_out_dirs():
    """Create common output subfolders if missing."""
    for sub in ("pdf", "png", "csv"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
