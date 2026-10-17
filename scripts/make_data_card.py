"""Regenerate docs/DATA_CARD.md from a released dataset (SIMULATED DATA). Run: make data-card"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vsensor.data.card import write_card  # noqa: E402
from vsensor.data.config import load_dataset_config  # noqa: E402
from vsensor.params import load_registry  # noqa: E402

cfg = load_dataset_config()
print(write_card(cfg, load_registry(), ROOT / "data" / cfg.name))
