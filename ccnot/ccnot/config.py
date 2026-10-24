"""Shared settings for embedding (main_ccnot.py), extraction (extract.py) and the
dataset runs (run_dataset.py).

Change the key, the map, the watermark size or the DCT step here ONLY, so every
script agrees.
"""
from pathlib import Path

# Passphrase for the 2D-LSCM scrambler
KEY = "your-passphrase-here"

# Initial value for the old logistic baseline (must be between 0 and 1)
LOGISTIC_KEY = 0.7345

# "lscm" = new 2D map, "logistic" = old baseline
MAP_CHOICE = "lscm"

# Watermark side length in pixels: 64 -> 4096 bits (team decision).
WM_SIZE = 64

# DCT-QIM base step. 32 survived JPEG 90/70/50 with BER 0 on the 17 SIPI images.
DCT_BASE_STEP = 32.0


def default_repeat(wm_size):
    """How many times each embedded bit is stored in the DCT method.
    64x64 has room for 3 copies; 128x128 only fits one copy."""
    return 3 if wm_size <= 64 else 1


def find_images_dir():
    """Same search order as main_ccnot.py:
    1) images beside the script  2) images in the current folder
    3) images in the parent folder  4) fall back to images beside the script
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if (script_dir / "images").is_dir():
        return script_dir / "images"
    if (Path.cwd() / "images").is_dir():
        return Path.cwd() / "images"
    if (project_root / "images").is_dir():
        return project_root / "images"
    return script_dir / "images"