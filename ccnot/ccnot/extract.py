"""Extraction side. Run it from the same folder as main_ccnot.py:
    python extract.py

Order follows the extraction half of the flow diagram:
  received image -> region scorer (step sizes) -> DCT-QIM read -> split the bit stream
  payload path   : QEC vote -> reverse CCNOT -> DNA / angle decode (8 values + 32-bit hash)
  watermark path : descramble with the key -> recovered watermark
  integrity check: hash test (exact) + CNN signature test
"""
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Same sys.path handling as main_ccnot.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in [str(SCRIPT_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import KEY, LOGISTIC_KEY, MAP_CHOICE as CONFIG_MAP, find_images_dir
from embed_common import (allocate_positions, decode_payload, watermark_hash_bits,
                          logistic_descramble, DNA_MAP, TOP_K)
from chaotic_lscm import chaotic_descramble

# ---------------- settings you can change ----------------
# None = read the saved watermarked.png as it is.
# 90 / 70 / 50 = JPEG-compress it first (a quick robustness demo).
JPEG_QUALITY = None
# True  = use the step sizes saved at embedding (side mode, most reliable).
# False = recompute them from the received image (blind mode).
USE_SAVED_STEPS = True
# ---------------------------------------------------------

IMAGES_DIR = find_images_dir()
print("Using images folder:", IMAGES_DIR)


def ber(a, b):
    return float(np.mean(a != b))


def ncc(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom else 0.0


# ---------- 1. load watermarked image and saved info ----------
wm_img = cv2.imread(str(IMAGES_DIR / "watermarked.png"), cv2.IMREAD_GRAYSCALE)
if wm_img is None:
    raise SystemExit(f"watermarked.png not found in {IMAGES_DIR}. Run main_ccnot.py first.")

if JPEG_QUALITY is not None:
    ok, buf = cv2.imencode(".jpg", wm_img, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
    wm_img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    print(f"Attack: JPEG quality {JPEG_QUALITY} applied before extraction")

meta_path = IMAGES_DIR / "embed_meta.npz"
if not meta_path.exists():
    raise SystemExit(f"embed_meta.npz not found in {IMAGES_DIR}. Run main_ccnot.py first.")
meta = np.load(meta_path)

pad_len = int(meta["pad_len"])
n_payload = int(meta["n_payload"])
n_wm = int(meta["n_wm"])
n_hash = int(meta["n_hash"]) if "n_hash" in meta.files else 0
method = str(meta["method"]) if "method" in meta.files else "lsb"

# Watermark size comes from the saved bit count
side = int(round(math.sqrt(n_wm)))
assert side * side == n_wm, f"n_wm = {n_wm} is not a square watermark"
print(f"Watermark size from saved info: {side}x{side} ({n_wm} bits)")
print("Embedding method:", method.upper())

map_choice = str(meta["map_choice"]) if "map_choice" in meta.files else CONFIG_MAP
print("Scrambler used at embedding:", map_choice)

# ---------- 2. read the embedded bits ----------
if method == "dct":
    from dct_embed import step_map, extract_stream

    repeat = int(meta["repeat"])
    base_step = float(meta["base_step"])
    adaptive = bool(int(meta["adaptive"]))
    if USE_SAVED_STEPS and "steps" in meta.files:
        steps = meta["steps"]
        print("Step sizes: saved at embedding (side mode)")
    else:
        steps = step_map(wm_img, base_step, adaptive=adaptive)
        print("Step sizes: recomputed from the received image (blind mode)")
    stream, _ = extract_stream(wm_img, n_payload + n_wm, repeat, steps)
    print(f"DCT-QIM read: {n_payload + n_wm} bits, {repeat} copies each (soft vote)")
else:
    assert int(meta["top_k"]) == TOP_K, "top_k in embed_meta.npz differs from embed_common.TOP_K"
    positions = allocate_positions(wm_img, n_payload + n_wm)
    stream = wm_img.flatten()[positions] & 1

stream = np.asarray(stream, dtype=np.uint8)

# ---------- 3. payload: QEC vote -> reverse CCNOT -> DNA / angle decode ----------
print("\n--- Payload path (QEC -> CCNOT -> DNA -> angles) ---")
qec_bits = "".join(map(str, stream[:n_payload]))
payload_all = decode_payload(qec_bits, pad_len)             # majority vote + reverse CCNOT
n_angle_bits = len(payload_all) - n_hash
angle_bits = payload_all[:n_angle_bits]
decoded_hash = payload_all[n_angle_bits:]

payload_values = [int(angle_bits[i:i + 8], 2) for i in range(0, len(angle_bits), 8)]
dna = "".join(DNA_MAP[angle_bits[i:i + 2]] for i in range(0, len(angle_bits), 2))

print("Recovered 8-bit values :", payload_values)
print("DNA sequence           :", dna)
print("Angles (degrees)       :", [round(v / 255 * 180, 1) for v in payload_values])

if "payload_bits" in meta.files:
    ok = payload_all == str(meta["payload_bits"])
    print("Payload matches embedded payload:", "SUCCESS" if ok else "FAILED")

# ---------- 4. watermark path: descramble ----------
print("\n--- Watermark path (descramble with key) ---")
scrambled = stream[n_payload:].reshape(side, side).astype(np.uint8)

if map_choice == "lscm":
    recovered = chaotic_descramble(scrambled, key=KEY)
    wrong = chaotic_descramble(scrambled, key=KEY + "!")
    wrong_label = "key + '!'"
else:
    recovered = logistic_descramble(scrambled, key=LOGISTIC_KEY)
    wrong = logistic_descramble(scrambled, key=LOGISTIC_KEY + 1e-9)
    wrong_label = "key + 1e-9"

if "orig_watermark" in meta.files:
    orig = meta["orig_watermark"].astype(np.uint8)
else:
    raw = cv2.imread(str(IMAGES_DIR / "watermark.png"), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise SystemExit("No orig_watermark in embed_meta.npz and watermark.png not found.")
    raw = cv2.resize(raw, (side, side), interpolation=cv2.INTER_AREA)
    _, orig = cv2.threshold(raw, 127, 1, cv2.THRESH_BINARY)
    orig = orig.astype(np.uint8)

print("=== Correct key ===")
print("BER:", ber(orig, recovered))
print("NCC:", ncc(orig, recovered))
print("Exact match:", np.array_equal(orig, recovered))

print(f"\n=== Wrong key ({wrong_label}) ===")
print("BER:", ber(orig, wrong), "(should be near 0.5)")
print("NCC:", ncc(orig, wrong))

# ---------- 5. integrity checks (neither needs the key) ----------
print("\n--- Integrity check ---")
hash_verdict = None
if n_hash:
    received_hash = watermark_hash_bits(scrambled.flatten(), n_hash)
    hash_verdict = (received_hash == decoded_hash)
    print("Hash stored in payload    :", decoded_hash)
    print("Hash of received bits     :", received_hash)
    print("1. Hash test              :", "MATCH (watermark bits exactly intact)" if hash_verdict
          else "MISMATCH (watermark bits were damaged)")
else:
    print("1. Hash test              : not available (no hash in this embedding)")

cnn_verdict = None
try:
    import torch
    from pipeline_model import build_model, watermark_to_angle_u8

    model = build_model()
    model.load_state_dict(torch.load(IMAGES_DIR / "pipeline_weights.pt", map_location="cpu"))
    model.eval()

    signature = [int(v) for v in watermark_to_angle_u8(scrambled, model)]
    diffs = [abs(a - b) for a, b in zip(signature, payload_values)]
    cnn_verdict = (max(diffs) == 0)
    print("CNN signature of received :", signature)
    print("Decoded payload values    :", payload_values)
    print("2. CNN test               :", "EXACT MATCH" if cnn_verdict
          else f"DIFFERS (max difference {max(diffs)})")
    print("   (the CNN signature is weak for small damage; trust the hash test first)")
except Exception as e:
    print(f"2. CNN test               : skipped ({type(e).__name__}: {e})")

if hash_verdict is not None:
    print("\nVERDICT:", "INTACT" if hash_verdict else "DAMAGED (BER above shows how much)")

# ---------- 6. save and show ----------
out_path = IMAGES_DIR / "recovered_watermark.png"
cv2.imwrite(str(out_path), (recovered * 255).astype(np.uint8))
print("\nRecovered watermark saved to:", out_path)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(ax, [orig, recovered, wrong],
                    ["Original", "Recovered (correct key)", "Recovered (wrong key)"]):
    a.imshow(im, cmap="gray")
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
plt.show()