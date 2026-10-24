"""Compare the old logistic scrambler with the 2D-LSCM scrambler on the same watermark.

    python compare_maps.py                 # 64x64 watermark, 20 random keys
    python compare_maps.py --size 128      # same test at 128x128
    python compare_maps.py --keys 50 --plot

Prints three things:
  1. Statistics of the scrambled watermark (balance, entropy, neighbour correlation)
  2. Keystream randomness over many keys (monobit and runs tests, 2 of the NIST tests)
  3. Key sensitivity: how tiny a key change still destroys the watermark
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in [str(SCRIPT_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import KEY, LOGISTIC_KEY, find_images_dir
import chaotic_lscm as lscm   # private helpers are used for the key-sensitivity test

IMAGES_DIR = find_images_dir()
DELTAS = [1e-4, 1e-8, 1e-10, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
RANGES = {"logistic x0": 0.9, "lscm x0": 0.9, "lscm y0": 0.9, "lscm theta": 0.55}


# ---------------- scramblers ----------------
def logistic_keystream(n, key):
    """Same as the logistic baseline in main_ccnot.py (r = 3.999)."""
    x = key
    seq = np.empty(n)
    for i in range(n):
        x = 3.999 * x * (1 - x)
        seq[i] = x
    return np.argsort(seq), (seq > 0.5).astype(np.uint8)


def scramble(flat, perm, bits):
    return np.bitwise_xor(flat[perm], bits)


def descramble(scrambled_flat, perm, bits):
    unxored = np.bitwise_xor(scrambled_flat, bits)
    out = np.empty_like(unxored)
    out[perm] = unxored
    return out


# ---------------- statistics ----------------
def ones_fraction(arr):
    return float(np.mean(arr))


def entropy_bits(arr):
    p = float(np.mean(arr))
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def adjacent_correlation(arr):
    a = arr.astype(np.float64)
    pairs = {
        "H": (a[:, :-1], a[:, 1:]),
        "V": (a[:-1, :], a[1:, :]),
        "D": (a[:-1, :-1], a[1:, 1:]),
    }
    out = {}
    for k, (x, y) in pairs.items():
        x, y = x.ravel(), y.ravel()
        out[k] = float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])
    return out


def monobit_p(bits):
    n = len(bits)
    s = abs(2 * int(bits.sum()) - n) / math.sqrt(n)
    return math.erfc(s / math.sqrt(2))


def runs_p(bits):
    n = len(bits)
    pi = float(bits.mean())
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0
    v = 1 + int(np.sum(bits[1:] != bits[:-1]))
    return math.erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, help="watermark side length (64 or 128)")
    ap.add_argument("--watermark", default=str(IMAGES_DIR / "watermark.png"))
    ap.add_argument("--keys", type=int, default=20, help="number of random keys to test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    raw = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise SystemExit(f"Watermark not found: {args.watermark}")
    raw = cv2.resize(raw, (args.size, args.size), interpolation=cv2.INTER_AREA)
    _, wm = cv2.threshold(raw, 127, 1, cv2.THRESH_BINARY)
    wm = wm.astype(np.uint8)
    flat = wm.flatten()
    n = flat.size

    print(f"Watermark: {args.size}x{args.size} = {n} bits | random keys: {args.keys}\n")

    # ---------- 1. scrambled-watermark statistics (keys from config.py) ----------
    perm_l, bits_l = logistic_keystream(n, LOGISTIC_KEY)
    perm_s, bits_s = lscm.generate_keystream(n, KEY)
    scr_l = scramble(flat, perm_l, bits_l)
    scr_s = scramble(flat, perm_s, bits_s)
    assert np.array_equal(descramble(scr_l, perm_l, bits_l), flat), "logistic round-trip failed"
    assert np.array_equal(descramble(scr_s, perm_s, bits_s), flat), "lscm round-trip failed"

    print("=" * 74)
    print("1. SCRAMBLED WATERMARK STATISTICS  (ideal: ones 0.5, entropy 1.0, correlation 0)")
    print("=" * 74)
    print(f"{'':<18}{'ones':>8}{'entropy':>10}{'corr H':>9}{'corr V':>9}{'corr D':>9}")
    for label, arr in (("original", wm),
                       ("logistic scrambled", scr_l.reshape(wm.shape)),
                       ("LSCM scrambled", scr_s.reshape(wm.shape))):
        c = adjacent_correlation(arr)
        print(f"{label:<18}{ones_fraction(arr):8.4f}{entropy_bits(arr):10.4f}"
              f"{c['H']:9.4f}{c['V']:9.4f}{c['D']:9.4f}")

    # ---------- 2. keystream randomness over many keys ----------
    rng = np.random.default_rng(args.seed)
    log_keys = rng.uniform(0.05, 0.95, args.keys)
    lscm_keys = [f"test-key-{i}" for i in range(args.keys)]

    stats = {"logistic": {"ones": [], "mono": 0, "runs": 0},
             "LSCM": {"ones": [], "mono": 0, "runs": 0}}
    for k in log_keys:
        _, b = logistic_keystream(n, float(k))
        s = stats["logistic"]
        s["ones"].append(b.mean())
        s["mono"] += monobit_p(b) >= 0.01
        s["runs"] += runs_p(b) >= 0.01
    for k in lscm_keys:
        _, b = lscm.generate_keystream(n, k)
        s = stats["LSCM"]
        s["ones"].append(b.mean())
        s["mono"] += monobit_p(b) >= 0.01
        s["runs"] += runs_p(b) >= 0.01

    print("\n" + "=" * 74)
    print("2. KEYSTREAM RANDOMNESS  (XOR bits, pass = p-value >= 0.01)")
    print("=" * 74)
    print(f"{'':<12}{'mean ones':>11}{'monobit pass':>15}{'runs pass':>13}")
    for name, s in stats.items():
        print(f"{name:<12}{np.mean(s['ones']):11.4f}{s['mono']:>11}/{args.keys:<3}"
              f"{s['runs']:>9}/{args.keys:<3}")
    print("(only 2 of the NIST SP 800-22 tests; do not describe this as passing the NIST suite)")

    # ---------- 3. key sensitivity ----------
    cols = list(RANGES)
    table = {d: {c: [] for c in cols} for d in DELTAS}

    for i in range(args.keys):
        # logistic: one secret value x0
        x0 = float(log_keys[i])
        p_true, b_true = logistic_keystream(n, x0)
        scr = scramble(flat, p_true, b_true)
        for d in DELTAS:
            x0b = x0 + d if x0 + d < 1 else x0 - d
            p2, b2 = logistic_keystream(n, x0b)
            table[d]["logistic x0"].append(float(np.mean(descramble(scr, p2, b2) != flat)))

        # LSCM: three secret values derived from the passphrase
        x, y, t, _ = lscm._derive_params(lscm_keys[i])
        p_true, b_true, _, _ = lscm._keystream_from_params(n, x, y, t)
        scr = scramble(flat, p_true, b_true)
        for d in DELTAS:
            for col, args_ in (("lscm x0", (x + d, y, t)),
                               ("lscm y0", (x, y + d, t)),
                               ("lscm theta", (x, y, t + d))):
                p2, b2, _, _ = lscm._keystream_from_params(n, *args_)
                table[d][col].append(float(np.mean(descramble(scr, p2, b2) != flat)))

    print("\n" + "=" * 74)
    print("3. KEY SENSITIVITY  (one secret value changed by delta; BER 0.5 = watermark destroyed)")
    print("=" * 74)
    print(f"{'delta':>9}" + "".join(f"{c:>14}" for c in cols))
    for d in DELTAS:
        print(f"{d:9.0e}" + "".join(f"{np.mean(table[d][c]):14.4f}" for c in cols))

    limits, key_space = {}, {}
    for c in cols:
        hit = [d for d in DELTAS if np.mean(table[d][c]) >= 0.45]
        limits[c] = min(hit) if hit else None
    print("\nSmallest delta that still destroys the watermark (BER >= 0.45):")
    for c in cols:
        print(f"  {c:<12}: {limits[c]:.0e}" if limits[c] else f"  {c:<12}: not reached down to 1e-16")

    def log10_space(names):
        if any(limits[nm] is None for nm in names):
            return None
        return sum(math.log10(RANGES[nm] / limits[nm]) for nm in names)

    ls = log10_space(["logistic x0"])
    ss = log10_space(["lscm x0", "lscm y0", "lscm theta"])
    print("\nRough key-space estimate from those limits (decade resolution only):")
    print(f"  logistic : about 10^{ls:.0f}" if ls is not None else "  logistic : n/a")
    print(f"  LSCM     : about 10^{ss:.0f}  (x0 * y0 * theta; the passphrase is hashed into these)"
          if ss is not None else "  LSCM     : n/a")

    # ---------- plaintext sensitivity (why NPCR/UACI do not apply) ----------
    flipped = flat.copy()
    flipped[0] ^= 1
    print("\nPlaintext sensitivity: flip 1 watermark bit ->",
          "logistic output bits changed:",
          int(np.sum(scramble(flipped, perm_l, bits_l) != scr_l)),
          "| LSCM output bits changed:",
          int(np.sum(scramble(flipped, perm_s, bits_s) != scr_s)),
          "\n  (the keystream depends only on the key, so NPCR/UACI are not meaningful here)")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a, im, t in zip(ax, [wm, scr_l.reshape(wm.shape), scr_s.reshape(wm.shape)],
                            ["Original", "Logistic scrambled", "LSCM scrambled"]):
            a.imshow(im, cmap="gray")
            a.set_title(t)
            a.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()