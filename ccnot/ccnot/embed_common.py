"""Shared code for embedding and extraction.
Both main_ccnot.py and extract.py import from here, so they cannot drift apart.
"""
import hashlib

import cv2
import numpy as np

ROWS, COLS, TOP_K = 10, 10, 50
DNA_MAP = {"00": "A", "01": "C", "10": "G", "11": "T"}
REV_DNA = {v: k for k, v in DNA_MAP.items()}


# ---------- region scoring (LSB-blind, so it is reproducible after embedding) ----------
def ranked_regions(image_u8, rows=ROWS, cols=COLS):
    img = image_u8 & 0xFE                      # ignore the LSB plane
    norm = img.astype(np.float32) / 255.0
    h, w = img.shape
    ys = np.linspace(0, h, rows + 1, dtype=int)
    xs = np.linspace(0, w, cols + 1, dtype=int)
    out = []
    for r in range(rows):
        for c in range(cols):
            y1, y2, x1, x2 = int(ys[r]), int(ys[r + 1]), int(xs[c]), int(xs[c + 1])
            var = float(np.var(norm[y1:y2, x1:x2]))
            edge = float(np.mean(cv2.Canny(img[y1:y2, x1:x2], 50, 150) > 0))
            out.append(dict(row=r, col=c, x=x1, y=y1, width=x2 - x1,
                            height=y2 - y1, score=var + edge))
    # deterministic order, ties broken by position
    return sorted(out, key=lambda d: (-d["score"], d["row"], d["col"]))


def allocate_positions(image_u8, n_bits, top_k=TOP_K):
    """Flat pixel indices for n_bits, in rank order. Detailed regions get more bits,
    the smoothest (100 - top_k) regions get none."""
    ranked = ranked_regions(image_u8)[:top_k]
    w = np.array([d["score"] for d in ranked])
    quota = np.floor(n_bits * w / w.sum()).astype(int)
    quota[: n_bits - quota.sum()] += 1         # leftover goes to top-ranked regions

    W = image_u8.shape[1]
    parts = []
    for d, q in zip(ranked, quota):
        n = d["width"] * d["height"]
        assert q <= n, "region quota exceeds region size; raise top_k"
        if q == 0:
            continue
        local = np.linspace(0, n - 1, int(q), dtype=np.int64)
        ly, lx = local // d["width"], local % d["width"]
        parts.append((d["y"] + ly) * W + (d["x"] + lx))
    return np.concatenate(parts)


# ---------- payload: angle bits -> CCNOT -> DNA -> QEC ----------
def ccnot_classical(bits):
    """(a, b, c) -> (a, b, c XOR (a AND b)) on each triple. Self-inverse."""
    b = [int(x) for x in bits]
    assert len(b) % 3 == 0
    for i in range(0, len(b), 3):
        b[i + 2] ^= b[i] & b[i + 1]
    return "".join(map(str, b))


def watermark_hash_bits(wm_bits, n_bits=32):
    """Short fingerprint (SHA-256, first n_bits) of the scrambled watermark bits.
    Change even ONE watermark bit and this comes out completely different."""
    packed = np.packbits(np.asarray(wm_bits, dtype=np.uint8).flatten()).tobytes()
    digest = hashlib.sha256(packed).digest()
    return "".join(format(b, "08b") for b in digest[: n_bits // 8])


def build_payload(angle_u8, extra_bits=""):
    """angle values (8 x 8 bits) [+ extra_bits, e.g. the 32-bit hash] -> CCNOT -> DNA -> QEC."""
    bits = "".join(format(int(v), "08b") for v in angle_u8) + extra_bits
    pad_len = (-len(bits)) % 6                 # multiple of 3 (CCNOT) and 2 (DNA)
    mixed = ccnot_classical(bits + "0" * pad_len)
    dna = "".join(DNA_MAP[mixed[i:i + 2]] for i in range(0, len(mixed), 2))
    qec = "".join(b * 3 for b in mixed)
    return dict(bits=bits, pad_len=pad_len, ccnot_bits=mixed, dna=dna, qec_bits=qec)


def decode_payload(qec_bits, pad_len):
    voted = "".join("1" if qec_bits[i:i + 3].count("1") >= 2 else "0"
                    for i in range(0, len(qec_bits), 3))
    bits = ccnot_classical(voted)              # same gate undoes itself
    return bits[:len(bits) - pad_len] if pad_len else bits


# ---------- inverse of the old logistic baseline scrambler ----------
def logistic_descramble(scrambled_image, key=0.7345):
    """Exact inverse of chaotic_scramble() in main_ccnot.py (logistic map, r = 3.999)."""
    if not 0 < key < 1:
        raise ValueError("Chaotic key must be between 0 and 1.")
    shape = np.asarray(scrambled_image).shape
    s = np.asarray(scrambled_image).flatten().astype(np.uint8)

    x = key
    seq = np.empty(len(s))
    for i in range(len(s)):
        x = 3.999 * x * (1 - x)
        seq[i] = x

    permutation = np.argsort(seq)
    bits = (seq > 0.5).astype(np.uint8)

    unxored = np.bitwise_xor(s, bits)
    original = np.empty_like(unxored)
    original[permutation] = unxored
    return original.reshape(shape)