"""DCT-domain QIM embedding and extraction.

Replaces ONLY the "Simple embedding" and "Extract embedded bits" boxes of the flow
diagram. Everything before them (chaotic map, CNN, SE, CCNOT, DNA, QEC) is unchanged.

How it works
  * The 512x512 carrier is cut into 4096 blocks of 8x8 and every block gets a DCT.
    This is the same transform JPEG uses, so JPEG's quantisation steps apply directly.
  * One bit is hidden in one low-frequency DCT coefficient with quantisation index
    modulation (QIM): the coefficient is moved onto one of two interleaved grids
    (bit 0 = multiples of the step, bit 1 = multiples of the step shifted by step/2).
  * Region scorer -> step size: blocks in detailed regions get a larger step (bolder,
    more robust, hidden by texture); blocks in smooth regions get a smaller one (gentle).
  * Every stream bit can be written `repeat` times into different blocks. The extractor
    adds up the soft decisions, which is a majority vote that also uses confidence.

Run `python dct_embed.py` for a quick self-test on a synthetic image (no dataset needed).
"""
import numpy as np

from embed_common import ranked_regions, ROWS, COLS

BLOCK = 8

# AC coefficients in zigzag order (row = vertical frequency, col = horizontal frequency),
# lowest frequency first. Low frequencies have the smallest JPEG quantisation steps,
# so they survive compression best.
COEF_POSITIONS = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)]

# Step multipliers for smooth / medium / detailed regions.
LEVEL_SCALE = (0.75, 1.0, 1.25)


# ---------------- block DCT (orthonormal, identical to the JPEG DCT) ----------------
def _dct_matrix(n=BLOCK):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    d = np.sqrt(2.0 / n) * np.cos((2 * i + 1) * k * np.pi / (2 * n))
    d[0, :] = np.sqrt(1.0 / n)
    return d


_D = _dct_matrix()


def block_dct(img):
    """(H, W) image -> (H/8, W/8, 8, 8) DCT coefficients."""
    h, w = img.shape
    b = img.astype(np.float64).reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).transpose(0, 2, 1, 3)
    return _D @ b @ _D.T


def block_idct(coef):
    nbh, nbw = coef.shape[:2]
    b = _D.T @ coef @ _D
    return b.transpose(0, 2, 1, 3).reshape(nbh * BLOCK, nbw * BLOCK)


# ---------------- region scorer -> step size per block ----------------
def step_map(image_u8, base, adaptive=True):
    """QIM step for every 8x8 block, shape (H/8, W/8).

    adaptive=True : the 10x10 region scorer ranks the regions; the top third gets
                    base*1.25, the middle third base*1.0, the bottom third base*0.75.
    adaptive=False: every block uses `base`.
    """
    h, w = image_u8.shape
    nbh, nbw = h // BLOCK, w // BLOCK
    if not adaptive:
        return np.full((nbh, nbw), float(base))

    ys = np.linspace(0, h, ROWS + 1, dtype=int)
    xs = np.linspace(0, w, COLS + 1, dtype=int)
    cy = np.arange(nbh) * BLOCK + BLOCK // 2
    cx = np.arange(nbw) * BLOCK + BLOCK // 2
    ry = np.clip(np.searchsorted(ys, cy, side="right") - 1, 0, ROWS - 1)
    rx = np.clip(np.searchsorted(xs, cx, side="right") - 1, 0, COLS - 1)

    # Score every region on the 64x64 image of block means (the DC coefficients):
    # variance + edge strength, like the original scorer. Embedding only changes AC
    # coefficients, so the block means (and the ranking) stay the same after embedding
    # and under mild attacks. That is what lets blind extraction find the same steps.
    means = image_u8.astype(np.float64).reshape(nbh, BLOCK, nbw, BLOCK).mean(axis=(1, 3)) / 255.0
    gy, gx = np.gradient(means)
    grad = np.abs(gy) + np.abs(gx)
    rid = (ry[:, None] * COLS + rx[None, :]).ravel()
    n_reg = ROWS * COLS
    cnt = np.maximum(np.bincount(rid, minlength=n_reg), 1)
    m1 = np.bincount(rid, weights=means.ravel(), minlength=n_reg) / cnt
    m2 = np.bincount(rid, weights=(means ** 2).ravel(), minlength=n_reg) / cnt
    variance = np.maximum(m2 - m1 ** 2, 0.0)
    edge = np.bincount(rid, weights=grad.ravel(), minlength=n_reg) / cnt
    score = variance / (variance.max() + 1e-12) + edge / (edge.max() + 1e-12)

    order = np.argsort(-score, kind="stable")        # highest score first
    level = np.empty(n_reg, dtype=np.int64)
    for rank, region in enumerate(order):
        level[region] = 2 if rank < n_reg // 3 else (1 if rank < 2 * n_reg // 3 else 0)
    level = level.reshape(ROWS, COLS)
    return float(base) * np.asarray(LEVEL_SCALE)[level[np.ix_(ry, rx)]]


# ---------------- where every bit goes ----------------
def slot_layout(n_bits, repeat, nbh, nbw):
    """Slot j lives in block (j % nblocks), coefficient (j // nblocks) and carries
    stream bit (j % n_bits). Repeat r of bit i is therefore slot i + r * n_bits."""
    nb = nbh * nbw
    total = n_bits * repeat
    if total > nb * len(COEF_POSITIONS):
        raise ValueError(
            f"{total} slots needed but only {nb * len(COEF_POSITIONS)} exist "
            f"({nb} blocks x {len(COEF_POSITIONS)} coefficients). Lower the repeat count."
        )
    j = np.arange(total)
    block = j % nb
    return block // nbw, block % nbw, j // nb, j % n_bits


# ---------------- QIM ----------------
def qim_embed_values(c, bits, step):
    d = bits * (step / 2.0)
    return step * np.round((c - d) / step) + d


def qim_soft(c, step):
    """Positive = bit 1 is more likely. Range -0.5 .. +0.5 (in units of step)."""
    e0 = np.abs(c - step * np.round(c / step))
    e1 = np.abs(c - (step * np.round((c - step / 2.0) / step) + step / 2.0))
    return (e0 - e1) / step


def extract_stream(received_u8, n_bits, repeat, steps):
    """Returns (bits, soft_sums). `steps` is the per-block step map."""
    coef = block_dct(received_u8)
    br, bc, ci, si = slot_layout(n_bits, repeat, coef.shape[0], coef.shape[1])
    pos = np.asarray(COEF_POSITIONS)
    vals = coef[br, bc, pos[ci, 0], pos[ci, 1]]
    soft = qim_soft(vals, steps[br, bc])
    acc = np.bincount(si, weights=soft, minlength=n_bits)
    return (acc > 0).astype(np.uint8), acc


def embed_stream(carrier_u8, stream, repeat, steps, max_iter=8):
    """Embed `stream` (0/1 uint8 array). Returns (watermarked_uint8, passes, residual_errors).

    Rounding to 0..255 (and clipping in pure white/black areas) slightly disturbs the
    coefficients, so the result is checked and, if needed, re-quantised a few times.
    residual_errors > 0 means some bits still read back wrong with no attack at all.
    """
    n_bits = int(stream.size)
    nbh, nbw = carrier_u8.shape[0] // BLOCK, carrier_u8.shape[1] // BLOCK
    br, bc, ci, si = slot_layout(n_bits, repeat, nbh, nbw)
    pos = np.asarray(COEF_POSITIONS)
    rows, cols = pos[ci, 0], pos[ci, 1]
    step_vec = steps[br, bc]
    bit_vec = stream[si].astype(np.float64)

    work = carrier_u8
    out = carrier_u8
    errors = n_bits
    for it in range(1, max_iter + 1):
        coef = block_dct(work)
        coef[br, bc, rows, cols] = qim_embed_values(coef[br, bc, rows, cols], bit_vec, step_vec)
        out = np.clip(np.rint(block_idct(coef)), 0, 255).astype(np.uint8)
        rx, _ = extract_stream(out, n_bits, repeat, steps)
        errors = int(np.sum(rx != stream))
        if errors == 0:
            return out, it, 0
        work = out
    return out, max_iter, errors


# ---------------- quick self-test ----------------
if __name__ == "__main__":
    import cv2

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:512, 0:512]
    img = 40 + 0.2 * xx + 0.12 * yy + 20 * np.sin(xx / 17.0) * np.cos(yy / 23.0) + rng.normal(0, 4, (512, 512))
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    n_bits, repeat = 4294, 3
    bits = rng.integers(0, 2, n_bits).astype(np.uint8)
    steps = np.full((64, 64), 24.0)

    marked, passes, errs = embed_stream(img, bits, repeat, steps)
    mse = float(np.mean((img.astype(np.float64) - marked) ** 2))
    print(f"embedded {n_bits} bits x{repeat}: passes={passes}, residual errors={errs}, "
          f"PSNR={10 * np.log10(255.0 ** 2 / mse):.2f} dB")

    rx, _ = extract_stream(marked, n_bits, repeat, steps)
    print("no attack      BER:", float(np.mean(rx != bits)), "(must be 0.0)")

    for q in (90, 70, 50):
        ok, buf = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, q])
        jp = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        rx, _ = extract_stream(jp, n_bits, repeat, steps)
        print(f"JPEG {q}        BER:", round(float(np.mean(rx != bits)), 4))

    sa = step_map(img, 24.0, adaptive=True)
    print("adaptive step map: min", sa.min(), "max", sa.max(), "shape", sa.shape)