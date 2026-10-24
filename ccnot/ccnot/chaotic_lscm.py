"""
Drop-in replacement for Step 10 (CHAOTIC SCRAMBLING) of main_ccnot.py.

Generator : 2D Logistic-Sine Coupling Map (2D-LSCM, Hua et al., 2018)

    x[n+1] = sin( pi * ( 4*t*x[n]*(1-x[n])  + (1-t)*sin(pi*y[n])   ) )
    y[n+1] = sin( pi * ( 4*t*y[n]*(1-y[n])  + (1-t)*sin(pi*x[n+1]) ) )

Return signature is unchanged:  (scrambled, permutation, chaotic_bits)
A matching chaotic_descramble() is included for the extraction step.
"""

import hashlib
import math
import numpy as np

BURN_IN = 1000
THETA_MIN = 0.45
THETA_MAX = 1.00
LE_MIN = 0.5
LE_STEPS = 2000
MAX_REDERIVE = 50


def _step(x, y, t):
    x = math.sin(math.pi * (4 * t * x * (1 - x) + (1 - t) * math.sin(math.pi * y)))
    y = math.sin(math.pi * (4 * t * y * (1 - y) + (1 - t) * math.sin(math.pi * x)))
    return x, y


def _largest_lyapunov(x, y, t, n=LE_STEPS, burn=300):
    """Largest Lyapunov exponent using the analytic Jacobian."""
    pi = math.pi
    for _ in range(burn):
        x, y = _step(x, y, t)

    w0, w1 = 1.0, 0.0
    total = 0.0
    for _ in range(n):
        A = 4 * t * x * (1 - x) + (1 - t) * math.sin(pi * y)
        cA = math.cos(pi * A)
        a = pi * cA * 4 * t * (1 - 2 * x)
        b = pi * cA * (1 - t) * pi * math.cos(pi * y)

        x2 = math.sin(pi * A)
        B = 4 * t * y * (1 - y) + (1 - t) * math.sin(pi * x2)
        cB = pi * math.cos(pi * B)
        dy_dy = cB * 4 * t * (1 - 2 * y)
        dy_dx2 = cB * (1 - t) * pi * math.cos(pi * x2)

        c = dy_dx2 * a
        d = dy_dy + dy_dx2 * b

        w0, w1 = a * w0 + b * w1, c * w0 + d * w1
        norm = math.hypot(w0, w1)
        if norm == 0.0:
            return -math.inf
        total += math.log(norm)
        w0, w1 = w0 / norm, w1 / norm
        x, y = x2, math.sin(pi * B)
    return total / n


def _derive_params(key):
    """Hash the key to (x0, y0, theta); re-derive if not chaotic."""
    key_text = repr(key) if isinstance(key, float) else str(key)

    for counter in range(MAX_REDERIVE):
        h = hashlib.sha256(f"{key_text}|{counter}".encode()).digest()
        u = [int.from_bytes(h[i:i + 8], "big") / 2 ** 64 for i in (0, 8, 16)]
        x0 = 0.05 + 0.90 * u[0]
        y0 = 0.05 + 0.90 * u[1]
        theta = THETA_MIN + (THETA_MAX - THETA_MIN) * u[2]

        if _largest_lyapunov(x0, y0, theta) >= LE_MIN:
            return x0, y0, theta, counter

    raise RuntimeError("Could not derive chaotic parameters from this key.")


def _keystream_from_params(n, x0, y0, theta):
    x, y = x0, y0
    for _ in range(BURN_IN):
        x, y = _step(x, y, theta)

    xs = np.empty(n)
    ys = np.empty(n)
    for i in range(n):
        x, y = _step(x, y, theta)
        xs[i] = x
        ys[i] = y

    permutation = np.argsort(xs, kind="stable")
    bits = (np.floor(ys * 2 ** 32).astype(np.int64) & 1).astype(np.uint8)
    return permutation, bits, xs, ys


def generate_keystream(n, key):
    x0, y0, theta, _ = _derive_params(key)
    permutation, bits, _, _ = _keystream_from_params(n, x0, y0, theta)
    return permutation, bits


def chaotic_scramble(binary_image, key="change-this-passphrase"):
    """Permutation + XOR substitution driven by 2D-LSCM."""
    data = np.asarray(binary_image).flatten().astype(np.uint8)
    permutation, bits = generate_keystream(len(data), key)

    scrambled = np.bitwise_xor(data[permutation], bits)
    scrambled = scrambled.reshape(np.asarray(binary_image).shape)
    return scrambled, permutation, bits


def chaotic_descramble(scrambled_image, key="change-this-passphrase"):
    """Exact inverse of chaotic_scramble()."""
    shape = np.asarray(scrambled_image).shape
    s = np.asarray(scrambled_image).flatten().astype(np.uint8)
    permutation, bits = generate_keystream(len(s), key)

    unxored = np.bitwise_xor(s, bits)
    original = np.empty_like(unxored)
    original[permutation] = unxored
    return original.reshape(shape)