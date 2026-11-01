"""Export Gauss-Legendre rules as fixtures for `gax::quadrature::gauss_legendre`.

For each `n`, a line `n <n>`, then the nodes and the weights, ascending: numpy's `leggauss`
(numga's `fluids/kirchhoff` calls it), and the same rule to 50 digits (Newton's method in
`decimal` from numpy's nodes), rounded to the nearest double: numpy's own weights are off by up
to 4e-15 (n = 48), so the exact rule is the reference and numpy's a cross-check.

    target/numga-venv-35797de/bin/python -I docs/research/experiments/numpy_leggauss.py > crates/gax/tests/fixtures/numpy_leggauss.txt
"""
from decimal import Decimal, getcontext

import numpy as np

getcontext().prec = 50


def exact(n, start):
    def legendre(t):
        p0, p1 = Decimal(1), t
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * t * p1 - (k - 1) * p0) / k
        if n == 1:
            p0 = Decimal(1)
        return p1, n * (t * p1 - p0) / (t * t - 1)

    t = Decimal(float(start))
    for _ in range(40):
        p, dp = legendre(t)
        t -= p / dp
    _, dp = legendre(t)
    return float(t), float(2 / ((1 - t * t) * dp * dp))


for n in [1, 2, 3, 4, 5, 8, 13, 20, 33, 48, 64, 100]:
    x, w = np.polynomial.legendre.leggauss(n)
    pairs = [exact(n, v) for v in x]
    print(f"n {n}")
    print(" ".join(repr(float(v)) for v in x))
    print(" ".join(repr(float(v)) for v in w))
    print(" ".join(repr(p[0]) for p in pairs))
    print(" ".join(repr(p[1]) for p in pairs))
