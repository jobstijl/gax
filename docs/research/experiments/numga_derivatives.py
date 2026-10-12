"""Export numga's `derivative` (JAX backend) as fixtures for gax's typed derivatives.

VGA3D only: numga's vector `x y z`, bivector `yz zx xy` and rotor `1 yz zx xy` are gax's
`e1 e2 e3`, `e23 e31 e12` and `1 e23 e31 e12`, same order and orientation, so the kernels are
written as they are. Run with numga and jax installed:

    NUMGA_REV=<commit> python docs/research/experiments/numga_derivatives.py > crates/gax/tests/fixtures/numga_derivatives.txt
"""
import os

import jax
import numpy as np
from numga.algebras import VGA3D
from numga.backend.jax import JaxContext, derivative


jax.config.update("jax_enable_x64", True)
mv = JaxContext(VGA3D, np.float64).multivector

a = mv.vector([0.4, -1.1, 0.7])
b = mv.vector([0.9, 0.2, -0.5])


def energy(x):
    """A non-quadratic scalar function of a vector: |x - a|³ / 3 + (x · b)²."""
    n = (x - a).norm()
    xb = x.scalar_product(b)
    return n * n * n / 3 + xb * xb


def flat(e):
    return " ".join(repr(float(v)) for v in np.asarray(e.kernel).ravel())


# The numga commit the venv was installed from (the installed package carries no git data).
rev = os.environ.get("NUMGA_REV", "unknown")
print(f"# numga {rev}: derivative (JAX) in VGA3D; kernels output first, flattened")
print("# energy(x) = |x - a|^3 / 3 + (x . b)^2 with a = (0.4, -1.1, 0.7), b = (0.9, 0.2, -0.5)")
for x0 in ([0.3, -0.2, 0.5], [-1.0, 0.25, 2.0]):
    x = mv.vector(x0)
    g = derivative(energy)(x)
    h = derivative(derivative(energy))(x)
    print("x", " ".join(map(repr, x0)))
    print("energy", flat(energy(x)))
    print("gradient", g.gatype.signature, "|", flat(g))
    print("hessian", h.gatype.signature, "|", flat(h))
for b0 in ([0.3, -0.2, 0.5], [1.2, 0.7, -2.1]):
    bv = mv.bivector(b0)
    r = bv.exp()
    j = derivative(lambda q: q.exp())(bv)
    print("bivector", " ".join(map(repr, b0)))
    print("exp", flat(r))
    print("jacobian", j.gatype.signature, "|", flat(j))
