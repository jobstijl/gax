"""Export numga's fields example (docs/fields.md, the bead string) as fixtures for gax's fields.

VGA3D, where numga's vector `x y z` is gax's `e1 e2 e3`. Kernels are written output first and
flattened: a field `vector[4]` as 4 x 3, a field map `vector[4] <- vector[4]` as 4 x 4 x 3 x 3.

    NUMGA_REV=<commit> python docs/research/experiments/numga_fields.py > crates/gax/tests/fixtures/numga_fields.txt
"""
import os

import jax
import numpy as np
from numga import NumpyContext
from numga.algebras import VGA3D
from numga.backend.jax import JaxContext, derivative
from numga.sparse import SparseExtensor

jax.config.update("jax_enable_x64", True)
mv = NumpyContext(VGA3D).multivector
Vector = VGA3D.gatype.vector()

positions = mv.vector([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]).field()
weights = np.array([[2, -1, 0, 0], [-1, 3, -1, 0], [0, -1, 3, -1], [0, 0, -1, 2.0]])
stiffness = (mv.scalar(weights[..., None]) * Vector).field(0, 1)
downward = [[0, 0, 0], [0, 0, -1], [0, 0, -1], [0, 0, 0]]
loads = mv.vector(downward).field()

ends = SparseExtensor(mv.scalar([[-1], [1], [-1], [1], [-1], [1]]),
                      np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 2, 2, 3]), (3, 4))
anchors = SparseExtensor.from_diagonal(mv.scalar(np.ones((4, 1))).field())
sparse_stiffness = (~ends * ends + anchors) * Vector

jm = JaxContext(VGA3D, np.float64).multivector
springs = (jm.scalar(weights[..., None]) * Vector).field(0, 1)
pull = jm.vector(downward).field()


def energy(beads):
    return (beads.scalar_product(springs(beads)) / 2 - pull.scalar_product(beads)).sites.sum()


rest = jm.vector(np.zeros((4, 3))).field()
gradient = derivative(energy)(rest)
curvature = derivative(derivative(energy))(rest)
step = -curvature.solve(gradient)


def flat(e):
    return " ".join(repr(float(v)) for v in np.asarray(e.kernel).ravel())


print(f"# numga {os.environ.get('NUMGA_REV', 'unknown')}: docs/fields.md, the bead string, VGA3D")
print("weights", " ".join(repr(float(w)) for w in weights.ravel()))
print("loads", flat(loads))
print("forces", flat(stiffness(loads)))
print("displacements", flat(stiffness.solve(loads)))
print("composed", flat(stiffness(stiffness)))
print("stretches", flat(ends * positions))
print("sparse_displacements", flat(sparse_stiffness.solve(loads)))
print("gradient", gradient.gatype.signature, "|", flat(gradient))
print("curvature", curvature.gatype.signature, "|", flat(curvature))
print("newton_step", flat(step))
