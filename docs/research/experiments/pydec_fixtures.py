"""Export Hirani's PyDEC volumes and Hodge stars as fixtures for gax's discrete exterior calculus.

PyDEC is the outside reference for `gax::dec` (ADR-050): for each mesh, the primal volumes, the
circumcentric dual volumes and the diagonal Hodge stars of the simplices of every dimension, each
simplex named by its sorted vertices. PyDEC signs the volumes of top simplices that fill their
space by orientation; gax's metric is orientation-free, so the tests compare those in absolute
value.

The meshes are jittered grids, cut into triangles (2D) or Freudenthal's six tetrahedra per cube
(3D). The jitter is large enough that some are not well-centred, so the signed duals are tested.

PyDEC comes from GitHub (the `pydec` on PyPI is another package):

    python -m venv target/pydec-venv
    target/pydec-venv/bin/pip install numpy scipy "git+https://github.com/hirani/pydec.git"
    target/pydec-venv/bin/python -I docs/research/experiments/pydec_fixtures.py > crates/gax/tests/fixtures/pydec.txt
"""
import itertools

import numpy as np
from pydec import simplicial_complex


def grid2(n, jitter, rng):
    pts = np.array([[i, j] for i in range(n + 1) for j in range(n + 1)], float)
    pts += rng.uniform(-jitter, jitter, pts.shape)
    idx = lambda i, j: i * (n + 1) + j
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    return pts, np.array(tris)


def grid3(n, jitter, rng):
    pts = np.array([[i, j, k] for i in range(n + 1) for j in range(n + 1) for k in range(n + 1)], float)
    pts += rng.uniform(-jitter, jitter, pts.shape)
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
    tets = []
    for i, j, k in itertools.product(range(n), repeat=3):
        for axes in itertools.permutations(range(3)):
            corner = [i, j, k]
            path = [idx(*corner)]
            for a in axes:
                corner[a] += 1
                path.append(idx(*corner))
            tets.append(path)
    return pts, np.array(tets)


def export(name, pts, simplices):
    sc = simplicial_complex((pts, simplices))
    dim = sc.complex_dimension()
    print(f"mesh {name} {dim} {len(pts)} {len(simplices)}")
    for p in pts:
        print(" ".join(repr(float(x)) for x in p))
    for s in simplices:
        print(" ".join(str(int(v)) for v in s))
    for k in range(dim + 1):
        data = sc[k]
        star = data.star.diagonal()
        print(f"dimension {k} {data.num_simplices}")
        for s, primal, dual, h in zip(data.simplices, data.primal_volume, data.dual_volume, star):
            verts = " ".join(str(int(v)) for v in sorted(s))
            print(f"{verts} {float(primal)!r} {float(dual)!r} {float(h)!r}")


rng = np.random.default_rng(2026)
export("grid2_mild", *grid2(4, 0.15, rng))
export("grid2_rough", *grid2(4, 0.35, rng))
export("grid3_mild", *grid3(2, 0.1, rng))
export("grid3_rough", *grid3(2, 0.25, rng))
