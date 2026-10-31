"""Export PyDEC's Whitney inner products (the Galerkin Hodge stars) for gax's Whitney forms.

For each mesh of `crates/gax/tests/fixtures/pydec.txt` (read from that file, so the meshes are
the same), and each degree `k`, the nonzero entries of `pydec.fem.whitney_innerproduct(sc, k)`:
the integrals of `<W_s, W_r>` over the mesh, each simplex named by its sorted vertices. Below
the top PyDEC orients simplices by their sorted vertices, as gax does; the top mass is diagonal,
so the top orientation does not enter (the tops are oriented positively first, below).

    target/pydec-venv/bin/python -I docs/research/experiments/pydec_whitney_fixtures.py \\
        crates/gax/tests/fixtures/pydec.txt > crates/gax/tests/fixtures/pydec_whitney.txt
"""
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
from pydec import simplicial_complex  # noqa: E402
from pydec.fem import innerproduct, whitney_innerproduct  # noqa: E402

# PyDEC's k = 1 branch stores a 1 x 1 array as a number, which NumPy 2 refuses: hand it the
# number (larger Gram matrices pass through to the determinant).
innerproduct.inner = lambda a, b: (lambda r: r.reshape(()) if r.size == 1 else r)(np.inner(a, b))


def meshes(path):
    lines = iter(open(path).read().splitlines())
    for head in lines:
        _, name, dim, nv, ns = head.split()
        dim, nv, ns = int(dim), int(nv), int(ns)
        pts = np.array([[float(x) for x in next(lines).split()] for _ in range(nv)])
        simplices = np.array([[int(x) for x in next(lines).split()] for _ in range(ns)])
        for _ in range(dim + 1):
            count = int(next(lines).split()[2])
            for _ in range(count):
                next(lines)
        yield name, pts, simplices


for name, pts, simplices in meshes(sys.argv[1]):
    # `whitney_innerproduct` scales each top simplex by its signed volume, so a mesh whose tops
    # are not all positively oriented (Freudenthal's tetrahedra alternate) comes out wrong: orient
    # them first. Below the top the faces are sorted either way, and the top mass is diagonal.
    simplices = simplices.copy()
    for s in simplices:
        if np.linalg.det(pts[s[1:]] - pts[s[0]]) < 0:
            s[[0, 1]] = s[[1, 0]]
    sc = simplicial_complex((pts, simplices))
    dim = sc.complex_dimension()
    for k in range(dim + 1):
        m = whitney_innerproduct(sc, k).tocoo()
        names = [" ".join(str(int(v)) for v in sorted(s)) for s in sc[k].simplices]
        entries = [(i, j, x) for i, j, x in zip(m.row, m.col, m.data) if x != 0.0]
        print(f"whitney {name} {k} {len(entries)}")
        for i, j, x in entries:
            print(f"{names[i]} ; {names[j]} ; {float(x)!r}")
