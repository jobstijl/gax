# Fields and field maps

A batch holds independent copies: a batch of rotors is many rotors, each acting on its own. A
*field* is one object spread over sites, such as the vertices of a mesh, the beads of a string or
the atoms of a lattice, with a value of the same kind at each site. `gax::field` (feature `std`)
gives the field its own type, `Field<M>`, and the couplings between its sites theirs, the sparse
field map `SparseMap<C>`. The design follows numga's fields (ADR-046 in the
[design record](design.md)), with run-time site counts.

```rust
use gax::field::{Field, SparseMap};
use gax::vga3d::{Scalar, Vector};

// Four beads on a string, the sites of one field.
let positions: Field<Vector<(), f64>> = (0..4).map(|i| Vector::new(i as f64, 0.0, 0.0)).collect();
assert_eq!(positions.len(), 4);
```

## Operations act at every site

An operation on elements acts on a field site by site through `map` (one field) and `zip_with`
(two fields over the same sites); `+`, `-` and unary `-` do the same for fields over the same
sites. A field's reductions run over its sites: `sum`, `mean`, and `dot`, the coefficient inner
product summed over the sites.

```rust
# use gax::field::Field;
# use gax::vga3d::{Bivector, Vector};
# let positions: Field<Vector<(), f64>> = (0..4).map(|i| Vector::new(i as f64, 0.0, 0.0)).collect();
let turn = Bivector::<(), f64>::new(0.0, 0.0, 0.25).exp();
let turned = positions.map(|p| turn >> p); // one turn of the whole string
let centre = positions.mean();
assert_eq!(centre, Vector::new(1.5, 0.0, 0.0));
assert!((turned.mean() - (turn >> centre)).c.iter().all(|d| d.abs() < 1e-15));
```

## Field maps couple sites

A `SparseMap<C>` stores only the couplings that are there. Each is a cell of type `C` that couples
one input site, its column, to one output site, its row. Repeated couplings add. The cell type is
checked at compile time; the site counts at run time, with a panic that names both.

Cells that are multivectors *multiply* the values, as in numga: `&ends * &positions` takes each
cell's product with the value at its column and sums at the rows. `&ends.reverse() * &ends`
multiplies the cells along every path through the shared sites, `&a + &b` adds couplings, and
`&cells * Vector::slot()` turns multivector cells into maps. The string's springs are the
differences of neighbouring beads, and its stiffness is the Laplacian `~ends * ends` with each
bead also tied to its rest position:

```rust
# use gax::field::{Field, SparseMap};
# use gax::vga3d::{Scalar, Vector};
# let positions: Field<Vector<(), f64>> = (0..4).map(|i| Vector::new(i as f64, 0.0, 0.0)).collect();
let s = Scalar::<(), f64>::new;
let ends = SparseMap::new(
    vec![s(-1.0), s(1.0), s(-1.0), s(1.0), s(-1.0), s(1.0)],
    vec![0, 0, 1, 1, 2, 2], // rows: the springs
    vec![0, 1, 1, 2, 2, 3], // columns: the beads
    (3, 4),
);
let stretches = &ends * &positions; // a field over the three springs
assert_eq!(stretches.site(1), Vector::new(1.0, 0.0, 0.0));

let anchors = SparseMap::from_diagonal(&Field::from_fn(4, |_| s(1.0)));
let stiffness = &(&(&ends.reverse() * &ends) + &anchors) * Vector::slot(); // cells Vector <- Vector
let loads = Field::from_fn(4, |i| Vector::new(0.0, 0.0, if i == 1 || i == 2 { -1.0 } else { 0.0 }));
let forces = stiffness.apply(&loads);
let displacements = stiffness.solve(&loads); // the coupled system over every site and blade
assert!((displacements.site(1).e3() + 2.0 / 3.0).abs() < 1e-15);
assert!((stiffness.apply(&displacements) - &loads).dot(&(stiffness.apply(&displacements) - &loads)) < 1e-28);
# let _ = forces;
```

Cells that are maps *apply* to the values: `apply` is the block matrix-vector product, `compose`
the block matrix product (summed along the paths through the shared sites), and `transpose` runs
every coupling back with its cell's matrix transposed, the adjoint under the fields' coefficient
inner product.

### Solving

* `solve` runs conjugate gradients with a diagonal preconditioner over the whole coupled system.
  The couplings must be symmetric positive definite, as a stiffness, a mass matrix or a Laplacian
  with anchors are. If the iteration fails, `solve` panics rather than return a wrong answer.
* `cg(b, tolerance, max_iterations)` is the same iteration, and reports its iterations, its
  residual and whether it converged.
* `factor` computes a sparse LU of the block matrix, and `solve_direct` is `factor` followed by
  one solve. It works for any nonsingular map: non-symmetric, indefinite, or a saddle-point
  system of constraints, where conjugate gradients fails.
  * The sites are ordered by minimum degree, which keeps the factors of meshes sparse.
  * The rows are pivoted for stability.
  * A singular map is reported as `Err(Singular { site, blade })`.
  * The factors solve again for every new right-hand side, as the steps of an implicit time
    integration need.

```rust
# use gax::field::{Field, SparseMap};
# use gax::vga3d::{Scalar, Vector};
# let s = Scalar::<(), f64>::new;
# let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1, 2, 2], vec![0, 1, 1, 2, 2, 3], (3, 4));
# let anchors = SparseMap::from_diagonal(&Field::from_fn(4, |_| s(1.0)));
# let stiffness = &(&(&ends.reverse() * &ends) + &anchors) * Vector::slot();
let lu = stiffness.factor().expect("the anchored string is nonsingular");
for step in 1..=3 {
    let loads = Field::from_fn(4, |_| Vector::new(0.0, 0.0, -f64::from(step)));
    let sag = lu.solve(&loads); // no new factorization
    assert!((sag.site(0).e3() - sag.site(3).e3()).abs() < 1e-15); // symmetric under the load
}
```

### Least squares

`lstsq` takes field maps of any shape:
- more output sites than input sites (fitting);
- fewer (under-determined);
- cells between kinds of different sizes.

It returns the least-squares solution of least norm, so a singular system's gauge (a free
mesh's translation, a constant potential) is left at zero, as in numga. It runs LSQR, iterating
with the map and its transpose; `lsqr(b, tolerance, max_iterations)` reports iterations and
residual.

```rust
# use gax::field::{Field, SparseMap};
# use gax::vga3d::{Scalar, Vector};
# let s = Scalar::<(), f64>::new;
# let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1, 2, 2], vec![0, 1, 1, 2, 2, 3], (3, 4));
// The string from its springs' stretches alone: the beads' positions up to a translation,
// which the least-norm solution leaves at zero (the beads centred on the origin).
let stretches = Field::from_fn(3, |_| Vector::new(1.0, 0.0, 0.0));
let beads = (&ends * Vector::slot()).lstsq(&stretches);
assert!((beads.site(0).e1() + 1.5).abs() < 1e-12 && (beads.site(3).e1() - 1.5).abs() < 1e-12);
```

### Eigenproblems

`eigh(metric, count)` gives the `count` eigenpairs nearest zero of `A x = λ M x`, for a
symmetric field map and a symmetric positive definite metric: the vibration modes of a mesh, or
the lowest states of a lattice Hamiltonian. The values come ascending, and the fields are
orthonormal in the metric. Free meshes have a null space (their rigid motions), which comes
first. Each value repeated on every coefficient of the kind (isotropic cells) is found with all
its copies.

```rust
# use gax::field::{Field, SparseMap};
# use gax::vga3d::{Scalar, Vector};
# let s = Scalar::<(), f64>::new;
# let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1, 2, 2], vec![0, 1, 1, 2, 2, 3], (3, 4));
let laplacian = &(&ends.reverse() * &ends) * Vector::slot(); // the free string
let mass = SparseMap::from_diagonal(&Field::from_fn(4, |_| Vector::<(), f64>::slot()));
let (values, modes) = laplacian.eigh(&mass, 6);
assert!(values[..3].iter().all(|v| v.abs() < 1e-12)); // translations along x, y and z
let wave = 2.0 - 2.0 * (std::f64::consts::PI / 4.0).cos(); // the half wave, on each axis
assert!(values[3..].iter().all(|v| (v - wave).abs() < 1e-12));
assert!((modes[3].dot(&mass.apply(&modes[3])) - 1.0).abs() < 1e-12);
```

The search space comes from block Lanczos on `(A − σM)⁻¹ M`, with σ just below zero and one
sparse LU of `A − σM`. The eigenpairs come from `A` and `M` themselves on that space, and the
space grows until each pair's residual `‖A x − λ M x‖` is at the level of rounding. Blocks of
`2 K::N + 2` vectors cover values repeated on every coefficient and doubled by a symmetry of the
sites; `eigh_with_block` takes larger blocks for more symmetric meshes.

## On SIMD lanes

`stiffness.packed()` lays out a field map of square map cells for the batch kernels (feature
`batch`), once. Its `apply` runs on SIMD lanes, at the best level the CPU has. The layout is sliced
ELLPACK: the rows are taken 16 at a time, each lane works on one row, the cell coefficients are
vector loads, and only the values at the columns are gathered.

It pays where a lane batch is wide, in `f32`, for a map applied many times, such as the step of
an explicit simulation. On a string of 1024 sites it applies in 0.78 of the scalar loop's time
(0.66 with `target-cpu=native`); in `f64`, with half the lanes, it is about even. Solving stays
with `SparseMap::solve`: in conjugate gradients the vector arithmetic weighs as much as the
products, and a packed solve measured no faster ([performance.md](performance.md)).

```rust
# use gax::field::{Field, SparseMap};
# use gax::vga3d::{Scalar, Vector};
# let s = Scalar::<(), f32>::new;
# let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1, 2, 2], vec![0, 1, 1, 2, 2, 3], (3, 4));
# let anchors = SparseMap::from_diagonal(&Field::from_fn(4, |_| s(1.0)));
# let stiffness = &(&(&ends.reverse() * &ends) + &anchors) * Vector::slot();
# let positions = Field::from_fn(4, |i| Vector::new(i as f32, 0.0, 0.0));
# #[cfg(feature = "batch")] {
let packed = stiffness.packed();
let forces = packed.apply(&positions); // the same as stiffness.apply(&positions), on lanes
let difference = &forces - &stiffness.apply(&positions);
assert!(difference.dot(&difference) < 1e-10);
# }
```

## Derivatives with respect to a field

The derivative of a scalar function of a field keeps the sites in the step's slot, so every site
couples to every other. `gax::field::gradient` gives a field of linear forms `Scalar <- Vector`,
one per site, and `gax::field::hessian` adds the second derivative as a sparse field map of form
cells `Scalar <- (Vector, Vector)`. Blocks that come out exactly zero are not stored. The function
is written generic over the coefficient type, as for `gax::dual`. Solving the one against the
other is the coupled Newton step of the whole string:

```rust
# use gax::field::{Field, SparseMap};
# use gax::vga3d::{Scalar, Vector};
use gax::Real;

/// The springs' energy less the work of the loads, `½ Σ x·K x − Σ load·x`.
fn energy<T: Real>(x: &Field<Vector<(), T>>) -> T {
    let c = T::from_f64;
    let mut e = T::zero();
    for i in 0..x.len() {
        let anchor = x.site(i).scalar_product(x.site(i)).s();
        e = e + c(0.5) * anchor;
        if i + 1 < x.len() {
            let d = x.site(i + 1) - x.site(i);
            e = e + c(0.5) * d.scalar_product(d).s();
        }
        if i == 1 || i == 2 {
            e = e + x.site(i).e3(); // a unit load downward
        }
    }
    e
}

let rest = Field::<Vector<(), f64>>::zeros(4);
let (_, gradient, curvature) = gax::field::hessian(energy, &rest);
let step = -&curvature
    .map_cells(Scalar::as_map) // the forms' cells as maps, Vector <- Vector
    .solve(&gradient.map(|g| Vector::from_coeffs(g.c[0])));
assert!((step.site(1).e3() + 2.0 / 3.0).abs() < 1e-15); // the displacements: the energy is quadratic
```

`hessian` evaluates the function once per pair of sites, `n (n + 1) / 2` times on jets of jets, and
`gradient` once per site. That suits small fields. An energy that is a sum of local terms, such as
the springs of a string or the elements of a mesh, has a sparse Hessian that the chain rule
assembles in one pass:

1. Let `D` be the sparse map from the field to the terms' arguments (`ends` here, the differences
   of neighbours).
2. Take `gax::dual::hessian` of each term at its argument (`ends * x`). That gives each term's
   gradient form and Hessian form.
3. The gradient is `~D * g`, a field of forms.
4. The Hessian is `~D * diag(H) * D`, with `diag(H)` the terms' Hessians as diagonal cells.

Each term's Hessian is a small dense form, and `D` keeps the coupling as sparse as the mesh. The
fields test (`tests/fields.rs`) does this for the string and matches numga's derivative exactly.

## Performance

On a string of 1024 sites with 3070 couplings, each its own PGA3D `Point <- Point` (`f32`), see
[performance.md](performance.md) for the timings of the scalar and the packed products against
the dense batch path (one map applied to as many values).

## What a field is not (yet)

* Sites at compile time. ADR-045 designs fields over a fixed number of sites as kinds,
  `Sites<K, N>`, so that they fill slots and the dense solvers apply. They are not built.
* A sparse SVD and a non-symmetric sparse eigensolver. The SVD of a small field is the dense
  SVD of its blocks.
