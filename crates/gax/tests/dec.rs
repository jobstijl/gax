//! Discrete exterior calculus (ADR-050): volumes and Hodge stars against Hirani's `PyDEC`, the same
//! in every model, both stars' cotangent Laplacian, and Stokes's theorem for the de Rham map,
//! exactly.

#![cfg(all(
    feature = "vga2d",
    feature = "vga3d",
    feature = "pga2d",
    feature = "pga3d",
    feature = "cga2d",
    feature = "cga3d"
))]

mod exact;

use exact::{Rng, cubic};
use gax::Coef;
use gax::dec::{Geometry, de_rham};
use gax::field::{Field, SparseMap};
use gax::fp::Fp;
use gax::mesh::Mesh;
use gax::simplex::Rule;
use std::collections::BTreeMap;

/// A mesh of `PyDEC`'s fixtures, with its volumes and stars per dimension, each simplex by its
/// sorted vertices: `(primal, dual, star)`.
struct Fixture {
    name: String,
    dim: usize,
    points: Vec<Vec<f64>>,
    simplices: Vec<Vec<usize>>,
    values: Vec<BTreeMap<Vec<usize>, (f64, f64, f64)>>,
}

fn fixtures() -> Vec<Fixture> {
    let text = include_str!("fixtures/pydec.txt");
    let mut lines = text.lines().peekable();
    let mut out = Vec::new();
    while let Some(head) = lines.next() {
        let w: Vec<&str> = head.split_whitespace().collect();
        assert_eq!(w[0], "mesh");
        let (name, dim, nv, ns) = (
            w[1].to_string(),
            w[2].parse().unwrap(),
            w[3].parse().unwrap(),
            w[4].parse().unwrap(),
        );
        let nums = |l: &str| {
            l.split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let ints = |l: &str| {
            l.split_whitespace()
                .map(|x| x.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        };
        let points: Vec<Vec<f64>> = (0..nv).map(|_| nums(lines.next().unwrap())).collect();
        let simplices: Vec<Vec<usize>> = (0..ns).map(|_| ints(lines.next().unwrap())).collect();
        let mut values = Vec::new();
        for k in 0..=dim {
            let w: Vec<usize> = ints(lines.next().unwrap().trim_start_matches("dimension"));
            assert_eq!(w[0], k);
            let mut map = BTreeMap::new();
            for _ in 0..w[1] {
                let x: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                let verts = x[..=k].iter().map(|v| v.parse().unwrap()).collect();
                let f = |i: usize| x[k + 1 + i].parse::<f64>().unwrap();
                map.insert(verts, (f(0), f(1), f(2)));
            }
            values.push(map);
        }
        out.push(Fixture {
            name,
            dim,
            points,
            simplices,
            values,
        });
    }
    out
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

/// Compare a geometry's volumes, duals and stars with `PyDEC`'s, simplex by simplex.
fn compare<const N: usize>(
    fx: &Fixture,
    mesh_complex: &gax::mesh::Complex<N>,
    g: &Geometry<f64, N>,
) {
    for k in 0..N {
        let star: Vec<f64> = g
            .star::<gax::vga3d::Scalar<(), f64>>(k)
            .couplings()
            .map(|(_, _, c)| c.s())
            .collect();
        for (i, s) in mesh_complex.simplices(k).enumerate() {
            let mut key = s.to_vec();
            key.sort_unstable();
            let (primal, dual, want_star) = fx.values[k][&key];
            let top = k == fx.dim;
            let (p, d, h) = (g.volumes(k)[i], g.dual_volumes(k)[i], star[i]);
            let ok = if top {
                close(p, primal.abs()) && close(d, dual) && close(h, want_star.abs())
            } else {
                close(p, primal) && close(d, dual) && close(h, want_star)
            };
            assert!(
                ok,
                "{} k = {k} {s:?}: gax ({p}, {d}, {h}), PyDEC ({primal}, {dual}, {want_star})",
                fx.name
            );
        }
    }
}

#[test]
fn volumes_and_stars_match_pydec() {
    for fx in fixtures() {
        match fx.dim {
            2 => {
                use gax::vga2d::Vector;
                let v: Field<_> = fx
                    .points
                    .iter()
                    .map(|p| Vector::<(), f64>::new(p[0], p[1]))
                    .collect();
                let tops: Vec<[usize; 3]> =
                    fx.simplices.iter().map(|s| [s[0], s[1], s[2]]).collect();
                let mesh = Mesh::new(v, &tops).unwrap();
                compare(&fx, mesh.complex(), &Geometry::new(&mesh));
            }
            3 => {
                use gax::vga3d::Vector;
                let v: Field<_> = fx
                    .points
                    .iter()
                    .map(|p| Vector::<(), f64>::new(p[0], p[1], p[2]))
                    .collect();
                let tops: Vec<[usize; 4]> = fx
                    .simplices
                    .iter()
                    .map(|s| [s[0], s[1], s[2], s[3]])
                    .collect();
                let mesh = Mesh::new(v, &tops).unwrap();
                compare(&fx, mesh.complex(), &Geometry::new(&mesh));
            }
            _ => unreachable!(),
        }
    }
}

/// The geometry depends on the model only through its distances: PGA's points and CGA's round
/// points give VGA's numbers.
#[test]
fn every_model_gives_the_same_geometry() {
    for fx in fixtures().into_iter().filter(|f| f.dim == 3) {
        let tops: Vec<[usize; 4]> = fx
            .simplices
            .iter()
            .map(|s| [s[0], s[1], s[2], s[3]])
            .collect();
        let vga: Field<_> = fx
            .points
            .iter()
            .map(|p| gax::vga3d::Vector::<(), f64>::new(p[0], p[1], p[2]))
            .collect();
        let pga: Field<_> = fx
            .points
            .iter()
            .map(|p| gax::pga3d::Point::<(), f64>::xyz(p[0], p[1], p[2]))
            .collect();
        let cga: Field<_> = fx
            .points
            .iter()
            .map(|p| gax::cga3d::Vector::<(), f64>::up(p[0], p[1], p[2]))
            .collect();
        let a = Geometry::new(&Mesh::new(vga, &tops).unwrap());
        let b = Geometry::new(&Mesh::new(pga, &tops).unwrap());
        let c = Geometry::new(&Mesh::new(cga, &tops).unwrap());
        for k in 0..4 {
            for ((x, y), z) in a
                .dual_volumes(k)
                .iter()
                .zip(b.dual_volumes(k))
                .zip(c.dual_volumes(k))
            {
                assert!(
                    close(*x, *y) && close(*x, *z),
                    "{} k = {k}: {x} {y} {z}",
                    fx.name
                );
            }
            for ((x, y), z) in a.volumes(k).iter().zip(b.volumes(k)).zip(c.volumes(k)) {
                assert!(
                    close(*x, *y) && close(*x, *z),
                    "{} k = {k}: {x} {y} {z}",
                    fx.name
                );
            }
        }
    }
}

type S2 = gax::vga2d::Scalar<(), f64>;

/// `Aᵀ B A` as a map of `(row, column) → value`.
fn sandwich(a: &SparseMap<S2>, b: &SparseMap<S2>) -> BTreeMap<(usize, usize), f64> {
    let at = a.transpose_with(|c| c);
    let m = &at * &(b * a);
    let mut out = BTreeMap::new();
    for (r, c, x) in m.couplings() {
        *out.entry((r, c)).or_insert(0.0) += x.s();
    }
    out
}

/// The stiffness of linear finite elements, `Kₐ_b = Σ V ∇λₐ · ∇λ_b` over the top simplices, with the
/// barycentric gradients computed here independently, as the reciprocal frame of each simplex's
/// edges: `∇λ₁ = (e₂ ∧ e₃)(e₁ ∧ e₂ ∧ e₃)⁻¹` and so on, `∇λ₀ = −Σ ∇λᵢ`.
fn linear_elements(points: &[Vec<f64>], tops: &[Vec<usize>]) -> BTreeMap<(usize, usize), f64> {
    let mut k = BTreeMap::new();
    for t in tops {
        let grads: Vec<Vec<f64>> = if t.len() == 3 {
            use gax::vga2d::Vector;
            let p: Vec<Vector<(), f64>> = t
                .iter()
                .map(|&i| Vector::new(points[i][0], points[i][1]))
                .collect();
            let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
            let area = e1 ^ e2;
            let inv = area.inverse();
            // e¹ = e₂ (e₁∧e₂)⁻¹ up to sign: e¹ · e₁ = 1, e¹ · e₂ = 0.
            let g1 = e2.lc(inv) * -1.0;
            let g2 = e1.lc(inv);
            let g = [g1, g2];
            let vol = area.c[0].abs() / 2.0;
            let mut out = vec![vec![vol]];
            out.push(vec![-(g1.c[0] + g2.c[0]), -(g1.c[1] + g2.c[1])]);
            out.extend(g.iter().map(|v| vec![v.c[0], v.c[1]]));
            out
        } else {
            use gax::vga3d::Vector;
            let p: Vec<Vector<(), f64>> = t
                .iter()
                .map(|&i| Vector::new(points[i][0], points[i][1], points[i][2]))
                .collect();
            let (e1, e2, e3) = (p[1] - p[0], p[2] - p[0], p[3] - p[0]);
            let cube = e1 ^ e2 ^ e3;
            let inv = cube.inverse();
            let g1 = ((e2 ^ e3) * inv).grade::<1>();
            let g2 = ((e3 ^ e1) * inv).grade::<1>();
            let g3 = ((e1 ^ e2) * inv).grade::<1>();
            let vol = cube.c[0].abs() / 6.0;
            let mut out = vec![vec![vol]];
            out.push((0..3).map(|i| -(g1.c[i] + g2.c[i] + g3.c[i])).collect());
            out.extend([g1, g2, g3].iter().map(|v| v.c.to_vec()));
            out
        };
        let vol = grads[0][0];
        let g = &grads[1..];
        for (a, &va) in t.iter().enumerate() {
            for (b, &vb) in t.iter().enumerate() {
                let dot: f64 = g[a].iter().zip(&g[b]).map(|(x, y)| x * y).sum();
                *k.entry((va, vb)).or_insert(0.0) += vol * dot;
            }
        }
    }
    k
}

fn assert_same(a: &BTreeMap<(usize, usize), f64>, b: &BTreeMap<(usize, usize), f64>, what: &str) {
    for key in a.keys().chain(b.keys()) {
        let (x, y) = (
            a.get(key).copied().unwrap_or(0.0),
            b.get(key).copied().unwrap_or(0.0),
        );
        assert!(
            (x - y).abs() < 1e-12 * (1.0 + x.abs()),
            "{what} {key:?}: {x} {y}"
        );
    }
}

/// The Galerkin stiffness on 0-forms, `d₀ᵀ M₁ d₀`, is the stiffness of linear finite elements, in
/// 2D and 3D (the Whitney forms' `d W⁰ = W¹ d₀`). On triangles the diagonal star gives it too, the
/// cotangent formula; on tetrahedra it does not, except on symmetric ones (a regular
/// tetrahedron), so there the diagonal star is checked against `PyDEC` alone. The top mass is the
/// top star, `1/V`.
#[test]
fn the_galerkin_stiffness_is_linear_finite_elements() {
    use gax::vga3d::Vector;
    for fx in fixtures() {
        let fem = linear_elements(&fx.points, &fx.simplices);
        if fx.dim == 2 {
            let v: Field<_> = fx
                .points
                .iter()
                .map(|p| gax::vga2d::Vector::<(), f64>::new(p[0], p[1]))
                .collect();
            let tops: Vec<[usize; 3]> = fx.simplices.iter().map(|s| [s[0], s[1], s[2]]).collect();
            let mesh = Mesh::new(v, &tops).unwrap();
            let g = Geometry::new(&mesh);
            assert_same(&sandwich(&mesh.d(0), &g.mass(1)), &fem, &fx.name);
            assert_same(&sandwich(&mesh.d(0), &g.star(1)), &fem, &fx.name);
            let top: Vec<f64> = g.mass::<S2>(2).couplings().map(|(_, _, c)| c.s()).collect();
            for (m, v) in top.iter().zip(g.volumes(2)) {
                assert!(close(*m, v.recip()));
            }
        } else {
            let v: Field<_> = fx
                .points
                .iter()
                .map(|p| gax::vga3d::Vector::<(), f64>::new(p[0], p[1], p[2]))
                .collect();
            let tops: Vec<[usize; 4]> = fx
                .simplices
                .iter()
                .map(|s| [s[0], s[1], s[2], s[3]])
                .collect();
            let mesh = Mesh::new(v, &tops).unwrap();
            let g = Geometry::new(&mesh);
            let d0: SparseMap<S2> = mesh.complex().d(0);
            assert_same(&sandwich(&d0, &g.mass(1)), &fem, &fx.name);
        }
    }
    // A regular tetrahedron: there the diagonal star agrees.
    let corners = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let v: Field<_> = corners
        .iter()
        .map(|p| Vector::<(), f64>::new(p[0], p[1], p[2]))
        .collect();
    let mesh = Mesh::new(v, &[[0, 1, 2, 3]]).unwrap();
    let g = Geometry::new(&mesh);
    let d0: SparseMap<S2> = mesh.complex().d(0);
    let points: Vec<Vec<f64>> = corners.iter().map(|c| c.to_vec()).collect();
    assert_same(
        &sandwich(&d0, &g.star(1)),
        &linear_elements(&points, &[vec![0, 1, 2, 3]]),
        "regular",
    );
}

/// Stokes's theorem for the de Rham map, `R_{k+1}(∇∧F) = d_k R_k(F)`, exactly on the prime
/// field, for cubic fields on random triangles and tetrahedra: scalar fields (the gradient
/// theorem), vector fields (Green's and Stokes's), bivector fields in 3D (Gauss's).
#[test]
fn de_rham_commutes_with_the_derivative() {
    use gax::calculus::curl;
    macro_rules! stokes {
        ($rng:expr, $alg:ident, $n:literal, $k:literal, $K:ident) => {{
            use gax::$alg::Vector;
            fn field<T: Coef>(c: &[i64], x: Vector<(), T>) -> gax::$alg::$K<(), T> {
                cubic::<_, gax::$alg::$K>(c, x)
            }
            let c = $rng.cubic();
            let vertices: Field<Vector<(), Fp>> = (0..8).map(|_| $rng.point()).collect();
            let tops: Vec<[usize; $n]> = (0..6)
                .map(|t| core::array::from_fn(|i| (t + i * 3) % 8))
                .collect();
            let mesh = Mesh::new(vertices, &tops).unwrap();
            let r_k: Field<gax::$alg::Scalar<(), Fp>> =
                de_rham::<_, _, _, $n, { $k + 1 }>(&mesh, Rule::degree(3), |x| field(&c, x));
            let r_dk: Field<gax::$alg::Scalar<(), Fp>> =
                de_rham::<_, _, _, $n, { $k + 2 }>(&mesh, Rule::degree(2), |x| {
                    curl(|y| field(&c, y), x).1
                });
            let d: SparseMap<gax::$alg::Scalar<(), Fp>> = mesh.complex().d($k);
            assert_eq!(
                &d * &r_k,
                r_dk,
                "{} k = {}, a {} field",
                stringify!($alg),
                $k,
                stringify!($K)
            );
        }};
    }
    let mut rng = Rng(21);
    stokes!(rng, vga2d, 3, 0, Scalar);
    stokes!(rng, vga2d, 3, 1, Vector);
    stokes!(rng, vga3d, 4, 0, Scalar);
    stokes!(rng, vga3d, 4, 1, Vector);
    stokes!(rng, vga3d, 4, 2, Bivector);
}

/// The Whitney map on 1-cochains inverts the de Rham map, `R ∘ W = id`: its integral along each
/// edge of each top simplex is the cochain's value there (the field is linear along the edge, so
/// its value at the midpoint gives the mean). And `W₁ d₀ = ∇ W₀`: the Whitney field of the
/// differences of a vertex field is the gradient of its linear interpolant, `Σ fᵥ ∇λᵥ`.
#[test]
fn whitney_inverts_de_rham() {
    use gax::vga3d::{Scalar, Vector};
    for fx in fixtures().into_iter().filter(|f| f.dim == 3) {
        let v: Field<_> = fx
            .points
            .iter()
            .map(|p| Vector::<(), f64>::new(p[0], p[1], p[2]))
            .collect();
        let tops: Vec<[usize; 4]> = fx
            .simplices
            .iter()
            .map(|s| [s[0], s[1], s[2], s[3]])
            .collect();
        let mesh = Mesh::new(v, &tops).unwrap();
        let g = Geometry::new(&mesh);
        let c = mesh.complex();
        let cochain: Field<Scalar<(), f64>> = (0..c.len(1))
            .map(|e| Scalar::new((e as f64 * 0.37).sin()))
            .collect();
        for t in 0..c.len(3) {
            let s = c.simplex(3, t);
            for a in 0..4 {
                for b in a + 1..4 {
                    let (lo, hi) = if s[a] < s[b] { (a, b) } else { (b, a) };
                    let mut mid = [0.0; 4];
                    mid[lo] = 0.5;
                    mid[hi] = 0.5;
                    let w = g.whitney_1(&mesh, &cochain, t, &mid);
                    let along = mesh.vertices.site(s[hi]) - mesh.vertices.site(s[lo]);
                    let e = c.index_of(1, &[s[lo], s[hi]]).unwrap();
                    let got = w.scalar_product(along).s();
                    assert!(
                        close(got, cochain.site(e).s()),
                        "{} top {t} edge {e}: {got}",
                        fx.name
                    );
                }
            }
        }
        // W₁ d₀ f is constant on each top simplex and equals ∇(Σ fᵥ λᵥ).
        let f: Field<Scalar<(), f64>> = (0..c.len(0))
            .map(|i| Scalar::new((i as f64 * 0.71).cos()))
            .collect();
        let df = &mesh.d(0) * &f;
        for t in 0..c.len(3) {
            let s = c.simplex(3, t);
            let at_centre = g.whitney_1(&mesh, &df, t, &[0.25; 4]);
            let at_corner = g.whitney_1(&mesh, &df, t, &[0.7, 0.1, 0.1, 0.1]);
            assert!((at_centre - at_corner).c.iter().all(|x| x.abs() < 1e-12));
            // The gradient's change along each edge from vertex 0 is the field's.
            for j in 1..4 {
                let along = mesh.vertices.site(s[j]) - mesh.vertices.site(s[0]);
                let change = f.site(s[j]).s() - f.site(s[0]).s();
                assert!(close(at_centre.scalar_product(along).s(), change));
            }
        }
    }
}

/// The icosahedron's faces subdivided `level` times at their edges' midpoints, projected to the
/// unit sphere: well-centred, its triangles near equilateral.
fn icosphere(level: usize) -> Mesh<gax::vga3d::Vector<(), f64>, 3> {
    use gax::vga3d::Vector;
    let t = f64::midpoint(1.0, 5f64.sqrt());
    let mut points: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..level {
        let mut midpoints = BTreeMap::new();
        let mut mid = |a: usize, b: usize, points: &mut Vec<[f64; 3]>| -> usize {
            *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
                points.push(core::array::from_fn(|i| {
                    f64::midpoint(points[a][i], points[b][i])
                }));
                points.len() - 1
            })
        };
        faces = faces
            .iter()
            .flat_map(|&[a, b, c]| {
                let (ab, bc, ca) = (
                    mid(a, b, &mut points),
                    mid(b, c, &mut points),
                    mid(c, a, &mut points),
                );
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            })
            .collect();
    }
    let vertices: Field<_> = points
        .iter()
        .map(|p| {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            Vector::<(), f64>::new(p[0] / r, p[1] / r, p[2] / r)
        })
        .collect();
    Mesh::new(vertices, &faces).unwrap()
}

/// A torus of revolution, radii `big` and `small`, as a lattice of `n × m` vertices skewed by
/// half a step per ring, so that its triangles are near equilateral.
fn torus(n: usize, m: usize, big: f64, small: f64) -> Mesh<gax::vga3d::Vector<(), f64>, 3> {
    use gax::vga3d::Vector;
    use std::f64::consts::TAU;
    assert!(
        m.is_multiple_of(2),
        "the skew closes up after an even number of rings"
    );
    // Row j sits half a step on per ring, so after m rings the lattice has moved m/2 steps:
    // row m is row 0 shifted, and the seam joins i to i + m/2.
    let id = |i: usize, j: usize| (j % m) * n + (i + (j / m) * (m / 2)) % n;
    let vertices: Field<_> = (0..m)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .map(|(i, j)| {
            let (u, v) = (
                TAU * (i as f64 + 0.5 * j as f64) / n as f64,
                TAU * j as f64 / m as f64,
            );
            let r = big + small * v.cos();
            Vector::<(), f64>::new(r * u.cos(), r * u.sin(), small * v.sin())
        })
        .collect();
    let faces: Vec<[usize; 3]> = (0..m)
        .flat_map(|j| {
            (0..n).flat_map(move |i| {
                [
                    [id(i, j), id(i + 1, j), id(i, j + 1)],
                    [id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)],
                ]
            })
        })
        .collect();
    Mesh::new(vertices, &faces).unwrap()
}

type S3 = gax::vga3d::Scalar<(), f64>;

/// `δ` is the adjoint of `d` in the stars' inner products, `⟨d a, b⟩_⋆ = ⟨a, δ b⟩_⋆`, so the
/// images of `d` and `δ` are orthogonal and the Hodge decomposition `ω = dα + δβ + h` is
/// orthogonal; and `D² = Δ` degree by degree. On a sphere and on the 3D fixtures (whose stars
/// are signed but not zero).
#[test]
fn codifferential_is_the_adjoint_and_dirac_squares_to_the_laplacian() {
    let sphere = icosphere(2);
    let g = Geometry::new(&sphere);
    assert!(g.is_well_centered());
    let field = |n: usize, seed: f64| -> Field<S3> {
        (0..n).map(|i| S3::new((i as f64 * seed).sin())).collect()
    };
    let c = sphere.complex();
    for k in 0..2 {
        let (a, b) = (field(c.len(k), 0.37), field(c.len(k + 1), 0.53));
        let da = &c.d::<S3>(k) * &a;
        let lhs = da.dot(&(&g.star::<S3>(k + 1) * &b));
        let rhs = a.dot(&(&g.star::<S3>(k) * &(&g.codifferential::<S3>(k + 1) * &b)));
        assert!(
            (lhs - rhs).abs() < 1e-12 * (1.0 + lhs.abs()),
            "k = {k}: {lhs} {rhs}"
        );
    }
    // D² is block diagonal, its blocks the Laplacians.
    let d = g.dirac::<S3>();
    let at = g.offsets();
    let squared = &d * &d;
    let mut dense = BTreeMap::new();
    for (r, col, x) in squared.couplings() {
        *dense.entry((r, col)).or_insert(0.0) += x.s();
    }
    let mut want = BTreeMap::new();
    for (k, &start) in at.iter().enumerate().take(3) {
        for (r, col, x) in g.laplacian::<S3>(k).couplings() {
            *want.entry((start + r, start + col)).or_insert(0.0) += x.s();
        }
    }
    assert_same(&dense, &want, "D² = Δ");
}

/// The smallest eigenvalues of the pencil `(⋆Δ_k, ⋆_k)`.
fn spectrum(g: &Geometry<f64, 3>, k: usize, count: usize) -> Vec<f64> {
    let form = &g.laplacian_form::<S3>(k) * S3::slot();
    let star = &g.star::<S3>(k) * S3::slot();
    form.eigh(&star, count).0
}

/// The harmonic cochains, the null space of `Δ_k`, have the Betti numbers' dimensions: a sphere
/// (1, 0, 1) and a torus (1, 2, 1).
#[test]
fn harmonic_cochains_count_the_holes() {
    for (mesh, betti) in [
        (icosphere(2), [1, 0, 1]),
        (torus(35, 10, 4.0, 1.0), [1, 2, 1]),
    ] {
        let g = Geometry::new(&mesh);
        assert!(g.is_well_centered());
        assert_eq!(mesh.complex().betti_numbers(), betti);
        for (k, &b) in betti.iter().enumerate() {
            let values = spectrum(&g, k, b + 3);
            let zeros = values.iter().filter(|v| v.abs() < 1e-8).count();
            assert_eq!(zeros, b, "k = {k}: {values:?}");
        }
    }
}

/// On the unit sphere the eigenvalues of the Laplacian are `ℓ(ℓ + 1)`, each `2ℓ + 1` times. Both
/// discretizations converge at second order: the error falls about fourfold per subdivision.
#[test]
fn the_sphere_spectrum_converges() {
    let exact: Vec<f64> = (0..4)
        .flat_map(|l: usize| std::iter::repeat_n((l * (l + 1)) as f64, 2 * l + 1))
        .collect();
    let error = |values: &[f64]| {
        values
            .iter()
            .zip(&exact)
            .map(|(v, e)| (v - e).abs())
            .fold(0.0, f64::max)
    };
    let mut diagonal = Vec::new();
    let mut galerkin = Vec::new();
    for level in 1..4 {
        let sphere = icosphere(level);
        let g = Geometry::new(&sphere);
        diagonal.push(error(&spectrum(&g, 0, 16)));
        let stiffness = &g.stiffness::<S3>(0) * S3::slot();
        let mass = &g.mass::<S3>(0) * S3::slot();
        galerkin.push(error(&stiffness.eigh(&mass, 16).0));
    }
    for errors in [&diagonal, &galerkin] {
        for w in errors.windows(2) {
            assert!(w[1] < w[0] / 3.0, "{errors:?}");
        }
    }
}
