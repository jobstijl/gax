//! Whitney forms of every degree (`Topology::whitney`, `whitney_d`, `Geometry::form_inner`) on
//! `PyDEC`'s meshes: they invert the de Rham map (`R ∘ W = id`), their derivative is the Whitney
//! form of the coboundary (`d W = W d`) and obeys Stokes's theorem face by face, the metric
//! pairs them into `PyDEC`'s Whitney inner products (and gax's Galerkin star), and the
//! 1-forms are `Geometry::whitney_1`'s fields.

#![cfg(all(feature = "vga2d", feature = "vga3d"))]

use gax::dec::{Form, Geometry};
use gax::field::Field;
use gax::mesh::{Mesh, Topology};
use gax::{vga2d, vga3d};
use std::collections::BTreeMap;

type S = vga3d::Scalar<(), f64>;

/// A mesh of the fixtures: its points and top simplices, and `PyDEC`'s Whitney inner products
/// per degree, by the sorted vertices of the two simplices.
/// Entries of a matrix over simplices, by the sorted vertices of its row and column simplex.
type Entries = BTreeMap<(Vec<usize>, Vec<usize>), f64>;

struct Fixture {
    name: String,
    dim: usize,
    points: Vec<Vec<f64>>,
    tops: Vec<Vec<usize>>,
    whitney: Vec<Entries>,
}

fn fixtures() -> Vec<Fixture> {
    let ints =
        |l: &str| -> Vec<usize> { l.split_whitespace().map(|x| x.parse().unwrap()).collect() };
    let mut out = Vec::new();
    let mut lines = include_str!("fixtures/pydec.txt").lines();
    while let Some(head) = lines.next() {
        let w: Vec<&str> = head.split_whitespace().collect();
        let (dim, nv, ns): (usize, usize, usize) = (
            w[2].parse().unwrap(),
            w[3].parse().unwrap(),
            w[4].parse().unwrap(),
        );
        let points = (0..nv)
            .map(|_| {
                lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect()
            })
            .collect();
        let tops = (0..ns).map(|_| ints(lines.next().unwrap())).collect();
        for _ in 0..=dim {
            let count = ints(lines.next().unwrap().trim_start_matches("dimension"))[1];
            for _ in 0..count {
                lines.next();
            }
        }
        out.push(Fixture {
            name: w[1].to_string(),
            dim,
            points,
            tops,
            whitney: vec![BTreeMap::new(); dim + 1],
        });
    }
    let mut lines = include_str!("fixtures/pydec_whitney.txt").lines();
    while let Some(head) = lines.next() {
        let w: Vec<&str> = head.split_whitespace().collect();
        let fx = out.iter_mut().find(|f| f.name == w[1]).unwrap();
        let (k, count): (usize, usize) = (w[2].parse().unwrap(), w[3].parse().unwrap());
        for _ in 0..count {
            let parts: Vec<&str> = lines.next().unwrap().split(';').collect();
            fx.whitney[k].insert(
                (ints(parts[0]), ints(parts[1])),
                parts[2].trim().parse().unwrap(),
            );
        }
    }
    out
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

/// A cochain of degree `k` with arbitrary values.
fn cochain<const N: usize>(c: &Topology<N>, k: usize, seed: f64) -> Field<S> {
    (0..c.len(k))
        .map(|i| S::new((seed * (i as f64 + 1.0) + k as f64).sin()))
        .collect()
}

/// The `k`-faces of top simplex `t` as local vertices, in their own orientation (sorted global
/// vertices below the top, the top as given), with their indices.
fn faces<const N: usize>(c: &Topology<N>, k: usize, t: usize) -> Vec<(Vec<usize>, usize)> {
    let top = c.simplex(N - 1, t);
    if k + 1 == N {
        return vec![((0..N).collect(), t)];
    }
    let mut out = Vec::new();
    for mask in 0u32..(1 << N) {
        if mask.count_ones() as usize != k + 1 {
            continue;
        }
        let mut local: Vec<usize> = (0..N).filter(|&i| mask & (1 << i) != 0).collect();
        local.sort_by_key(|&i| top[i]);
        let global: Vec<usize> = local.iter().map(|&i| top[i]).collect();
        out.push((local, c.index_of(k, &global).unwrap()));
    }
    out
}

/// The centroid of a face of a top simplex, as barycentric coordinates of the top.
fn centroid<const N: usize>(face: &[usize]) -> [f64; N] {
    let mut at = [0.0; N];
    for &v in face {
        at[v] = 1.0 / face.len() as f64;
    }
    at
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|x| x as f64).product()
}

/// `R ∘ W = id`, `d W = W d` and Stokes's theorem on every top simplex, every degree.
fn check_topology<const N: usize>(name: &str, c: &Topology<N>) {
    let n = N - 1;
    for k in 0..=n {
        let x = cochain(c, k, 0.37);
        let dx: Field<S> = if k < n {
            &c.d::<S>(k) * &x
        } else {
            Field::zeros(0)
        };
        for t in 0..c.len(n) {
            for (face, index) in faces(c, k, t) {
                // The form integrated over each k-face at its centroid (it is linear there).
                let w = c.whitney(k, &x, t, &centroid::<N>(&face));
                let integral = w.on_face(&face).s() / factorial(k);
                assert!(
                    close(integral, x.site(index).s()),
                    "{name}: R W ≠ id, k = {k}, top {t}, face {face:?}: {integral}"
                );
            }
            if k == n {
                assert_eq!(c.whitney_d(k, &x, t).coefficients(), &[]);
                continue;
            }
            // d W x = W d x, at any point (both are constant).
            let d = c.whitney_d(k, &x, t);
            let total = (N * (N + 1) / 2) as f64;
            let point: [f64; N] = core::array::from_fn(|i| (i + 1) as f64 / total);
            let w = c.whitney(k + 1, &dx, t, &point);
            for (a, b) in d.coefficients().iter().zip(w.coefficients()) {
                assert!(close(a.s(), b.s()), "{name}: d W ≠ W d, k = {k}, top {t}");
            }
            // Stokes on each (k+1)-face: ∫_τ d W x = Σ (−1)ⁱ ∫_{∂ᵢτ} W x, through the forms alone.
            for (tau, _) in faces(c, k + 1, t) {
                let inside = d.on_face(&tau).s() / factorial(k + 1);
                let boundary: f64 = (0..=k + 1)
                    .map(|i| {
                        let side: Vec<usize> = tau
                            .iter()
                            .enumerate()
                            .filter(|&(j, _)| j != i)
                            .map(|(_, &v)| v)
                            .collect();
                        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                        sign * c
                            .whitney(k, &x, t, &centroid::<N>(&side))
                            .on_face(&side)
                            .s()
                            / factorial(k)
                    })
                    .sum();
                assert!(
                    close(inside, boundary),
                    "{name}: Stokes, k = {k}, top {t}, face {tau:?}"
                );
            }
        }
    }
}

/// The Whitney inner products `∫ ⟨W_σ, W_ρ⟩`, from the forms at the top's vertices (they are
/// linear: `∫ λₐ λ_b = (1 + δₐ_b) V / ((n+1)(n+2))`) paired by the metric.
fn mass_from_forms<const N: usize>(c: &Topology<N>, g: &Geometry<f64, N>, k: usize) -> Entries {
    let n = N - 1;
    let mut out = BTreeMap::new();
    for t in 0..c.len(n) {
        let vol = g.volumes(n)[t];
        let basis: Vec<(usize, Vec<Form<S>>)> = faces(c, k, t)
            .into_iter()
            .map(|(_, index)| {
                let one: Field<S> = (0..c.len(k))
                    .map(|i| S::new(if i == index { 1.0 } else { 0.0 }))
                    .collect();
                let at_vertices = (0..N)
                    .map(|a| c.whitney(k, &one, t, &centroid::<N>(&[a])))
                    .collect();
                (index, at_vertices)
            })
            .collect();
        for (i, wi) in &basis {
            for (j, wj) in &basis {
                let mut sum = 0.0;
                for (a, fa) in wi.iter().enumerate() {
                    for (b, fb) in wj.iter().enumerate() {
                        let weight =
                            if a == b { 2.0 } else { 1.0 } * vol / ((n + 1) * (n + 2)) as f64;
                        sum += weight * g.form_inner(t, fa, fb);
                    }
                }
                let key = |s: usize| {
                    let mut v = c.simplex(k, s).to_vec();
                    v.sort_unstable();
                    v
                };
                *out.entry((key(*i), key(*j))).or_insert(0.0) += sum;
            }
        }
    }
    out
}

fn check_metric<const N: usize>(fx: &Fixture, c: &Topology<N>, g: &Geometry<f64, N>) {
    for k in 0..N {
        let ours = mass_from_forms(c, g, k);
        let pydec = &fx.whitney[k];
        // gax's Galerkin star, as the same map.
        let galerkin: BTreeMap<(Vec<usize>, Vec<usize>), f64> =
            g.mass::<S>(k)
                .couplings()
                .fold(BTreeMap::new(), |mut m, (r, col, x)| {
                    let key = |s: usize| {
                        let mut v = c.simplex(k, s).to_vec();
                        v.sort_unstable();
                        v
                    };
                    *m.entry((key(r), key(col))).or_insert(0.0) += x.s();
                    m
                });
        for (key, &x) in &ours {
            let want = pydec.get(key).copied().unwrap_or(0.0);
            assert!(
                close(x, want),
                "{} k = {k} {key:?}: forms {x}, PyDEC {want}",
                fx.name
            );
            let same = galerkin.get(key).copied().unwrap_or(0.0);
            assert!(
                close(x, same),
                "{} k = {k} {key:?}: forms {x}, mass {same}",
                fx.name
            );
        }
        for (key, &want) in pydec {
            assert!(
                ours.contains_key(key) || want.abs() < 1e-14,
                "{} k = {k}: {key:?} missing",
                fx.name
            );
        }
    }
}

#[test]
fn whitney_forms_match_de_rham_stokes_and_pydec() {
    for fx in fixtures() {
        if fx.dim == 2 {
            let v: Field<_> = fx
                .points
                .iter()
                .map(|p| vga2d::Vector::<(), f64>::new(p[0], p[1]))
                .collect();
            let tops: Vec<[usize; 3]> = fx.tops.iter().map(|s| [s[0], s[1], s[2]]).collect();
            let mesh = Mesh::new(v, &tops).unwrap();
            check_topology(&fx.name, mesh.topology());
            check_metric(&fx, mesh.topology(), &Geometry::new(&mesh));
        } else {
            let v: Field<_> = fx
                .points
                .iter()
                .map(|p| vga3d::Vector::<(), f64>::new(p[0], p[1], p[2]))
                .collect();
            let tops: Vec<[usize; 4]> = fx.tops.iter().map(|s| [s[0], s[1], s[2], s[3]]).collect();
            let mesh = Mesh::new(v, &tops).unwrap();
            check_topology(&fx.name, mesh.topology());
            check_metric(&fx, mesh.topology(), &Geometry::new(&mesh));
        }
    }
}

/// The 1-forms are `whitney_1`'s vector fields, read on the edges; and a cochain of vectors
/// gives the vector-valued form whose components are the forms of the components.
#[test]
fn one_forms_are_whitney_1_and_values_may_be_vectors() {
    use vga3d::Vector;
    let fx = fixtures().into_iter().find(|f| f.dim == 3).unwrap();
    let v: Field<_> = fx
        .points
        .iter()
        .map(|p| Vector::<(), f64>::new(p[0], p[1], p[2]))
        .collect();
    let tops: Vec<[usize; 4]> = fx.tops.iter().map(|s| [s[0], s[1], s[2], s[3]]).collect();
    let mesh = Mesh::new(v, &tops).unwrap();
    let (c, g) = (mesh.topology(), Geometry::new(&mesh));
    let x = cochain(c, 1, 0.53);
    let at = [0.1, 0.2, 0.3, 0.4];
    for t in 0..c.len(3) {
        let s = c.simplex(3, t);
        let field = g.whitney_1(&mesh, &x, t, &at);
        let form = c.whitney(1, &x, t, &at);
        for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
            let along = mesh.vertices.site(s[b]) - mesh.vertices.site(s[a]);
            assert!(close(
                field.scalar_product(along).s(),
                form.on_face(&[a, b]).s()
            ));
        }
    }
    let y = cochain(c, 2, 0.91);
    let vectors: Field<Vector<(), f64>> = (0..c.len(2))
        .map(|i| Vector::new(x.site(i % c.len(1)).s(), y.site(i).s(), -1.0))
        .collect();
    let first: Field<S> = (0..c.len(2)).map(|i| x.site(i % c.len(1))).collect();
    let w = c.whitney(2, &vectors, 5, &at);
    let (w0, w1) = (c.whitney(2, &first, 5, &at), c.whitney(2, &y, 5, &at));
    for ((v, a), b) in w
        .coefficients()
        .iter()
        .zip(w0.coefficients())
        .zip(w1.coefficients())
    {
        assert!(close(v.c[0], a.s()) && close(v.c[1], b.s()));
    }
}
