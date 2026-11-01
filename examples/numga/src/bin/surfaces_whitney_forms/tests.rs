//! numga's tests of the Whitney forms, its scenario checks, and the same forms from gax's own
//! Whitney maps and metric (`gax::mesh::Topology::whitney`, `gax::dec::Geometry`).

use super::scenes::*;
use super::whitney::*;
use crate::simplicial::parity;
use gax::dec::{Form, Geometry};
use gax::field::{Field, SparseMap};
use gax::mesh::corners;
use gax::mesh::{Mesh, Topology};
use gax::{vga2d, vga3d};
use std::collections::BTreeMap;

/// The largest size of a list of values, each the norm of its coefficients.
fn largest<M: gax::Extensor<Coef = f64>>(values: impl IntoIterator<Item = M>) -> f64 {
    values
        .into_iter()
        .map(|m| Field::new(vec![m]).norm())
        .fold(0.0, f64::max)
}

/// The largest coefficient of the difference of two fields.
fn apart<M>(a: &Field<M>, b: &Field<M>) -> f64
where
    M: gax::Extensor<Coef = f64> + core::ops::Sub<Output = M>,
{
    largest(a.iter().zip(b.iter()).map(|(x, y)| x - y))
}

/// The number a conductance couples each pair of sites by: its amount for a unit potential.
fn couplings(m: &SparseMap<AmountMap>) -> BTreeMap<(usize, usize), f64> {
    let mut out = BTreeMap::new();
    for (r, c, cell) in m.couplings() {
        *out.entry((r, c)).or_insert(0.0) += cell.of(Scalar::new(1.0)).e12();
    }
    out
}

/// The largest difference of two maps' couplings.
fn couplings_apart(a: &BTreeMap<(usize, usize), f64>, b: &BTreeMap<(usize, usize), f64>) -> f64 {
    let keys = a.keys().chain(b.keys());
    let at = |m: &BTreeMap<(usize, usize), f64>, k| m.get(k).copied().unwrap_or(0.0);
    keys.map(|k| (at(a, k) - at(b, k)).abs())
        .fold(0.0, f64::max)
}

/// A field on the vertices from its points' coordinates.
fn on_vertices(vertices: &Field<Point>, f: impl Fn(f64, f64) -> f64) -> Field<Scalar> {
    vertices.map(|p| {
        let [x, y] = coordinates(p);
        Scalar::new(f(x, y))
    })
}

/// The indicator of one site among `n`.
fn unit(n: usize, i: usize) -> Field<Scalar> {
    Field::from_fn(n, |k| Scalar::new(if k == i { 1.0 } else { 0.0 }))
}

#[test]
fn barycentric_hats_and_edge_whitney_forms_on_single_triangle_and_diamond() {
    let (corners, points) = single_triangle(8);
    let (hats, gradients) = barycentric_hats(corners, &points);
    let (corner_hats, _) = barycentric_hats(corners, &corners);
    for (i, row) in corner_hats.iter().enumerate() {
        for (j, h) in row.iter().enumerate() {
            assert!((h.s() - f64::from(u8::from(i == j))).abs() < 1e-14);
        }
    }
    for h in &hats {
        assert!(((h[0] + h[1] + h[2]).s() - 1.0).abs() < 1e-14);
    }
    assert!(largest([gradients[0] + gradients[1] + gradients[2]]) < 1e-14);

    // Barycentric interpolation reproduces any affine function and its gradient exactly.
    let true_gradient = Plane::new(1.4, -0.85);
    let value = |p: Point| 0.6 + (true_gradient & p).s();
    let corner_values = corners.map(value);
    for (p, h) in points.iter().zip(&hats) {
        let interpolated: f64 = (0..3).map(|k| corner_values[k] * h[k].s()).sum();
        assert!((interpolated - value(*p)).abs() < 1e-14);
    }
    let reconstructed = (0..3).map(|k| gradients[k] * corner_values[k]);
    let reconstructed = reconstructed.reduce(|a, b| a + b).expect("three corners");
    assert!(largest([reconstructed - true_gradient]) < 1e-14);

    // The edge 1-forms match Whitney's alternating formula and read one along their own edge.
    let forms = edge_whitney_forms(corners, &points);
    let around = complements(2, 0);
    for (s, form) in forms.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (around[k][0], around[k][1]);
            let alternating = gradients[b] * hats[s][a].s() - gradients[a] * hats[s][b].s();
            assert!(largest([form[k] - alternating]) < 1e-14);
        }
    }
    let sides = sides(corners);
    for fraction in [0.15, 0.5, 0.82] {
        let along: Vec<Point> = (0..3)
            .map(|q| corners[around[q][0]] * (1.0 - fraction) + corners[around[q][1]] * fraction)
            .collect();
        let forms = edge_whitney_forms(corners, &along);
        for (q, form) in forms.iter().enumerate() {
            for (k, w) in form.iter().enumerate() {
                let reading = (*w & sides[q]).s();
                assert!((reading - f64::from(u8::from(k == q))).abs() < 1e-14);
            }
        }
    }

    // Across the shared edge of a diamond the tangential readings match and the normal ones
    // jump.
    let (mesh, vertices, _) = edge_diamond(6);
    let diamond = Operators::new(&mesh, &vertices);
    let shared = interior_edge(&mesh);
    let [tail, head] = [0, 1].map(|end| vertices.site(mesh.simplex(1, shared)[end]));
    let shared_edge = head - tail;
    let circulation = unit(mesh.len(1), shared);
    let mut jump: f64 = 0.0;
    for k in 0..7 {
        let fraction = 0.1 + 0.8 * k as f64 / 6.0;
        let seam = Field::from_fn(2, |_| tail * (1.0 - fraction) + head * fraction);
        let seams = diamond.whitney1_at(&seam).of(&circulation);
        for f in 0..2 {
            assert!(((seams.site(f) & shared_edge).s() - 1.0).abs() < 1e-14);
        }
        let normal = (seams.site(0) - seams.site(1)) & point(1.0, 0.0);
        jump = jump.max(normal.s().abs());
    }
    assert!(jump > 0.5, "{jump}");
}

/// The one interior edge of a mesh of two triangles.
fn interior_edge(mesh: &Topology<3>) -> usize {
    let rim = mesh.exterior(1);
    (0..mesh.len(1))
        .find(|e| !rim.contains(e))
        .expect("a shared edge")
}

#[test]
fn tetrahedral_whitney_0_1_2_forms_and_opposite_edge_helicity() {
    let (mesh, vertices, points) = single_tetrahedron(6);
    let tet = tetrahedron(&mesh, &vertices, 0);
    for (i, &c) in tet.corners.iter().enumerate() {
        for (j, h) in tet.hats(c).iter().enumerate() {
            assert!((h.s() - f64::from(u8::from(i == j))).abs() < 1e-13);
        }
    }
    for &p in &points {
        assert!((tet.hats(p).iter().map(|h| h.s()).sum::<f64>() - 1.0).abs() < 1e-13);
    }
    assert!(largest([tet.gradients.iter().copied().reduce(|a, b| a + b).unwrap()]) < 1e-13);
    assert!(
        largest([tet
            .face_patches
            .iter()
            .copied()
            .reduce(|a, b| a + b)
            .unwrap()])
            < 1e-13
    );

    // Barycentric interpolation reproduces any affine potential in space and its gradient.
    let true_gradient = SpacePlane::new(1.3, -0.75, 0.55);
    let value = |p: SpacePoint| 0.4 + (true_gradient & p).s();
    for &p in &points {
        let h = tet.hats(p);
        let interpolated: f64 = (0..4).map(|k| value(tet.corners[k]) * h[k].s()).sum();
        assert!((interpolated - value(p)).abs() < 1e-13);
    }
    let reconstructed = (0..4).map(|k| tet.gradients[k] * value(tet.corners[k]));
    let reconstructed = reconstructed.reduce(|a, b| a + b).unwrap();
    assert!(largest([reconstructed - true_gradient]) < 1e-13);

    // The edge 1-forms match Whitney's formula, have constant curls, and read one along their
    // own edge only.
    let ends = corners(3, 1);
    let curls = tet.curls();
    for &p in &points {
        let (h, forms) = (tet.hats(p), tet.edge_forms(p));
        for (e, end) in ends.iter().enumerate() {
            let (t, s) = (end[0], end[1]);
            let alternating = tet.gradients[s] * h[t].s() - tet.gradients[t] * h[s].s();
            assert!(largest([forms[e] - alternating]) < 1e-13);
        }
    }
    for (e, end) in ends.iter().enumerate() {
        let alternating = (tet.gradients[end[0]] ^ tet.gradients[end[1]]) * 2.0;
        assert!(largest([curls[e] - alternating]) < 1e-13);
    }
    let edges = &mesh.d::<vga3d::Scalar>(0) * &vertices;
    let own = mesh.oriented(3, 1).read_all(&edges);
    for fraction in [0.2, 0.5, 0.85] {
        for (e, end) in ends.iter().enumerate() {
            let at = tet.corners[end[0]] * (1.0 - fraction) + tet.corners[end[1]] * fraction;
            for (i, form) in tet.edge_forms(at).iter().enumerate() {
                let reading = (*form & own[e].site(0)).s();
                assert!((reading - f64::from(u8::from(i == e))).abs() < 1e-13);
            }
        }
    }

    // The face 2-forms match Whitney's formula one grade up and read the flux through their own
    // face only.
    let around = complements(3, 0);
    for &p in &points {
        let (h, forms) = (tet.hats(p), tet.face_forms(p));
        let g = tet.gradients;
        for (k, a) in around.iter().enumerate() {
            let (f, s, t) = (a[0], a[1], a[2]);
            let alternating =
                ((g[s] ^ g[t]) * h[f].s() + (g[t] ^ g[f]) * h[s].s() + (g[f] ^ g[s]) * h[t].s())
                    * 2.0;
            assert!(largest([forms[k] - alternating]) < 1e-13);
        }
    }
    for q in 0..4 {
        let others = (0..4).filter(|&k| k != q).map(|k| tet.corners[k]);
        let centre = others.reduce(|a, b| a + b).unwrap() * (1.0 / 3.0);
        for (k, form) in tet.face_forms(centre).iter().enumerate() {
            let reading = (*form & tet.face_patches[q]).s();
            assert!((reading - f64::from(u8::from(k == q))).abs() < 1e-13);
        }
    }

    // The volume pairing of edge forms with curls is the same for a sheared tetrahedron, and
    // links only opposite edges, with the sense of the corners as the two edges list them.
    let helicity = tet.helicity();
    let deformed = tetrahedron(&mesh, &deformed_tetrahedron(), 0).helicity();
    for i in 0..6 {
        for j in 0..6 {
            assert!((helicity[i][j] - deformed[i][j]).s().abs() < 1e-13);
            let pair = [ends[i].clone(), ends[j].clone()].concat();
            let disjoint = ends[i].iter().all(|c| !ends[j].contains(c));
            let linking = if disjoint {
                f64::from(parity(&pair)) / 6.0
            } else {
                0.0
            };
            assert!((helicity[i][j].s() - linking).abs() < 1e-13, "{i} {j}");
        }
    }
}

#[test]
fn cochain_complex_and_exact_midpoint_quadrature_including_obtuse_triangles() {
    let (mesh, vertices) = disk(1.0, 5);
    let ops = Operators::new(&mesh, &vertices);
    assert!(
        (&ops.d1 * &ops.d0)
            .couplings()
            .all(|(_, _, c)| c.s() == 0.0)
    );

    // Whitney reconstruction commutes with the exterior derivative on 0- and 1-cochains.
    let potential = on_vertices(&vertices, |x, y| 0.9 * x - 1.3 * y);
    let steps = &ops.d0 * &potential;
    assert!(apart(&ops.whitney1.of(&steps), &ops.gradient.of(&potential)) < 1e-14);
    let cochain = &steps + &Field::from_fn(mesh.len(1), |e| Scalar::new(cos(e)));
    let circulations = &ops.d1 * &cochain;
    assert!(apart(&ops.curl.of(&cochain), &ops.whitney2.of(&circulations)) < 1e-12);
    let read = ops
        .patches
        .zip_with(&ops.whitney2.of(&circulations), |p, w| p & w);
    assert!(apart(&read, &circulations) < 1e-13);

    // The three-midpoint quadrature matches the interior degree-2 Gauss rule and pulls back to
    // the 0-form conductance.
    let [isotropic_material, fibred] = conductivities();
    let midpoints = ops.edge_conductance(fibred);
    let gauss = [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ];
    let weights = SparseMap::from_diagonal(&ops.areas.map(|a| fibred * (a.s() / 3.0)));
    let interior_rule = gauss.iter().map(|w| {
        let at = Field::from_fn(mesh.len(2), |f| {
            (0..3)
                .map(|k| ops.corners[k].site(f) * w[k])
                .reduce(|a, b| a + b)
                .unwrap()
        });
        let forms = ops.whitney1_at(&at);
        forms.adjugate().compose(&weights.compose(&forms))
    });
    let interior_rule = interior_rule.reduce(|a, b| &a + &b).unwrap();
    assert!(couplings_apart(&couplings(&midpoints), &couplings(&interior_rule)) < 1e-14);
    assert!(couplings_apart(&couplings(&midpoints), &couplings(&midpoints.adjugate())) < 1e-14);
    let conductance = ops.conductance(fibred);
    let quadratic = on_vertices(&vertices, |x, y| x * x - 0.4 * y * y);
    let pulled = ops
        .d0
        .map_cells(|c| c * Scalar::slot())
        .adjugate()
        .of(&midpoints.of(&(&ops.d0 * &quadratic)));
    assert!(apart(&pulled, &conductance.of(&quadratic)) < 1e-13);
    let _ = isotropic_material;

    // On the obtuse diamond circumcentric DEC's weight on the shared edge is negative, while
    // Whitney's 1-form mass is positive definite.
    let (obtuse, obtuse_vertices) = obtuse_diamond();
    let flat = Mesh::new(obtuse_vertices.map(|p| p.undual()), &triangles(&obtuse)).unwrap();
    let star = Geometry::new(&flat).star::<vga2d::Scalar>(1);
    let shared = interior_edge(&obtuse);
    let weight = star
        .couplings()
        .find(|&(r, _, _)| r == shared)
        .unwrap()
        .2
        .s();
    assert!(weight < -1.0, "{weight}");
    let obtuse_ops = Operators::new(&obtuse, &obtuse_vertices);
    let mass = obtuse_ops.edge_conductance(isotropic());
    for e in 0..obtuse.len(1) {
        let c = unit(obtuse.len(1), e);
        // The adjugate pairs the current before the gradient: the energy with a minus sign.
        let energy: f64 = mass
            .of(&c)
            .zip_with(&c, |a, c| a & c)
            .iter()
            .map(|s| -s.s())
            .sum();
        assert!(energy > 0.0);
    }
}

/// `cos(e)`, numga's arbitrary cochain on the edges, from the turn by `e` radians.
fn cos(e: usize) -> f64 {
    vga2d::Vector::from_angle(e as f64).e1()
}

/// `sin(e)`.
fn sin(e: usize) -> f64 {
    vga2d::Vector::from_angle(e as f64).e2()
}

/// The faces of a surface's topology.
fn triangles(mesh: &Topology<3>) -> Vec<[usize; 3]> {
    crate::simplicial::faces(mesh)
}

#[test]
fn isotropic_and_anisotropic_conduction_patch_test_and_flux_balance() {
    let (mesh, vertices) = disk(1.0, 6);
    let ops = Operators::new(&mesh, &vertices);
    let materials = conductivities();

    // The linear patch test: a linear potential held on the whole rim is recovered everywhere,
    // for any material.
    let rim = mesh.exterior(0);
    let applied = Plane::new(1.25, -0.6);
    let exact = vertices.map(|p| applied & p);
    let held = Field::from_fn(vertices.len(), |v| {
        if rim.contains(&v) {
            exact.site(v)
        } else {
            Scalar::new(0.0)
        }
    });
    for material in materials {
        let (potential, current, _) = ops.conduction(material, &rim, &held);
        assert!(apart(&potential, &exact) < 1e-13);
        let expected = -material.of(applied);
        assert!(current.iter().all(|c| largest([c - expected]) < 1e-12));
    }

    // Between the electrodes every free vertex balances its flux and the total current is
    // conserved.
    let (electrodes, prescribed) = electrodes(&mesh, &vertices);
    let free: Vec<usize> = (0..vertices.len())
        .filter(|v| !electrodes.contains(v))
        .collect();
    for material in materials {
        let (potential, _, streamfunction) = ops.conduction(material, &electrodes, &prescribed);
        let source = ops.conductance(material).of(&potential);
        assert!(free.iter().all(|&v| source.site(v).e12().abs() < 1e-11));
        assert!(source.sum().e12().abs() < 1e-11);
        assert!(streamfunction.iter().all(|s| s.s().is_finite()));
    }
}

/// numga's scenario checks on its own sizes.
#[test]
fn the_scenes_pass_numgas_checks() {
    let (mesh, vertices) = disk(DISK_RADIUS, DISK_DIVISIONS);
    let ops = Operators::new(&mesh, &vertices);
    let linear = on_vertices(&vertices, |x, y| 0.7 * x - 1.1 * y);
    assert!(largest((&ops.d1 * &(&ops.d0 * &linear)).iter()) < 1e-14);
    assert!(
        apart(
            &ops.whitney1.of(&(&ops.d0 * &linear)),
            &ops.gradient.of(&linear)
        ) < 1e-14
    );
    let cochain = &(&ops.d0 * &linear) + &Field::from_fn(mesh.len(1), |e| Scalar::new(sin(e)));
    assert!(
        apart(
            &ops.curl.of(&cochain),
            &ops.whitney2.of(&(&ops.d1 * &cochain))
        ) < 1e-12
    );
    let [_, fibred] = conductivities();
    let midpoints = ops.edge_conductance(fibred);
    let conductance = ops.conductance(fibred);
    let quadratic = on_vertices(&vertices, |x, y| x * x - 0.5 * y);
    let d0 = ops.d0.map_cells(|c| c * Scalar::slot());
    assert!(
        apart(
            &d0.adjugate().of(&midpoints.of(&(&ops.d0 * &quadratic))),
            &conductance.of(&quadratic)
        ) < 1e-13
    );
    assert!(
        couplings_apart(
            &couplings(&conductance),
            &couplings(&conductance.adjugate())
        ) < 1e-13
    );
    let (electrodes, prescribed) = electrodes(&mesh, &vertices);
    let (potential, _, _) = ops.conduction(fibred, &electrodes, &prescribed);
    let source = conductance.of(&potential);
    let free = (0..vertices.len()).filter(|v| !electrodes.contains(v));
    assert!(free.into_iter().all(|v| source.site(v).e12().abs() < 1e-11));
}

/// The 2-form of a face as a bivector: its coefficients on the barycentric differentials
/// `dλ_I`, each the wedge of the corners' gradients (`dλᵢ` is the gradient of corner `i`).
fn bivector2(form: &Form<vga2d::Scalar>, gradients: [Plane; 3]) -> Bivector {
    (gradients[1] ^ gradients[2]) * form.coefficients()[0].s()
}

/// A 1-form of space as a vector, and a 2-form as a bivector, from the corners' gradients.
fn vector3(form: &Form<vga3d::Scalar>, g: [SpacePlane; 4]) -> SpacePlane {
    let c = form.coefficients();
    g[1] * c[0].s() + g[2] * c[1].s() + g[3] * c[2].s()
}
fn bivector3(form: &Form<vga3d::Scalar>, g: [SpacePlane; 4]) -> SpaceBivector {
    let c = form.coefficients();
    (g[1] ^ g[2]) * c[0].s() + (g[1] ^ g[3]) * c[1].s() + (g[2] ^ g[3]) * c[2].s()
}

/// gax's own Whitney maps give the forms numga writes in the algebra: its 1-forms at the face
/// centroids, the derivative of an edge cochain, the forms on the tetrahedron of every degree,
/// and its metric pairs them into the Galerkin star, the isotropic conductance of the edges.
#[test]
fn gax_whitney_maps_give_the_same_forms() {
    let (mesh, vertices) = disk(1.0, 5);
    let ops = Operators::new(&mesh, &vertices);
    let flat = Mesh::new(vertices.map(|p| p.undual()), &triangles(&mesh)).unwrap();
    let geometry = Geometry::new(&flat);
    let cochain = Field::from_fn(mesh.len(1), |e| vga2d::Scalar::new(cos(3 * e) + 0.2));
    let centroid_forms = ops.whitney1.of(&cochain);
    let circulations = ops.curl.of(&cochain);
    for f in 0..mesh.len(2) {
        let at_centroid = geometry.whitney_1(&flat, &cochain, f, &[1.0 / 3.0; 3]);
        assert!(largest([at_centroid - centroid_forms.site(f)]) < 1e-12);
        let gradients = [0, 1, 2].map(|k| ops.corner_gradients[k].site(f));
        let curl = bivector2(&mesh.whitney_d(1, &cochain, f), gradients);
        assert!(largest([curl - circulations.site(f)]) < 1e-12);
    }
    // The midpoint rule of the isotropic material is the Galerkin star of the edges, and the
    // pairing puts the current before the gradient: the amounts are minus its couplings.
    let mass = geometry.mass::<vga2d::Scalar>(1);
    let mut galerkin = BTreeMap::new();
    for (r, c, x) in mass.couplings() {
        *galerkin.entry((r, c)).or_insert(0.0) -= x.s();
    }
    assert!(couplings_apart(&couplings(&ops.edge_conductance(isotropic())), &galerkin) < 1e-13);

    // On the tetrahedron, the forms of unit cochains at a point of it.
    let (tet_mesh, tet_vertices, points) = single_tetrahedron(5);
    let tet = tetrahedron(&tet_mesh, &tet_vertices, 0);
    let solid = Mesh::new(tet_vertices.map(|p| p.undual()), &[[0, 1, 2, 3]]).unwrap();
    let metric = Geometry::new(&solid);
    let facing = tet_mesh.facets(3);
    for &p in &points {
        let at = tet.hats(p).map(|h| h.s());
        let edge_forms = tet.edge_forms(p);
        let unit_forms: Vec<Form<vga3d::Scalar>> = (0..6)
            .map(|e| tet_mesh.whitney(1, &unit3(6, e), 0, &at))
            .collect();
        for e in 0..6 {
            assert!(largest([vector3(&unit_forms[e], tet.gradients) - edge_forms[e]]) < 1e-12);
            for k in 0..6 {
                let inner = metric.form_inner(0, &unit_forms[e], &unit_forms[k]);
                assert!((inner - (edge_forms[e] | edge_forms[k]).s()).abs() < 1e-11);
            }
        }
        let face_forms = tet.face_forms(p);
        for (k, form) in face_forms.iter().enumerate() {
            let (face, sign) = (facing.site(0, k), f64::from(facing.sign(0, k)));
            let whitney = tet_mesh.whitney(2, &unit3(4, face), 0, &at);
            assert!(largest([bivector3(&whitney, tet.gradients) * sign - *form]) < 1e-12);
        }
    }
}

/// The indicator of one site among `n`, in space's algebra.
fn unit3(n: usize, i: usize) -> Field<vga3d::Scalar> {
    Field::from_fn(n, |k| vga3d::Scalar::new(if k == i { 1.0 } else { 0.0 }))
}

#[test]
fn every_figure_draws() {
    let shown = super::shown();
    for t in [1.0, 5.0, 9.0, 13.0] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
