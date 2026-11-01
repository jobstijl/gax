//! numga's tests of Regge calculus, and its scenario checks.

use super::regge::*;
use crate::simplicial::{edges as edge_list, exterior, interior};
use gax::field::{Field, SparseMap};
use gax::mesh::Topology;

/// The size of a value: the norm of its coefficients.
fn size<M: gax::Extensor<Coef = f64>>(m: M) -> f64 {
    Field::new(vec![m]).norm()
}

/// The point-to-plane map of the Euclidean metric: the inverse of the dual (numga's
/// `Point.dual_inverse()`).
fn euclidean() -> Metric {
    Point::slot().hodge_inverse()
}

#[test]
fn regge_metric_basis_side_readings_heron_and_corner_rotors() {
    let corners = single_triangle();
    let sides = sides(corners);
    let (basis, gradients, patch) = regge_metric_basis(corners);
    assert!(size(gradients[0] + gradients[1] + gradients[2]) < 1e-14);

    // Each basis map reads one on its own side and nothing on the other two.
    for (k, b) in basis.iter().enumerate() {
        for (q, &s) in sides.iter().enumerate() {
            let reading = (b.of(s) & s).s();
            assert!((reading - f64::from(u8::from(k == q))).abs() < 1e-14);
        }
    }

    // With the Euclidean squared lengths the metric is the Euclidean one and the area the
    // reference area.
    let lengths = sides.map(|s| s.norm_squared());
    let (metric, area, rotors, turns) = triangle_geometry(corners, lengths);
    assert!(size(metric - euclidean()) < 1e-14);
    assert!((area - patch.s().abs()).abs() < 1e-14);
    let half_turn = Bivector::new(core::f64::consts::PI);
    assert!(size(turns[0] + turns[1] + turns[2] - half_turn) < 1e-14);
    assert!(size(rotors[0] * rotors[1] * rotors[2] - Rotor::new(-1.0, 0.0)) < 1e-14);

    // Stretched, the metric reads the squared lengths, its area is Heron's, and the corner
    // rotors still turn through a half turn together.
    let squared: [f64; 3] = core::array::from_fn(|k| lengths[k] * EDGE_STRETCHES[k]);
    let (metric, area, rotors, turns) = triangle_geometry(corners, squared);
    for (k, &s) in sides.iter().enumerate() {
        assert!(((metric.of(s) & s).s() - squared[k]).abs() < 1e-14);
    }
    #[allow(clippy::disallowed_methods)] // Heron's formula, the test's reference
    let heron = {
        let l = squared.map(f64::sqrt);
        let s = (l[0] + l[1] + l[2]) / 2.0;
        (s * (s - l[0]) * (s - l[1]) * (s - l[2])).sqrt()
    };
    assert!((area - heron).abs() < 1e-14);
    assert!(
        rotors
            .iter()
            .all(|r| (r.norm_squared() - 1.0).abs() < 1e-14)
    );
    assert!(size(turns[0] + turns[1] + turns[2] - half_turn) < 1e-14);
    assert!(size(rotors[0] * rotors[1] * rotors[2] - Rotor::new(-1.0, 0.0)) < 1e-14);
}

/// The metric's intrinsic geometry for each of the three cases on a disk.
fn cases(mesh: &Topology<3>, vertices: &Field<Point>) -> Vec<Intrinsic> {
    (0..3)
        .map(|case| mesh_geometry(mesh, vertices, &stretched(mesh, vertices, case)))
        .collect()
}

/// numga's Gauss–Bonnet, flatness, curvature and inverse-metric checks.
fn check_geometry(mesh: &Topology<3>, geometry: &[Intrinsic], centre_tolerance: Option<f64>) {
    let inside = interior(mesh, 0);
    for (case, g) in geometry.iter().enumerate() {
        // Discrete Gauss–Bonnet: the defects add up to a whole turn, whatever the metric.
        let total = g.defects.sum();
        assert!(
            (total.e12() - core::f64::consts::TAU).abs() < 1e-12,
            "{case}"
        );
        // The side dyads with the corner cotangents are the area-weighted inverse metric.
        for f in 0..mesh.len(2) {
            let inverse = g.metrics[f].inverse() * g.areas.site(f).s();
            assert!(size(g.conductivities[f] - inverse) < 1e-13);
        }
    }
    let flat = &geometry[0];
    assert!(
        inside
            .iter()
            .all(|&v| flat.defects.site(v).e12().abs() < 1e-13)
    );
    assert!(
        inside
            .iter()
            .all(|&v| flat.curvatures.site(v).e12().abs() < 1e-11)
    );
    // Stretching squared radial and circumferential distances by one plus the rates times the
    // squared radius curves the surface by `radial − 3 hoop` at its centre: positive, then
    // negative.
    let centre: Vec<f64> = geometry
        .iter()
        .map(|g| g.curvatures.site(0).e12())
        .collect();
    assert!(centre[1] > 0.0 && centre[2] < 0.0);
    if let Some(rtol) = centre_tolerance {
        for case in 0..3 {
            let expected = RADIAL_RATES[case] - 3.0 * HOOP_RATES[case];
            assert!(
                (centre[case] - expected).abs() <= rtol * expected.abs() + 1e-12,
                "{centre:?}"
            );
        }
    }
}

#[test]
fn gauss_bonnet_curvature_defects_and_cotangent_inverse_metric() {
    let (mesh, vertices) = disk(1.0, 6);
    let geometry = cases(&mesh, &vertices);
    check_geometry(&mesh, &geometry, Some(4.5e-2));

    // Conformal invariance in the plane: scaling every squared length scales the areas and
    // keeps the defects and the area-weighted inverse metrics.
    let scale = 2.7;
    for (case, g) in geometry.iter().enumerate() {
        let scaled = stretched(&mesh, &vertices, case).map(|l| l * scale);
        let h = mesh_geometry(&mesh, &vertices, &scaled);
        for f in 0..mesh.len(2) {
            assert!((h.areas.site(f).s() - scale * g.areas.site(f).s()).abs() < 1e-13);
            assert!(size(h.conductivities[f] - g.conductivities[f]) < 1e-13);
        }
        assert!(
            (0..mesh.len(0)).all(|v| (h.defects.site(v) - g.defects.site(v)).e12().abs() < 1e-13)
        );
    }
}

/// The amounts a conductance couples each pair of vertices by.
fn couplings(m: &SparseMap<Amount<(Scalar,)>>) -> std::collections::BTreeMap<(usize, usize), f64> {
    let mut out = std::collections::BTreeMap::new();
    for (r, c, cell) in m.couplings() {
        *out.entry((r, c)).or_insert(0.0) += cell.of(Scalar::new(1.0)).e12();
    }
    out
}

/// The flux balance of a conduction: zero amount at every free vertex, and in total.
fn check_balance(mesh: &Topology<3>, vertices: &Field<Point>, g: &Intrinsic) -> Field<Point> {
    let (held, prescribed) = electrodes(mesh, vertices);
    let (potential, current, streamfunction) =
        intrinsic_conduction(mesh, vertices, &g.conductivities, &held, &prescribed);
    let source = conductance(mesh, vertices, &g.conductivities).of(&potential);
    let free = (0..vertices.len()).filter(|v| !held.contains(v));
    assert!(free.into_iter().all(|v| source.site(v).e12().abs() < 1e-11));
    assert!(source.sum().e12().abs() < 1e-11);
    assert!(streamfunction.iter().all(|s| s.s().is_finite()));
    current
}

#[test]
fn intrinsic_laplace_beltrami_cotangent_equivalence_and_flow() {
    let (mesh, vertices) = disk(1.0, 6);
    let geometry = cases(&mesh, &vertices);
    let quadratic = vertices.map(|p| {
        let [x, y] = coordinates(p);
        Scalar::new(x * x - 0.35 * y * y)
    });
    let d0 = mesh.d::<Scalar>(0).map_cells(|c| c * Scalar::slot());
    let at = crate::simplicial::incidence(&mesh, 2, 1);
    let mut currents = Vec::new();
    for g in &geometry {
        let c = conductance(&mesh, &vertices, &g.conductivities);
        let (a, b) = (couplings(&c), couplings(&c.adjugate()));
        let keys = a.keys().chain(b.keys());
        assert!(
            keys.into_iter()
                .all(|k| (a.get(k).unwrap_or(&0.0) - b.get(k).unwrap_or(&0.0)).abs() < 1e-13)
        );

        // Pulled back through the face gradient, the Regge inverse metric is the intrinsic
        // cotangent Laplacian, minus the difference operator's adjugate times the edges' half
        // cotangents (amounts, numga's `ones.dual()`) times the differences.
        let mut weights = vec![0.0; mesh.len(1)];
        for (f, cot) in g.cotangents.iter().enumerate() {
            for k in 0..3 {
                weights[at.sites[f][k]] += cot[k];
            }
        }
        let weights = SparseMap::from_diagonal(
            &weights
                .iter()
                .map(|&w| Amount::new(w) * Scalar::slot())
                .collect(),
        );
        let cotangent = d0.adjugate().compose(&weights.compose(&d0));
        let (lhs, rhs) = (c.of(&quadratic), cotangent.of(&quadratic));
        assert!((0..vertices.len()).all(|v| (lhs.site(v) + rhs.site(v)).e12().abs() < 1e-13));
        currents.push(check_balance(&mesh, &vertices, g));
    }

    // Shortening the distances around lowers the current through the centre; lengthening them
    // raises it.
    let centres = &mesh.means::<Scalar>(2) * &vertices;
    let nearest = (0..mesh.len(2))
        .min_by(|&a, &b| centres.site(a).norm().total_cmp(&centres.site(b).norm()))
        .expect("a face");
    let speeds: Vec<f64> = currents.iter().map(|c| c.site(nearest).norm()).collect();
    assert!(speeds[1] < speeds[0] && speeds[0] < speeds[2], "{speeds:?}");
}

/// numga's scenario checks on its own sizes.
#[test]
fn the_scenes_pass_numgas_checks() {
    let corners = single_triangle();
    let sides = sides(corners);
    let (basis, _, _) = regge_metric_basis(corners);
    for (k, b) in basis.iter().enumerate() {
        for (q, &s) in sides.iter().enumerate() {
            assert!(((b.of(s) & s).s() - f64::from(u8::from(k == q))).abs() < 1e-14);
        }
    }
    let (mesh, vertices) = disk(DISK_RADIUS, DISK_DIVISIONS);
    let geometry = cases(&mesh, &vertices);
    check_geometry(&mesh, &geometry, None);
    for g in &geometry {
        check_balance(&mesh, &vertices, g);
    }
    assert_eq!(
        exterior(&mesh, 0).len() + interior(&mesh, 0).len(),
        vertices.len()
    );
    assert_eq!(edge_list(&mesh).len(), mesh.len(1));
}

#[test]
fn every_figure_draws() {
    let shown = super::shown();
    for t in [1.0, 5.0, 9.0] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
