//! numga's tests of the Dirac–Kähler fields, and its scenario checks.

use super::dirac::*;
use super::scenes::*;
use gax::field::Field;

/// The largest size of a list of values, each the norm of its coefficients.
fn largest<M: gax::Extensor<Coef = f64>>(values: impl IntoIterator<Item = M>) -> f64 {
    values
        .into_iter()
        .map(|m| Field::new(vec![m]).norm())
        .fold(0.0, f64::max)
}

/// The largest size of the difference of two fields.
fn apart<M>(a: &Field<M>, b: &Field<M>) -> f64
where
    M: gax::Extensor<Coef = f64> + core::ops::Sub<Output = M>,
{
    largest(a.iter().zip(b.iter()).map(|(x, y)| x - y))
}

/// A field on the vertices from its points' coordinates.
fn on_vertices<M: gax::Extensor>(vertices: &Field<Point>, f: impl Fn(f64, f64) -> M) -> Field<M> {
    vertices.map(|p| {
        let [x, y] = coordinates(p);
        f(x, y)
    })
}

/// The grade identities of the Dirac–Kähler product on a disk: with the vertex multivector
/// `φ + u + ω`, its face product's grades against the corner gradients' inner and outer
/// products, and the mixed Whitney reconstruction's grades.
fn check_grades(divisions: usize, fields: &Fields) {
    let (mesh, vertices) = disk(1.0, divisions);
    let ops = Operators::new(&mesh, &vertices);
    let grades = grades(&ops, &mesh, &vertices, fields);
    let at = crate::simplicial::incidence(&mesh, 2, 0);
    let u = on_vertices(&vertices, |x, y| fields.vector(x, y));
    let omega = on_vertices(&vertices, |x, y| Bivector::new(fields.bivector(x, y)));
    let terms = |f: usize, part: &dyn Fn(Plane, Plane, Bivector) -> Full| {
        (0..3)
            .map(|k| {
                let v = at.sites[f][k];
                part(ops.corner_gradients[k].site(f), u.site(v), omega.site(v))
            })
            .reduce(|a, b| a + b)
            .expect("three corners")
    };
    for f in 0..mesh.len(2) {
        let product = grades.face_dirac.site(f);
        let divergence = terms(f, &|g, u, _| Full::new((g | u).s(), 0.0, 0.0, 0.0));
        let turned = terms(f, &|g, _, w| {
            let v = g.lc(w);
            Full::new(0.0, v.e1(), v.e2(), 0.0)
        });
        let curl = terms(f, &|g, u, _| Full::new(0.0, 0.0, 0.0, (g ^ u).e12()));
        let gradient = grades.scalar_gradient.site(f);
        let grade1 = Full::new(0.0, gradient.e1(), gradient.e2(), 0.0) + turned;
        assert!(largest([Full::new(product.s(), 0.0, 0.0, 0.0) - divergence]) < 1e-14);
        assert!(largest([Full::new(0.0, product.e1(), product.e2(), 0.0) - grade1]) < 1e-14);
        assert!(largest([Full::new(0.0, 0.0, 0.0, product.e12()) - curl]) < 1e-14);
        let even = grades.even_gradient.site(f);
        assert!(largest([Full::new(0.0, even.e1(), even.e2(), 0.0) - grade1]) < 1e-14);
        let whitney = grades.whitney.site(f);
        assert!((whitney.e12() - product.e12()).abs() < 1e-13);
    }
}

#[test]
fn mixed_grade_whitney_reconstruction_and_dirac_kahler_product_on_disk() {
    let fields = Fields {
        scalar: [0.6, -0.6, 0.4, -0.25, 0.0],
        vector: [[0.7, -0.5], [0.4, 0.9]],
        bivector: [0.8, -0.3, 0.0],
    };
    check_grades(5, &fields);
    // The mixed reconstruction holds each cochain's form in its own grade.
    let (mesh, vertices) = disk(1.0, 5);
    let ops = Operators::new(&mesh, &vertices);
    let grades = grades(&ops, &mesh, &vertices, &fields);
    let scalar = ops.whitney0.of(&grades.potential);
    let forms = ops.whitney1.of(&grades.edge_cochain);
    for f in 0..mesh.len(2) {
        let w = grades.whitney.site(f);
        assert!((w.s() - scalar.site(f).s()).abs() < 1e-15);
        assert!((w.e1() - forms.site(f).e1()).abs() + (w.e2() - forms.site(f).e2()).abs() < 1e-15);
    }
}

#[test]
fn interior_decoupling_and_hodge_helmholtz_decomposition_of_a_velocity_on_annulus() {
    let a = annulus(0.32, 1.0, 7);
    let ops = Operators::new(&a.mesh, &a.vertices);
    assert_eq!(a.mesh.euler_characteristic(), 0);

    // On every interior vertex star a stream velocity has no net divergence and a potential
    // velocity no net circulation.
    let potential = on_vertices(&a.vertices, |x, y| {
        Scalar::new(x * x - 0.4 * x * y + 0.7 * y)
    });
    let stream = on_vertices(&a.vertices, |x, y| {
        Scalar::new(0.9 * x * x + 0.5 * y * y - 0.3 * x)
    });
    let divergence = ops.gradient.adjugate().of(&ops.stream_velocity.of(&stream));
    let circulation = ops.stream_velocity.adjugate().of(&ops
        .isotropic_resistance
        .of(&ops.potential_velocity.of(&potential)));
    for &v in &a.interior {
        assert!(divergence.site(v).e12().abs() < 1e-13);
        assert!(circulation.site(v).e12().abs() < 1e-13);
    }

    // The harmonic stream function has no interior stream Laplacian.
    let (harmonic_stream, harmonic) = harmonic_circulation(&ops, &a.inner_ring, &a.interior);
    let laplacian = ops.stream_laplacian.of(&harmonic_stream);
    assert!(
        a.interior
            .iter()
            .all(|&v| laplacian.site(v).e12().abs() < 1e-13)
    );

    // The potential, stream and harmonic parts of a velocity on the perforated disk.
    let truth = Truth::new(&a, 0.32, [1.9, -1.1], [1.4, 1.7]);
    let total = truth.flow(&ops, &harmonic, 0.7);
    let parts = hodge_decomposition(&ops, &total, &a.interior);
    assert!(apart(&parts.potential, &truth.potential) < 1e-12);
    assert!(apart(&parts.stream, &truth.stream) < 1e-12);
    assert!(apart(&parts.harmonic_flow, &harmonic.scaled(0.7)) < 1e-12);
    let sum = &(&parts.potential_flow + &parts.stream_flow) + &parts.harmonic_flow;
    assert!(apart(&sum, &total) < 1e-14);
    check_orthogonal(&ops, &parts);
}

/// The three parts are orthogonal in the energy: each one's resistance paired with another.
fn check_orthogonal(ops: &Operators, parts: &Decomposition) {
    let pair = |a: &Field<Point>, b: &Field<Point>| {
        ops.isotropic_resistance
            .of(a)
            .zip_with(b, |r, v| r & v)
            .sum()
            .s()
    };
    assert!(pair(&parts.potential_flow, &parts.stream_flow).abs() < 1e-12);
    assert!(pair(&parts.potential_flow, &parts.harmonic_flow).abs() < 1e-12);
    assert!(pair(&parts.stream_flow, &parts.harmonic_flow).abs() < 1e-12);
}

/// The true potential and stream function of numga's velocity: a bubble that vanishes on both
/// rims, times a linear function each.
struct Truth {
    potential: Field<Scalar>,
    stream: Field<Scalar>,
}

impl Truth {
    fn new(a: &Annulus, hole: f64, potential: [f64; 2], stream: [f64; 2]) -> Truth {
        let bubble = |x: f64, y: f64| {
            let r2 = x * x + y * y;
            (r2 - hole * hole) * (1.0 - r2)
        };
        Truth {
            potential: on_vertices(&a.vertices, |x, y| {
                Scalar::new(bubble(x, y) * (potential[0] * x + potential[1] * y))
            }),
            stream: on_vertices(&a.vertices, |x, y| {
                Scalar::new(bubble(x, y) * (stream[0] * x + stream[1] * y))
            }),
        }
    }

    /// The velocity of the potential, the stream function, and the harmonic circulation.
    fn flow(&self, ops: &Operators, harmonic: &Field<Point>, strength: f64) -> Field<Point> {
        let parts =
            &ops.potential_velocity.of(&self.potential) + &ops.stream_velocity.of(&self.stream);
        &parts + &harmonic.scaled(strength)
    }
}

#[test]
fn skew_adjoint_dirac_kahler_wave_operators_and_boundary_mode_conversion() {
    let a = annulus(0.32, 1.0, 6);
    let ops = Operators::new(&a.mesh, &a.vertices);

    // The power of the even divergence cancels that of the even gradient exactly.
    let test_even = on_vertices(&a.vertices, |x, y| {
        Even::new(0.8 * x - 0.5 * y, 0.6 * x * y - 0.2 * y)
    });
    let other = on_vertices(&a.vertices, |x, y| {
        Even::new(0.3 * x + 0.9 * y, -0.4 * x + 0.7 * y)
    });
    let test_field = ops.even_gradient.of(&other);
    let back = ops.even_divergence.of(&test_field);
    let power_even = ops
        .vertex_areas
        .zip_with(
            &back.zip_with(&test_even, |d, e| d.reverse().scalar_product(e)),
            |a, p| a * p,
        )
        .sum();
    let ahead = ops.even_gradient.of(&test_even);
    let power_field = ops
        .areas
        .zip_with(
            &test_field.zip_with(&ahead, |f, g| f.scalar_product(g)),
            |a, p| a * p,
        )
        .sum();
    assert!((power_even + power_field).s().abs() < 1e-14);

    // A compressional pulse (grade 0 alone) converts partly into vorticity (grade 2) at the
    // rims, keeping its energy.
    let states = pulse(&ops, &a.vertices, 0.16, 0.01, 45);
    let vortical = |even: &Field<Even>| {
        ops.vertex_areas
            .zip_with(even, |a, e| a * (e.e12() * e.e12()))
            .sum()
            .s()
            * 0.5
    };
    let energies: Vec<f64> = states.iter().map(|(e, f)| energy(&ops, e, f)).collect();
    assert!(vortical(&states[0].0).abs() < 1e-16);
    assert!(vortical(&states.last().expect("states").0) > 1e-3);
    assert!(
        energies
            .iter()
            .all(|e| (e / energies[0] - 1.0).abs() < 1e-4)
    );
}

/// numga's scenario checks on its own sizes.
#[test]
fn the_scenes_pass_numgas_checks() {
    check_grades(DISK_DIVISIONS, &GRADES);
    let a = annulus(INNER_RADIUS, OUTER_RADIUS, ANNULUS_RINGS);
    let ops = Operators::new(&a.mesh, &a.vertices);
    let (_, harmonic) = harmonic_circulation(&ops, &a.inner_ring, &a.interior);
    let truth = Truth::new(&a, INNER_RADIUS, [-2.2, 0.9], [1.6, 1.9]);
    let total = truth.flow(&ops, &harmonic, CIRCULATION);
    let parts = hodge_decomposition(&ops, &total, &a.interior);
    assert!(apart(&parts.potential, &truth.potential) < 1e-12);
    assert!(apart(&parts.stream, &truth.stream) < 1e-12);
    assert!(apart(&parts.harmonic_flow, &harmonic.scaled(CIRCULATION)) < 1e-12);
    check_orthogonal(&ops, &parts);
    let states = pulse(&ops, &a.vertices, PULSE_WIDTH, DT, STEPS);
    let first = energy(&ops, &states[0].0, &states[0].1);
    assert!(
        states
            .iter()
            .all(|(e, f)| (energy(&ops, e, f) / first - 1.0).abs() < 1e-4)
    );
}

#[test]
fn every_figure_draws() {
    let shown = super::shown();
    for t in [1.0, 4.0, 8.0] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
