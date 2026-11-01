//! numga's tests of the projective readings, in both signatures, and of diffusion on the fixed
//! sphere under unequal vertex weights, with the scenario's checks of the affine slice.

use super::flow::*;
use gax::field::Field;
use gax::mesh::corners as local_corners;

/// numga's parametrized test of the readings and the quotient's derivative, for one algebra:
/// the corners `[1, 0, 1]`, `[0, 1, 1]`, `[−0.5, −0.2, 1]` (numga's antivector coefficients,
/// `yz zx xy` in VGA3D, `yw wx xy` in PGA2D, which are the same coefficients in gax's kinds).
macro_rules! readings_test {
    ($name:ident, $module:ident) => {
        #[test]
        fn $name() {
            use $module::*;
            let corners = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-0.5, -0.2, 1.0]]
                .map(|c| Point::from_coeffs(c));
            let scales = [0.35, 2.2, 1.1];
            let lifted = [0, 1, 2].map(|k| corners[k] * scales[k]);
            let r = reciprocals(lifted);
            let point = corners[0] * 0.2 + corners[1] * 0.35 + corners[2] * 0.45;
            let displacement = corners[1] - corners[0];
            let (hats, gradients) = hats(r, point);
            let forms = edge_forms(r, point);
            let ends = local_corners(2, 1);
            let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
            let zero = |c: &[f64]| c.iter().all(|x| x.abs() < 1e-12);
            assert!(near(hats.iter().map(|h| h.s()).sum(), 1.0));
            assert!(zero(&(gradients[0] + gradients[1] + gradients[2]).c));
            assert!(gradients.iter().all(|&g| near((g & point).s(), 0.0)));
            for e in 0..3 {
                let (t, h) = (ends[e][0], ends[e][1]);
                let alternating = gradients[h] * hats[t] - gradients[t] * hats[h];
                assert!(zero(&(forms[e] - alternating).c));
            }

            // A varying scale along a path adds radial velocity; neither the hats nor the edge
            // readings change.
            let (scale, scale_rate) = (2.6, -0.7);
            let other_point = point * scale;
            let other_displacement = displacement * scale + point * scale_rate;
            let (other_hats, _) = $module::hats(r, other_point);
            let other_forms = edge_forms(r, other_point);
            for k in 0..3 {
                assert!(near(other_hats[k].s(), hats[k].s()));
                let along = (other_forms[k] & other_displacement).s();
                assert!(near(along, (forms[k] & displacement).s()));
            }

            // Compensating the corner reciprocals keeps a chosen chart, and so the whole basis.
            let unweighted = reciprocals(corners);
            let chart = section(unweighted);
            for k in 0..3 {
                let compensated = r[k] * (chart & lifted[k]);
                assert!(zero(&(compensated - unweighted[k]).c));
            }
        }
    };
}

readings_test!(projective_readings_and_quotient_derivative_vga3d, sphere);
readings_test!(projective_readings_and_quotient_derivative_pga2d, plane);

/// The largest difference of two fields of numbers.
fn apart(a: &Field<sphere::Scalar>, b: &Field<sphere::Scalar>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).s().abs())
        .fold(0.0, f64::max)
}

#[test]
fn fixed_sphere_diffusion_with_unequal_vertex_scales() {
    let scene = SphereScene::new();
    let (mesh, vertices, q) = (&scene.mesh, &scene.vertices, &scene.quadrature);
    let reference = diffusion(mesh, vertices, q, isotropic()).potential;
    let common = diffusion(mesh, &vertices.map(|v| v * 3.2), q, isotropic()).potential;
    assert!(apart(&common, &reference) < 1e-11);

    let up = sphere::Plane::new(0.0, 0.0, 1.0);
    let height = vertices.map(|v| up & v);
    for radius in [0.4, 2.5] {
        let lifted = Field::from_fn(vertices.len(), |v| {
            vertices.site(v) * if v == 0 { radius } else { 1.0 }
        });
        let potential = diffusion(mesh, &lifted, q, isotropic()).potential;
        // The height is the known physical solution, an eigenfunction of the sphere's
        // Laplacian, to this coarse resolution.
        assert!(apart(&potential, &height) < 0.08);
        assert!(apart(&potential, &reference) > 1e-3);
    }
    let still = Quadrature {
        samples: q.samples.clone(),
        measures: q.measures.clone(),
        source: q.source.iter().map(|s| s.map(|x| x * 0.0)).collect(),
    };
    let equilibrium = diffusion(mesh, vertices, &still, isotropic()).potential;
    assert!(equilibrium.iter().all(|p| p.s().abs() < 1e-12));
}

#[test]
fn the_affine_slice_reads_hats_that_sum_to_one_and_a_spatial_current() {
    // The scenario's checks.
    let planar = Planar::new();
    for ((h, g), &p) in planar
        .hats
        .iter()
        .zip(&planar.gradients)
        .zip(&planar.points)
    {
        assert!((h.iter().map(|h| h.s()).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(g.iter().all(|&g| (g & p).s().abs() < 1e-12));
    }
    // The current is a direction: the line at infinity holds it.
    let infinity = plane::Plane::new(0.0, 0.0, 1.0);
    assert!(
        planar
            .current
            .iter()
            .all(|&c| (infinity & c).s().abs() < 1e-12)
    );
    // The corners take their values.
    for (k, &value) in CORNER_VALUES.iter().enumerate() {
        let at = planar_point(PLANAR_CORNERS[k]);
        let i = planar
            .points
            .iter()
            .position(|&p| (p - at).ideal_norm() < 1e-12);
        assert!((planar.potential[i.expect("a lattice node at the corner")] - value).abs() < 1e-12);
    }
}

#[test]
fn the_frames_keep_the_physical_quadrature_and_move_one_vertex() {
    let scene = SphereScene::new();
    assert_eq!((scene.mesh.len(0), scene.mesh.len(2)), (42, 80));
    assert_eq!(scene.quadrature.samples.len(), QUADRATURE_DIVISIONS.pow(2));
    // The measures add up to the sphere's area.
    let area: f64 = scene
        .quadrature
        .measures
        .iter()
        .flat_map(|m| m.iter())
        .map(|m| m.s())
        .sum();
    assert!((area - 4.0 * core::f64::consts::PI).abs() < 0.05, "{area}");
    let rest = scene.frame(&scene.scales(0.0));
    let swung = scene.frame(&scene.scales(core::f64::consts::FRAC_PI_2));
    let [x, _, _] = coordinates(swung.lifted.site(scene.chosen));
    let [x0, _, _] = coordinates(scene.vertices.site(scene.chosen));
    let radius = LOG_RADIUS_AMPLITUDE.exp();
    assert!((x / x0 - radius).abs() < 1e-12);
    // The display values follow the height either way.
    for frame in [&rest, &swung] {
        for (f, points) in scene.display.iter().enumerate() {
            for &p in points {
                let [_, _, z] = coordinates(p);
                let v = frame.value(f, p);
                assert!((v - z).abs() < 0.12, "{v} {z}");
                // Continuous across the edges: where the drawing finds a display point, which
                // may be a neighbour on a shared edge, the value is the same.
                let found = scene
                    .face_of(p * 3.0)
                    .expect("every ray lies in a face's cone");
                assert!((frame.value(found, p) - v).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 1.3, 3.0] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
