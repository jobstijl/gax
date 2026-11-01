//! numga's tests of connections, and its scenario checks.

use super::connections::*;
use super::scenes::*;

use gax::field::Field;
use gax_numga_examples::rng::{Draw, rng};

/// The angle `x` wrapped into `(−π, π]`.
fn wrapped(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    (x + pi).rem_euclid(core::f64::consts::TAU) - pi
}

/// The cosine of an angle, from the turn by it.
fn cos(angle: f64) -> f64 {
    gax::vga2d::Vector::from_angle(angle).e1()
}

#[test]
fn extra_turns_add_to_every_face_holonomy() {
    // Carried around a face by the adjusted rotors, a tangent vector turns by the face's
    // curvature plus the circulation of the extra turns around it.
    let s = Surface::sphere(2);
    let mut draw = rng(0);
    let angles: Vec<f64> = (0..s.mesh.len(1)).map(|_| draw.normal() * 0.3).collect();
    let forward = adjusted_transport(&s.mesh, &s.normals, &angles);
    let circulations =
        &circulation(&s.mesh, &[]) * &angles.iter().map(|&a| Scalar::new(a)).collect::<Field<_>>();
    let facing = s.mesh.facets(2);
    let curvatures = face_curvatures(&s.mesh, &s.normals);
    for f in 0..s.mesh.len(2) {
        // The side facing each corner, forward where the face runs along its edge.
        let side = |k: usize| {
            let r = forward[facing.site(f, k)];
            if facing.sign(f, k) > 0 {
                r
            } else {
                r.reverse()
            }
        };
        // Around the face from its first corner: the sides facing the third, first and second.
        let around = side(1) * side(0) * side(2);
        let n = s.normals.site(s.mesh.simplex(2, f)[0]);
        let turned = (around * around).unit().log().scalar_product(plane(n)).s();
        let expected = curvatures[f] + circulations.site(f).s();
        assert!(
            wrapped(turned - expected).abs() < 1e-12,
            "{f}: {turned} {expected}"
        );
    }
}

/// numga's checks of the carried vectors: tangent, unit, and turned by the cap's solid angle.
fn check_holonomy() {
    let (rings, starts) = holonomy();
    let carried = carried(&rings, &starts);
    for (r, along) in carried.iter().enumerate() {
        for (k, p) in along.iter().enumerate() {
            assert!((*p & rings[k][r]).s().abs() < 1e-12);
            assert!((p.norm() - 1.0).abs() < 1e-12);
        }
        let solid = core::f64::consts::TAU * (1.0 - cos(COLATITUDES[r]));
        assert!((dot(along[along.len() - 1], starts[r]) - cos(solid)).abs() < 2e-3);
    }
}

/// numga's checks of the torus: heat and the modes stay tangent, and the modes' energies come in
/// equal pairs, a quarter-turned mode keeping its energy.
fn check_torus(t: &TorusFields) {
    let tangent = |field: &Field<Point>, tolerance: f64| {
        field
            .zip_with(&t.surface.normals, |p, n| p & n)
            .iter()
            .all(|x| x.s().abs() < tolerance)
    };
    assert!(t.waves.iter().all(|w| tangent(w, 1e-12)));
    assert!(t.modes.iter().all(|m| tangent(m, 1e-11)));
    assert!(t.transported.iter().all(|p| (p.norm() - 1.0).abs() < 1e-12));
    for pair in t.energies.chunks(2) {
        assert!(
            (pair[0] - pair[1]).abs() <= 1e-10 * pair[0].abs(),
            "{:?}",
            t.energies
        );
    }
    let energy = |field: &Field<Point>| {
        let d = t.covariant.of(field);
        t.surface
            .weights
            .zip_with(&d, |w, x| w * x.norm_squared())
            .sum()
            .s()
    };
    for mode in &t.modes {
        let quarter = mode.zip_with(&t.surface.normals, |p, n| p.commutator(plane(n)));
        assert!((energy(&quarter) - energy(mode)).abs() <= 1e-10 * energy(mode));
    }
}

#[test]
fn transport_turns_by_the_solid_angle_and_stays_tangent() {
    check_holonomy();
    check_torus(&torus_fields());
}

/// Every design's circulation is met and its parallel fields carry no energy.
fn check_designs(designs: &[(Design, Vec<usize>)]) {
    for (d, _) in designs {
        assert!(
            d.residual.iter().all(|r| r.abs() < 1e-12),
            "{:?}",
            d.residual.iter().copied().fold(0.0, f64::max)
        );
        assert!(
            d.energies.iter().all(|e| e.abs() < 1e-12),
            "{:?}",
            d.energies
        );
    }
}

#[test]
fn designed_fields_are_parallel_and_their_indices_add_up() {
    let (sphere, sphere_designs) = field_spheres();
    let (torus, torus_designs) = field_tori();
    check_designs(&sphere_designs);
    check_designs(&torus_designs);
    for (d, _) in &sphere_designs {
        assert!(
            d.field
                .zip_with(&sphere.normals, |p, n| p & n)
                .iter()
                .all(|x| x.s().abs() < 1e-12)
        );
    }
    // Indices one short of the Euler characteristic leave one whole turn spread over the faces.
    let indices = face_indices(&sphere, &[0], &[1.0]);
    let (angles, circulation, target) =
        trivial_connection(&sphere.mesh, &sphere.normals, &indices, &[]);
    let spread = core::f64::consts::TAU / sphere.mesh.len(2) as f64;
    assert!(
        residual(&circulation, &angles, &target)
            .iter()
            .all(|r| (r - spread).abs() <= 1e-9 * spread)
    );
    // Turned about every normal, a parallel field stays parallel: unit and tangent.
    let field = turned(&sphere, &sphere_designs[0].0.field, 1.0);
    assert!(field.iter().all(|p| (p.norm() - 1.0).abs() < 1e-12));
    let _ = torus;
}

/// The face curvatures add up to two whole turns.
#[test]
fn the_curvatures_add_up_to_two_turns() {
    let s = Surface::sphere(SPHERE_LEVELS);
    let total: f64 = face_curvatures(&s.mesh, &s.normals).iter().sum();
    assert!((total - 2.0 * core::f64::consts::TAU).abs() <= 1e-12 * total);
}

/// Every stripe field comes with its quarter turn, of equal energy, and across most edges its
/// phase advances by the edge's target.
fn check_stripes(s: &Surface, patterns: &[[Stripes; 2]]) {
    for pattern in patterns {
        for family in pattern {
            let [a, b] = [family.energies[0], family.energies[1]];
            assert!((a - b).abs() <= 1e-6 * b.abs(), "{a} {b}");
            let mismatch = median(stripe_mismatch(
                s,
                &family.line,
                &family.stripes,
                &family.advance,
            ));
            assert!(mismatch < 0.25, "{mismatch}");
        }
    }
}

/// Carried across every edge, each of the tail's lines lands along or against one of the head's,
/// and some edges trade or reverse them: around a singularity the cross turns by a quarter or a
/// half.
fn check_crosses(s: &Surface, scenes: &[(Vec<f64>, Cross)]) {
    for (angles, cross) in scenes {
        let forward = adjusted_transport(&s.mesh, &s.normals, angles);
        let mut traded = false;
        for (k, [tail, head]) in crate::simplicial::edges(&s.mesh).into_iter().enumerate() {
            for a in 0..2 {
                for b in 0..2 {
                    let m = dot(
                        cross.lines[a].site(head),
                        forward[k] >> cross.lines[b].site(tail),
                    );
                    assert!((m.abs() * (1.0 - m.abs())).abs() < 1e-8, "{m}");
                }
            }
            traded |= dot(
                cross.lines[0].site(head),
                forward[k] >> cross.lines[0].site(tail),
            ) < 0.5;
        }
        assert!(traded);
    }
}

#[test]
fn stripes_follow_their_targets_and_crosses_land_on_their_lines() {
    let (sphere, patterns) = stripe_spheres();
    check_stripes(&sphere, &patterns);
    let (crossed, scenes) = crosses();
    check_crosses(&crossed, &scenes);
}

/// The torus's generators by tree and cotree hold the same trivial connections as numga's loops
/// around the tube and the axis.
#[test]
fn tree_and_cotree_generators_make_the_same_designs() {
    let s = Surface::torus(FIELD_MAJOR / 2, FIELD_MINOR / 2);
    let loops = s.mesh.cycles();
    assert_eq!(loops.len(), 2);
    let indices = face_indices(&s, &[3, 40], &[1.0, -1.0]);
    let d = design(&s, &indices, &loops);
    check_designs(&[(d, vec![])]);
}

/// numga's scenario checks on its own sizes (the torus's stripes too), and every figure drawn.
#[test]
fn the_scenes_pass_numgas_checks_and_draw() {
    let shown = super::shown();
    check_holonomy();
    check_torus(&shown.torus);
    check_designs(&shown.sphere_designs);
    check_designs(&shown.torus_designs);
    check_stripes(&shown.stripe_sphere.surface, &shown.sphere_stripes);
    check_stripes(&shown.stripe_torus.surface, &shown.torus_stripes);
    check_crosses(&shown.stripe_sphere.surface, &shown.crosses);
    for t in [1.0, 4.0, 8.0, 12.0, 16.0, 20.0] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
