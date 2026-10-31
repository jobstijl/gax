//! numga's tests of the smoke rings, and its scenario checks.

use super::rings::*;
use gax_numga_examples::rng::{self, Draw};

/// The largest size of a list of points.
fn largest(points: impl IntoIterator<Item = Point>) -> f64 {
    points.into_iter().map(|p| p.norm()).fold(0.0, f64::max)
}

/// The largest difference of an edge's length from one.
fn off_unit(edges: &[Point]) -> f64 {
    edges
        .iter()
        .map(|e| (e.norm() - 1.0).abs())
        .fold(0.0, f64::max)
}

/// The velocity is square to the edges in and out of its vertex, so no edge changes length.
#[test]
fn each_vertex_moves_across_both_its_edges() {
    let (a, b) = KNOT_WINDINGS;
    let points = torus_knot(a, b, KNOT_RADII.0, KNOT_RADII.1, KNOT);
    let outgoing = edges(&points);
    let n = points.len();
    for (k, v) in velocity(&points, circulation()).iter().enumerate() {
        assert!((*v | outgoing[k]).s().abs() <= 1e-13);
        assert!((*v | outgoing[(k + n - 1) % n]).s().abs() <= 1e-13);
    }
}

/// The transformed polygon has unit edges, and lies a time of half the parameter along the smoke
/// ring flow, to first order in the parameter: the remainder shrinks with its square.
#[test]
fn a_backlund_transformation_is_a_step_of_the_flow() {
    let points = wavy_ring(DISCRETE, WAVY_MODE, WAVY_TILT);
    let rate = velocity(&points, circulation());
    for parameter in [0.02, 0.01] {
        let moved = advance(&points, parameter);
        assert!(off_unit(&edges(&moved)) <= 1e-13);
        let step = moved
            .iter()
            .zip(&points)
            .zip(&rate)
            .map(|((m, p), v)| *m - (*p + *v * (parameter / 2.0)));
        assert!(largest(step) < 2.0 * parameter * parameter);
    }
}

/// Folding an offset about the quad's diagonal is the Lorentz rotor of the edge, a boost along
/// the direction the edge's point stands for and a turn about it, acting on the offset's null
/// paravector.
#[test]
fn a_fold_is_a_lorentz_transformation_of_the_offsets() {
    let mut rng = rng::rng(0);
    let edge = point(rng.direction());
    let parameter = 0.3;
    let (rapidity, angle) = fold_rapidity_and_angle(parameter);
    let rotor = lorentz_rotor(edge, rapidity, angle);
    for _ in 0..6 {
        let offset = point(rng.direction());
        let folded = carried(rotor, offset);
        let difference = folded - fold(edge, offset, intrinsic(parameter));
        assert!(
            difference.c.iter().all(|x| x.abs() <= 1e-13),
            "{difference:?}"
        );
    }
}

#[test]
fn a_ring_flies_rigidly_and_the_flow_keeps_edges_and_impulse() {
    let velocities = velocity(&ring(RING), circulation());
    assert!(largest(velocities.iter().map(|v| *v - velocities[0])) <= 1e-13);
    let curves = sweep(wavy_ring(WAVY, WAVY_MODE, WAVY_TILT), SURFACE_STEP, 20, 10);
    let lengths = |c: &[Point]| -> Vec<f64> { edges(c).iter().map(|e| e.norm()).collect() };
    let first = lengths(&curves[0]);
    for c in &curves {
        assert!(
            lengths(c)
                .iter()
                .zip(&first)
                .all(|(a, b)| (a - b).abs() <= 1e-9)
        );
    }
}

#[test]
fn the_soliton_keeps_unit_edges_and_turns_once_through_its_loop() {
    for curve in soliton() {
        assert!(off_unit(&sides(&curve)) <= 1e-12);
        let total: f64 = turn_angles(&curve).iter().sum();
        assert!((total - core::f64::consts::TAU).abs() <= 1e-3, "{total}");
    }
}

/// The rings take turns, each passing through the other twice in each cycle, the narrower one
/// passing.
#[test]
fn two_rings_leapfrog_the_narrower_passing_through_the_wider() {
    let frogs = leapfrog();
    let (heights, radii, crossings) = passes(&frogs);
    assert!(crossings.len() >= 4, "{crossings:?}");
    for t in crossings {
        let behind = usize::from(heights[t][1] < heights[t][0]);
        assert!(radii[t][behind] < radii[t][1 - behind]);
    }
}

/// numga's scenario checks of the continuous flows: the ring flies along its axis at twice the
/// tangent of half its turn per vertex; the flows keep every edge's length and the polygon's
/// vector area, the tolerance on the area testing the time steps.
#[test]
fn the_flows_keep_edges_and_impulse() {
    let velocities = velocity(&ring(RING), circulation());
    #[allow(clippy::disallowed_methods)] // the closed form, numga's reference
    let expected = -2.0 * (core::f64::consts::PI / RING as f64).tan();
    for v in &velocities {
        assert!((height(*v) / expected - 1.0).abs() <= 1e-12);
    }
    for curves in [wavy_surface(), knot()] {
        let lengths = |c: &[Point]| -> Vec<f64> { edges(c).iter().map(|e| e.norm()).collect() };
        let first = lengths(&curves[0]);
        let area = impulse(&curves[0]);
        let scale = area.c.iter().map(|x| x.abs()).fold(0.0, f64::max);
        for c in &curves {
            assert!(
                lengths(c)
                    .iter()
                    .zip(&first)
                    .all(|(a, b)| (a - b).abs() <= 1e-9)
            );
            let now = impulse(c);
            for (a, b) in now.c.iter().zip(area.c) {
                assert!((a - b).abs() <= 1e-3 * scale + 1e-3 * b.abs());
            }
        }
    }
}

/// In discrete time every edge stays exactly of unit length, the ring moves rigidly, and each
/// step is a time of about half the parameter along the smoke ring flow.
#[test]
fn the_discrete_flow_is_rigid_and_follows_the_flow() {
    let stepped = discrete();
    for rings in &stepped {
        for c in rings {
            assert!(off_unit(&edges(c)) <= 1e-12);
        }
    }
    let moved = &stepped[stepped.len() - 1][0];
    for p in moved {
        assert!((radius(*p) / radius(moved[0]) - 1.0).abs() <= 1e-12);
        assert!((height(*p) - height(moved[0])).abs() <= 1e-12);
    }
    let wavy = &stepped[0][1];
    let continuous = sweep(
        wavy.clone(),
        PARAMETER / 2.0,
        DISCRETE_STEPS,
        DISCRETE_STEPS,
    );
    let last = &continuous[continuous.len() - 1];
    let travel = largest(last.iter().zip(wavy).map(|(a, b)| *a - *b));
    let discrete_last = &stepped[stepped.len() - 1][1];
    assert!(largest(discrete_last.iter().zip(last).map(|(a, b)| *a - *b)) < 0.2 * travel);
}

/// The soliton's loop keeps its shape, up to how the vertices sample it, and travels further
/// than its width.
#[test]
fn the_soliton_keeps_its_shape_and_travels() {
    let moments: Vec<(f64, f64)> = soliton()
        .iter()
        .map(|curve| {
            let angles = turn_angles(curve);
            let centre = angles
                .iter()
                .enumerate()
                .map(|(k, a)| a * k as f64)
                .sum::<f64>()
                / core::f64::consts::TAU;
            let spread = angles
                .iter()
                .enumerate()
                .map(|(k, a)| a * (k as f64 - centre).powi(2))
                .sum::<f64>()
                / core::f64::consts::TAU;
            (centre, spread)
        })
        .collect();
    let (first_centre, first_spread) = moments[0];
    assert!(
        moments
            .iter()
            .all(|(_, s)| (s / first_spread - 1.0).abs() <= 1e-2)
    );
    let (last_centre, _) = moments[moments.len() - 1];
    #[allow(clippy::disallowed_methods)] // the loop's width, a statistic
    let width = first_spread.sqrt();
    assert!(last_centre - first_centre > 2.0 * width);
}

/// A ring alone flies along its axis without changing shape, and the leapfrogging rings keep
/// their impulse, each ring's vector area times its circulation.
#[test]
fn the_leapfrogging_rings_keep_their_impulse() {
    let frogs = leapfrog();
    let alone = induced(&frogs[0][..1], &FROG_CIRCULATIONS[..1], FROG_CORE);
    assert!(largest(alone[0].iter().map(|v| *v - alone[0][0])) <= 1e-12);
    let total = |rings: &[Vec<Point>]| -> Patch {
        rings
            .iter()
            .zip(FROG_CIRCULATIONS)
            .map(|(r, g)| impulse(r) * g)
            .sum()
    };
    let first = total(&frogs[0]);
    let scale = first.c.iter().map(|x| x.abs()).fold(0.0, f64::max);
    for rings in &frogs {
        let now = total(rings);
        assert!(
            now.c
                .iter()
                .zip(first.c)
                .all(|(a, b)| (a - b).abs() <= 3e-3 * scale)
        );
    }
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
}
