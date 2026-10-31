//! numga's tests of the floating bodies, and its scenario checks.

use super::hydro::*;
use gax::pga3d::Point;

/// A box two by one by one, floating with its waterline on no edge, its centre of gravity this
/// high over its bottom.
const SIZES: [f64; 3] = [2.0, 1.0, 1.0];
const BOX_DRAFT: f64 = 0.37;
const GRAVITY_HEIGHT: f64 = 0.3;

#[test]
fn box_buoyancy_metacentric_heights_and_righting_arm_are_the_closed_forms() {
    let body = cuboid(SIZES).triangles();
    let [length, beam, depth] = SIZES;
    let (push, growth) = buoyancy(&body, surface_at(BOX_DRAFT - depth / 2.0), 1.0);
    assert!(((push & heave()).s() - length * beam * BOX_DRAFT).abs() < 1e-12);
    assert!(((growth & heave()).s() - length * beam).abs() < 1e-12);
    let weight = length * beam * BOX_DRAFT;
    let centre = Point::xyz(0.0, 0.0, GRAVITY_HEIGHT - depth / 2.0);
    // The metacentric heights: the centre of buoyancy's height, plus the waterplane's second
    // moment over the displaced volume, less the centre of gravity's height.
    let across = BOX_DRAFT / 2.0 + beam * beam / (12.0 * BOX_DRAFT) - GRAVITY_HEIGHT;
    let along = BOX_DRAFT / 2.0 + length * length / (12.0 * BOX_DRAFT) - GRAVITY_HEIGHT;
    let k = stiffness(&body, BOX_DRAFT - depth / 2.0, weight, centre, 1.0);
    let [low, high] = eigenvalues(tilt_block(k)).map(|e| e / weight);
    assert!(
        (low - across).abs() < 1e-12 && (high - along).abs() < 1e-12,
        "{low} {high}"
    );
    // Wall-sided until the deck edge dips: the righting arm is sin(heel) (GM + BM / 2 tan²).
    for degrees in [10.0, 20.0, 30.0] {
        let turn = heeled(degrees);
        let height = afloat(&body, turn, weight, 0.0, 1.0, NEWTON_STEPS);
        let (push, weighing) = forces(&body, turn, height, weight, centre, 1.0);
        let heel: f64 = f64::to_radians(degrees);
        #[allow(clippy::disallowed_methods)] // the closed form, numga's reference
        let expected =
            heel.sin() * (across + beam * beam / (12.0 * BOX_DRAFT) / 2.0 * heel.tan().powi(2));
        assert!((arm(push + weighing, weight) - expected).abs() < 1e-12);
    }
}

/// A cube floats face up only below a density of (3 - √3) / 6 or above (3 + √3) / 6: just inside
/// the band its upright stiffness against heel is negative, just outside positive.
#[test]
fn cube_floats_face_up_outside_the_band_of_densities() {
    let cube = cuboid([1.0; 3]).triangles();
    #[allow(clippy::disallowed_methods)] // the band's edges, numga's reference
    let root = 3f64.sqrt();
    let edges = [(3.0 - root) / 6.0, (3.0 + root) / 6.0];
    let signs: Vec<bool> = edges
        .iter()
        .flat_map(|e| [e - 1e-3, e + 1e-3])
        .map(|d| {
            let k = stiffness(&cube, d - 0.5, d, Point::origin(), 1.0);
            tilt_block(k)[0][0] > 0.0
        })
        .collect();
    assert_eq!(signs, [true, false, false, true]);
}

/// numga's scenario checks: the slope of each righting arm curve at upright is the metacentric
/// height in roll; a cube floats face up exactly outside the band; the hull with the low load
/// comes back upright, with the high one it capsizes and floats upside down.
#[test]
fn the_scenes_right_themselves_or_capsize() {
    let (body, weight, centres) = ship();
    let hull = body.triangles();
    let (couples, stiffnesses) = righting_curves(&hull, weight, &centres);
    let second = heels()[1].to_radians();
    #[allow(clippy::disallowed_methods)] // the small-heel slope's reference
    let sine = second.sin();
    for (load, k) in stiffnesses.iter().enumerate() {
        let height = tilt_block(*k)[0][0] / weight;
        let slope = arm(couples[1][load], weight) / sine;
        assert!((slope - height).abs() < 1e-2, "{load}: {slope} {height}");
    }
    let cube = cuboid([1.0; 3]).triangles();
    #[allow(clippy::disallowed_methods)] // the band's edges
    let root = 3f64.sqrt();
    let edges = [(3.0 - root) / 6.0, (3.0 + root) / 6.0];
    let densities = densities();
    for (d, row) in densities.iter().zip(cube_couples(&cube, &densities)) {
        let face_up = stable_heel(&row).abs() < 1e-6;
        assert_eq!(face_up, *d < edges[0] || *d > edges[1], "{d}");
    }
    let [low, high] = RELEASED.map(|load| released(&hull, weight, centres[load]));
    let (low, high) = (
        low[FRAMES - 1].roll.to_degrees(),
        high[FRAMES - 1].roll.to_degrees(),
    );
    assert!(
        low.abs() < 5.0 && (high - 180.0).abs() < 10.0,
        "{low} {high}"
    );
}

/// The roll's angle reads back the turn it was made from.
#[test]
fn a_roll_reads_its_angle() {
    let roll = heel_axis() * 0.4;
    assert!(((roll & heel_axis().dual()).s() - 0.4).abs() < 1e-15);
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
}
