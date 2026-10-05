//! numga's `math/hopf`: the Hopf fibration, the spinors of three-dimensional space sorted by the
//! direction they point.
//!
//! A spinor of VGA3D is a rotor, four real numbers, and the unit spinors form the three-sphere.
//! Sandwiching z with a spinor gives a unit vector, its direction. With the spinor left open,
//! `Rotor >> z` is that sandwich as a form with two spinor slots, a vector from a pair of
//! spinors; on one spinor twice it is that spinor's direction.
//!
//! Turning a spinor on the right in the xy plane leaves its direction alone, since z commutes
//! with xy: the spinors that point the same way form a circle, the fibre over that direction.
//! The fibre is also where the form `direction | HOPF`, a scalar from two spinors, is largest:
//! its eigenvalues are minus one twice and plus one twice, and the eigenspace of plus one is the
//! plane of the circle.
//!
//! Through the stereographic projection of the three-sphere from the spinor -1, every fibre is a
//! circle in space, any two of them linked once. Carrying a spinor around a closed loop of
//! directions, by the smallest rotation from each direction to the next, brings it back on its
//! own fibre, turned along it by half the solid angle the loop encloses: the Berry phase of a
//! spin one half.
//!
//! The animation: a direction spirals from near the south pole to above the equator, and its
//! fibre is added to space at each step, the newest drawn heavier, while the view turns. Bottom
//! left, a spinor carried around a circle of directions leaves its fibre (grey) and comes back
//! onto it further along.

use gax::vga3d::{Bivector, Rotor, Scalar, Vector};

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::wave;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop,
    caption, colormap, palette, run,
};

mod hopf {
    use super::*;

    pub type V = Vector<(), f64>;
    pub type R = Rotor<(), f64>;
    /// A spinor's direction, with the spinor left open in both places it appears.
    pub type Hopf = Vector<(Rotor, Rotor), f64>;

    /// The Hopf form: `spinor z ~spinor`, with both spinors open.
    pub fn hopf() -> Hopf {
        let z: V = Vector::new(0.0, 0.0, 1.0);
        ((Rotor::slot() * z) * Rotor::slot().reverse()).cast::<Vector>()
    }

    /// A unit spinor pointing along a direction: the top eigenvector of the form
    /// `direction | HOPF`, whose eigenvalues are minus one twice and plus one twice.
    pub fn fibre_start(direction: V) -> R {
        let form: Scalar<(Rotor, Rotor), f64> = direction | hopf();
        form.eigh().1[3]
    }

    /// The rotor that turns on the right in the xy plane by an angle.
    pub fn along_fibre(angle: f64) -> R {
        Bivector::<(), f64>::new(0.0, 0.0, angle).exp().into_inner()
    }

    /// The spinors pointing the same way as the start: the start turned on the right in the xy
    /// plane, at each angle.
    pub fn fibre(start: R, angles: &[f64]) -> Vec<R> {
        angles.iter().map(|a| start * along_fibre(*a)).collect()
    }

    /// The stereographic projection of a unit spinor from -1 into space: the bivector part over
    /// one plus the scalar part, read as the vector it is dual to.
    pub fn stereographic(spinor: R) -> V {
        (spinor.grade::<2>().gp(1.0 / (1.0 + spinor.s()))).dual()
    }

    /// The rotors that carry a frame along a curve of unit directions, by the smallest rotation
    /// from each direction to the next: from the first direction to each of the others.
    pub fn transport(directions: &[V]) -> Vec<R> {
        let one: R = Rotor::new(1.0, 0.0, 0.0, 0.0);
        let mut total = one;
        directions
            .windows(2)
            .map(|w| {
                let step = (one + w[1] * w[0]).normalized().into_inner();
                total = step * total;
                total
            })
            .collect()
    }

    /// The unit direction at a polar angle from z and an azimuth about it: z turned toward x by
    /// the polar angle, then about z by the azimuth.
    pub fn sphere(polar: f64, azimuth: f64) -> V {
        let about_z = Bivector::<(), f64>::new(0.0, 0.0, -azimuth / 2.0).exp();
        let toward_x = Bivector::<(), f64>::new(0.0, -polar / 2.0, 0.0).exp();
        (about_z * toward_x) >> Vector::new(0.0, 0.0, 1.0)
    }

    /// Angles once around a fibre.
    pub fn turn(samples: usize) -> Vec<f64> {
        (0..=samples)
            .map(|k| core::f64::consts::TAU * k as f64 / samples as f64)
            .collect()
    }

    /// The fibre over a direction, projected into space.
    pub fn projected_fibre(direction: V, samples: usize) -> Vec<V> {
        fibre(fibre_start(direction), &turn(samples))
            .into_iter()
            .map(stereographic)
            .collect()
    }

    /// The fibres over directions spaced around circles of latitude, projected into space:
    /// the directions and the projected fibres.
    pub fn tori(polars: &[f64], per_circle: usize, samples: usize) -> Vec<Vec<(V, Vec<V>)>> {
        polars
            .iter()
            .map(|polar| {
                (0..per_circle)
                    .map(|k| {
                        let d = sphere(
                            *polar,
                            core::f64::consts::TAU * k as f64 / per_circle as f64,
                        );
                        (d, projected_fibre(d, samples))
                    })
                    .collect()
            })
            .collect()
    }

    /// A spinor carried once around the circle of directions at a polar angle: the loop of
    /// directions, the start, and the carried spinors.
    pub fn lift(polar: f64, count: usize) -> (Vec<V>, R, Vec<R>) {
        let angles = turn(count);
        let lp: Vec<V> = angles.iter().map(|a| sphere(polar, *a)).collect();
        let start = fibre_start(lp[0]);
        let carried = transport(&lp).into_iter().map(|r| r * start).collect();
        (lp, start, carried)
    }

    /// The direction of the spiral at a fraction of the sweep: from near the south pole to just
    /// above the equator, four times around.
    pub fn spiral(fraction: f64) -> V {
        sphere(
            core::f64::consts::PI * (0.95 - 0.5 * fraction),
            core::f64::consts::TAU * 4.0 * fraction,
        )
    }
}

use hopf::*;

const SECONDS: f32 = 9.0;
const FRAMES: usize = 90;
/// Projected fibres are drawn out to this distance from the origin; the fibres near the
/// projection point run off towards infinity.
const LIMIT: f64 = 2.6;
const LIFT_POLAR: f64 = 2.2;

/// A projected point as drawn, a point of space: the origin plus the vector, and beyond the
/// limit a gap in the curve.
fn clipped(v: V) -> gax::pga3d::Point<(), f64> {
    if v.norm() > LIMIT {
        gax::pga3d::Point::xyz(f64::NAN, f64::NAN, f64::NAN)
    } else {
        at(v)
    }
}

/// The point a vector reaches from the origin: a direction as a point of the sphere of
/// directions, a projected spinor as a point of space.
fn at(v: V) -> gax::pga3d::Point<(), f64> {
    gax::pga3d::Point::xyz(0.0, 0.0, 0.0) + gax::pga3d::Point::direction(v.e1(), v.e2(), v.e3())
}

/// A direction's azimuth: the angle about z from x to its part in the xy plane, read off the
/// logarithm of the rotor that turns one into the other (`level x` turns by minus the angle).
fn azimuth(d: V) -> f64 {
    let level: V = Vector::new(d.e1(), d.e2(), 0.0);
    let turn: Bivector<(), f64> = (level * Vector::new(1.0, 0.0, 0.0)).normalized().log();
    -turn.e12()
}

/// A direction's colour: its azimuth as the hue, whitened a little, fainter towards the south
/// pole.
fn colour(d: V) -> Light {
    let hue = (azimuth(d) / core::f64::consts::TAU) as f32;
    let base = (colormap::hsv(hue)).whitened(0.25);
    base.faded(0.45 + 0.55 * (1.0 + d.e3() as f32) / 2.0)
}

/// The part of the loop spent on the sweep; the rest shows the tori.
const SWEEP: f32 = 0.72;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let phase = t / SECONDS;
    let azimuth = -0.96 + 0.6 * wave(phase * core::f32::consts::TAU);
    let elevation = 0.38;
    // First a direction spirals over the sphere and its fibres pile up; then the fibres over
    // three circles of latitude, which fill nested tori.
    let fibres: Vec<(V, Vec<V>, bool)> = if phase < SWEEP {
        let count = ((phase / SWEEP * FRAMES as f32) as usize + 1).min(FRAMES);
        (0..count)
            .map(|k| {
                let d = spiral(k as f64 / (FRAMES - 1) as f64);
                (d, projected_fibre(d, 240), k + 1 == count)
            })
            .collect()
    } else {
        tori(&[0.9, 0.7, 0.5].map(|f| core::f64::consts::PI * f), 18, 240)
            .into_iter()
            .flatten()
            .map(|(d, fibre)| (d, fibre, false))
            .collect()
    };

    // Space: the fibres so far, the newest heavier.
    let (w, h) = (screen.width(), screen.height());
    let right = Rect::new(0.3 * w, 0.0, w, h);
    let cam = Camera::orbit(
        right,
        Point3::xyz(0.0, 0.0, -0.3),
        11.0,
        azimuth,
        elevation,
        Lens::Perspective(0.42),
    );
    panel3(c, right, cam, |scene| {
        for (d, fibre, newest) in &fibres {
            let pts: Vec<_> = fibre.iter().map(|v| clipped(*v)).collect();
            let (width, strength) = if *newest { (2.6, 1.0) } else { (0.9, 0.75) };
            scene.polyline(&pts, width, (colour(*d)).faded(strength));
        }
    });
    // Any two fibres link once: the first and the last drawn.
    let link = gax_numga_examples::measure::linking(&fibres[0].1, &fibres[fibres.len() - 1].1);
    if fibres.len() > 1 {
        let text = format!("LINKING NUMBER OF THE FIRST AND LAST FIBRE: {link:+.3}");
        let corner = screen.hi + Point2::direction(-14.0, -14.0);
        c.text(&text, corner, 11.0, palette::ink(), Align::Right);
    }
    if phase >= SWEEP {
        let text = "FIBRES OVER CIRCLES OF DIRECTIONS FILL NESTED TORI";
        let corner = screen.top_right() + Point2::direction(-14.0, 30.0);
        c.text(text, corner, 11.0, palette::ink(), Align::Right);
    }

    // The sphere of directions, on the left above the lift.
    let split = h * 0.5 + 30.0;
    let sphere_rect = Rect::new(0.0, 60.0, 0.3 * w, split);
    let cam = Camera::orbit(
        sphere_rect,
        ORIGIN3,
        6.0,
        azimuth,
        elevation,
        Lens::Perspective(0.45),
    );
    panel3(c, sphere_rect, cam, |scene| {
        scene.sphere_wire(ORIGIN3, 1.0, 24, palette::grid().faded(0.6));
        for (d, _, newest) in &fibres {
            let size = if *newest { 10.0 } else { 5.0 };
            scene.dot(at(*d), Marker::Dot, size, colour(*d));
        }
    });
    // The label clear of the caption, beside the sphere's top.
    let label = sphere_rect.lo + Point2::direction(14.0, 30.0);
    c.text("DIRECTIONS", label, 11.0, palette::ink(), Align::Left);

    // The lift: a spinor carried around a circle of directions, drawn up to now.
    let lift_rect = Rect::new(0.0, split, 0.3 * w, h);
    let (_, start, carried) = lift(LIFT_POLAR, 200);
    let cam = Camera::orbit(
        lift_rect,
        ORIGIN3,
        9.0,
        azimuth,
        elevation,
        Lens::Perspective(0.42),
    );
    panel3(c, lift_rect, cam, |scene| {
        let circle: Vec<_> = fibre(start, &turn(200))
            .into_iter()
            .map(|r| clipped(stereographic(r)))
            .collect();
        scene.polyline(&circle, 1.2, palette::grid());
        let upto = ((phase / SWEEP * carried.len() as f32) as usize).clamp(1, carried.len());
        let track: Vec<_> = core::iter::once(start)
            .chain(carried[..upto].iter().copied())
            .map(|r| clipped(stereographic(r)))
            .collect();
        scene.polyline(&track, 1.8, palette::red());
        let first = clipped(stereographic(start));
        scene.dot(first, Marker::Dot, 8.0, palette::sky());
        scene.dot(track[track.len() - 1], Marker::Dot, 8.0, palette::red());
    });
    let down = Point2::direction(0.0, 1.0);
    let label = lift_rect.lo + Point2::direction(14.0, 8.0);
    let ink = palette::ink();
    c.text(
        "BERRY PHASE: BACK ON THE FIBRE,",
        label,
        10.0,
        ink,
        Align::Left,
    );
    let next = label + down.gp(14.0);
    c.text(
        "TURNED BY HALF THE SOLID ANGLE",
        next,
        10.0,
        ink,
        Align::Left,
    );

    caption(
        c,
        "THE HOPF FIBRATION",
        "SPINORS (VGA3D ROTORS) THAT POINT THE SAME WAY: LINKED CIRCLES IN SPACE",
    );
}

fn main() {
    run(Anim::new("hopf", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::hopf::*;
    use gax::ApproxEq;
    use gax::vga3d::Vector;

    /// numga's test: `(direction | HOPF)` has eigenvalues -1, -1, 1, 1, and both spinors of the
    /// top pair point along the direction; turning a spinor on the right in the xy plane keeps
    /// it on its fibre.
    #[test]
    fn the_fibre_is_the_top_eigenspace_of_the_paired_form() {
        let direction: V = Vector::new(0.3, -0.5, 0.8).normalized().into_inner();
        let (values, spinors) = (direction | hopf()).eigh();
        for (v, want) in values.iter().zip([-1.0, -1.0, 1.0, 1.0]) {
            assert!((v - want).abs() < 1e-12, "{values:?}");
        }
        for s in &spinors[2..] {
            assert!(hopf().of(*s).of(*s).approx_eq(&direction, 1e-12));
        }
        for r in fibre(spinors[3], &[0.4, 2.1]) {
            assert!(hopf().of(r).of(r).approx_eq(&direction, 1e-9));
        }
    }

    /// numga's `tori` checks, at the test's sizes: every spinor on a fibre points along the
    /// fibre's direction, and two fibres are linked once.
    #[test]
    fn fibres_point_along_and_link_once() {
        let polars = [0.8, 0.5].map(|f| core::f64::consts::PI * f);
        let per_circle = 5;
        let samples = 120;
        let tori = tori(&polars, per_circle, samples);
        for circle in &polars {
            for k in 0..per_circle {
                let d = sphere(
                    *circle,
                    core::f64::consts::TAU * k as f64 / per_circle as f64,
                );
                for r in fibre(fibre_start(d), &turn(samples)) {
                    assert!(hopf().of(r).of(r).approx_eq(&d, 1e-9));
                }
            }
        }
        let link = gax_numga_examples::measure::linking(&tori[0][0].1, &tori[1][per_circle / 3].1);
        assert!((link - 1.0).abs() < 1e-2, "{link}");
    }

    /// numga's `lift` check: the spinor comes back on its own fibre, turned along it by half
    /// the solid angle the loop encloses.
    #[test]
    fn the_carried_spinor_turns_by_half_the_solid_angle() {
        let polar = 2.2;
        let (_, start, carried) = lift(polar, 200);
        #[allow(clippy::disallowed_methods)] // the reference it is checked against
        let half_solid_angle = core::f64::consts::PI * (1.0 - polar.cos());
        let turned = start * along_fibre(-half_solid_angle);
        let last = carried[carried.len() - 1];
        let gap = last.max_abs_diff(&turned);
        assert!(gap < 1e-4, "{gap}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
