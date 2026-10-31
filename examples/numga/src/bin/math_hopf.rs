//! numga's `math/hopf`: the Hopf fibration, the unit spinors of three-dimensional space sorted by
//! their spin.
//!
//! A unit spinor turns the north pole of the unit sphere into another point of that sphere, its
//! spin. The spinor holds more than its spin: turning it in the xy plane before it acts changes
//! the spinor and not the point it turns the pole into. The spinors with each spin therefore form
//! a circle, the fibre over that spin; the fibres fill the three-sphere of unit spinors, and every
//! two of them are linked once. Carried around a closed loop of spins, a spinor comes back onto
//! its own fibre, turned along it by half the solid angle the loop encloses: the Berry phase of a
//! spin one half.
//!
//! Spinors are VGA3D rotors and spins points of the unit sphere, intrinsic bivectors like every
//! position (numga's antivectors, ADR-051), and the sandwich `Rotor >> z.dual()` of the pole is a
//! form on pairs of spinors whose value on one spinor twice is that spinor's spin. Paired with a
//! spin by numga's `dot`, `⟨a ~b⟩₀`, it is a scalar form on pairs of spinors with eigenvalues
//! minus one twice and plus one twice; its top eigenspace is the plane of the fibre over that
//! spin. Stereographic projection takes a spinor to a point of space, its bivector part over one
//! plus its scalar part, made a point by a handedness: an intrinsic scalar, the right-hand rule
//! that takes a plane of turning to a point. The fibres' linking number is a sum of regressive
//! products of points, an intrinsic scalar: its sign is the side the turns along the fibres act
//! on, measured against that handedness.
//!
//! The animation: a spin spirals from near the south pole to above the equator, and its fibre is
//! added to space at each step, the newest drawn heavier, while the view turns; then the fibres
//! over three circles of latitude fill nested tori. Bottom left, a spinor carried around a circle
//! of spins leaves its fibre (grey) and comes back onto it further along.

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::wave;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop,
    caption, colormap, palette, reach3, run,
};

mod hopf {
    use gax::vga3d::{Bivector, Rotor, Scalar, Vector, intrinsic};

    /// A spinor's spin, a point of the unit sphere, and a point of the space spinors project
    /// into: an intrinsic bivector, numga's antivector.
    pub type Spin = intrinsic::Bivector;
    /// A spinor's spin, the pole it turns, as a form on pairs of spinors.
    pub type Hopf = intrinsic::Bivector<(Rotor, Rotor)>;

    /// The north pole of the unit sphere, `z.dual()`: the point `xy`.
    pub fn pole() -> Spin {
        Vector::new(0.0, 0.0, 1.0).dual()
    }

    /// The right-hand rule, measured against the basis volume `x ∧ y ∧ z`: it takes the bivector
    /// part of a spinor to a point of space.
    pub fn handedness() -> intrinsic::Scalar {
        intrinsic::Scalar::new(1.0)
    }

    /// The Hopf form: `spinor pole ~spinor`, with both spinors open.
    pub fn hopf() -> Hopf {
        ((Rotor::slot() * pole()) * Rotor::slot().reverse()).cast::<Bivector>()
    }

    /// A unit spinor with a spin: the top eigenvector of the form `spin.dot(hopf)` (numga's
    /// `dot`, `⟨a ~b⟩₀`), whose eigenvalues are minus one twice and plus one twice.
    pub fn fibre_start(spin: Spin) -> Rotor {
        let form: Scalar<(Rotor, Rotor)> = spin.scalar_product(hopf().reverse());
        form.eigh(Scalar::identity()).1[3]
    }

    /// The rotor that turns on the right in the xy plane by an angle.
    pub fn along_fibre(angle: f64) -> Rotor {
        Bivector::new(0.0, 0.0, angle).exp().into_inner()
    }

    /// The spinors with the start's spin: the start turned on the right in the xy plane, at each
    /// angle.
    pub fn fibre(start: Rotor, angles: &[f64]) -> Vec<Rotor> {
        angles.iter().map(|a| start * along_fibre(*a)).collect()
    }

    /// The stereographic projection of a unit spinor from -1 into space: the bivector part over
    /// one plus the scalar part, made a point by the handedness.
    pub fn stereographic(spinor: Rotor) -> Spin {
        handedness() * spinor.grade::<2>().gp(1.0 / (1.0 + spinor.s()))
    }

    /// The rotors that carry a frame along a curve of unit spins, by the smallest rotation from
    /// each spin to the next: from the first spin to each of the others.
    pub fn transport(spins: &[Spin]) -> Vec<Rotor> {
        let one: Rotor = Rotor::new(1.0, 0.0, 0.0, 0.0);
        let mut total = one;
        spins
            .windows(2)
            .map(|w| {
                let step = (one + w[1] * w[0].reverse()).normalized();
                total = step * total;
                total
            })
            .collect()
    }

    /// Gauss's linking number of two closed polygons: the regressive product of each pair of
    /// segments with the line between them, over the cube of its length, summed and divided by
    /// four pi: an intrinsic scalar, its sign measured against the handedness.
    pub fn linking(first: &[Spin], second: &[Spin]) -> intrinsic::Scalar {
        let pairs = first
            .windows(2)
            .flat_map(|a| second.windows(2).map(move |b| (a, b)));
        let sum: intrinsic::Scalar = pairs
            .map(|(a, b)| {
                let separation = (a[1] + a[0] - b[1] - b[0]).gp(0.5);
                let volume = separation & (a[1] - a[0]) & (b[1] - b[0]);
                volume.gp(1.0 / separation.norm().powi(3))
            })
            .fold(intrinsic::Scalar::new(0.0), |s, v| s + v);
        sum.gp(1.0 / (4.0 * core::f64::consts::PI))
    }

    /// The unit spin at a polar angle and an azimuth: the pole tilted in the zx plane by the
    /// polar angle, then turned in the xy plane by the azimuth.
    pub fn sphere(polar: f64, azimuth: f64) -> Spin {
        let tilt = Bivector::new(0.0, -polar / 2.0, 0.0).exp();
        let turn = Bivector::new(0.0, 0.0, -azimuth / 2.0).exp();
        (turn * tilt) >> pole()
    }

    /// Angles once around a fibre.
    pub fn turn(samples: usize) -> Vec<f64> {
        (0..=samples)
            .map(|k| core::f64::consts::TAU * k as f64 / samples as f64)
            .collect()
    }

    /// The fibre over a spin, projected into space.
    pub fn projected_fibre(spin: Spin, samples: usize) -> Vec<Spin> {
        fibre(fibre_start(spin), &turn(samples))
            .into_iter()
            .map(stereographic)
            .collect()
    }

    /// The fibres over spins spaced around circles of latitude, projected into space: the spins
    /// and the projected fibres.
    pub fn tori(polars: &[f64], per_circle: usize, samples: usize) -> Vec<Vec<(Spin, Vec<Spin>)>> {
        polars
            .iter()
            .map(|polar| {
                (0..per_circle)
                    .map(|k| {
                        let azimuth = core::f64::consts::TAU * k as f64 / per_circle as f64;
                        let spin = sphere(*polar, azimuth);
                        (spin, projected_fibre(spin, samples))
                    })
                    .collect()
            })
            .collect()
    }

    /// A spinor carried once around the circle of spins at a polar angle: the loop of spins, the
    /// start, and the carried spinors.
    pub fn lift(polar: f64, steps: usize) -> (Vec<Spin>, Rotor, Vec<Rotor>) {
        let angles = turn(steps);
        let lp: Vec<Spin> = angles.iter().map(|a| sphere(polar, *a)).collect();
        let start = fibre_start(lp[0]);
        let carried = transport(&lp).into_iter().map(|r| r * start).collect();
        (lp, start, carried)
    }

    /// numga's spiral of spins: its polar angles at the start and the end (fractions of pi), and
    /// the turns it makes about the pole.
    pub const SPIRAL_POLARS: [f64; 2] = [0.95, 0.45];
    pub const SPIRAL_TURNS: f64 = 4.0;

    /// The spin of the spiral at a fraction of the sweep: from near the south pole to just above
    /// the equator, four times around.
    pub fn spiral(fraction: f64) -> Spin {
        let [from, to] = SPIRAL_POLARS;
        sphere(
            core::f64::consts::PI * (from + (to - from) * fraction),
            core::f64::consts::TAU * SPIRAL_TURNS * fraction,
        )
    }

    /// A point (a spin on the sphere, or a projected spinor) as the vector from the origin to
    /// it, for drawing: its pairings `x & p`, `y & p`, `z & p` with the planes x, y and z, numga's
    /// `coordinates`; the metric-free dual reads the same.
    pub fn position(p: Spin) -> Vector {
        p.dual()
    }
}

use hopf::*;

const SECONDS: f32 = 9.0;
/// numga's frames of the spiral, one fibre each.
const FRAMES: usize = 68;
/// Projected fibres are drawn out to this distance from the origin; the fibres near the
/// projection point run off towards infinity.
const LIMIT: f64 = 2.6;
/// numga's tori: the polar angles of the circles of spins (fractions of pi), the spins on each
/// circle, and the samples of each fibre.
const POLARS: [f64; 3] = [0.9, 0.7, 0.5];
const PER_CIRCLE: usize = 18;
const SAMPLES: usize = 240;
/// numga's lift: the polar angle of the loop of spins a spinor is carried around, and the steps
/// around it.
const LIFT_POLAR: f64 = 2.2;
const LIFT_STEPS: usize = 400;

/// A projected point as drawn, a point of space, and beyond the limit a gap in the curve.
fn clipped(p: Spin) -> Point3 {
    if p.norm() > LIMIT {
        Point3::xyz(f32::NAN, f32::NAN, f32::NAN)
    } else {
        reach3(position(p))
    }
}

/// A spin's azimuth: the signed angle about z from x to its part in the xy plane.
fn azimuth(spin: Spin) -> f64 {
    let v = position(spin);
    let x = gax::vga2d::Vector::new(1.0, 0.0);
    x.angle_to(gax::vga2d::Vector::new(v.e1(), v.e2()))
}

/// A spin's colour: its azimuth as the hue, whitened a little, fainter towards the south pole.
fn colour(spin: Spin) -> Light {
    let hue = (azimuth(spin) / core::f64::consts::TAU) as f32;
    let base = (colormap::hsv(hue)).whitened(0.25);
    let height = position(spin).e3() as f32;
    base.faded(0.45 + 0.55 * (1.0 + height) / 2.0)
}

/// The part of the loop spent on the sweep; the rest shows the tori.
const SWEEP: f32 = 0.72;

/// A fibre as drawn: the spin it lies over, its points in space, and whether it is the newest.
type Fibre = (Spin, Vec<Spin>, bool);

/// The fibres at `phase` of the loop: first a spin spirals over the sphere and its fibres pile
/// up; then the fibres over three circles of latitude, which fill nested tori.
fn fibres(phase: f32) -> Vec<Fibre> {
    if phase < SWEEP {
        let count = ((phase / SWEEP * FRAMES as f32) as usize + 1).min(FRAMES);
        (0..count)
            .map(|k| {
                let spin = spiral(k as f64 / (FRAMES - 1) as f64);
                (spin, projected_fibre(spin, SAMPLES), k + 1 == count)
            })
            .collect()
    } else {
        tori(
            &POLARS.map(|f| core::f64::consts::PI * f),
            PER_CIRCLE,
            SAMPLES,
        )
        .into_iter()
        .flatten()
        .map(|(spin, fibre)| (spin, fibre, false))
        .collect()
    }
}

/// Space: the fibres so far, the newest heavier, and the linking number of the first and the
/// last.
fn draw_space(
    c: &mut Canvas,
    rect: Rect,
    fibres: &[Fibre],
    (azimuth, elevation): (f32, f32),
    phase: f32,
) {
    let screen = c.rect();
    let cam = Camera::orbit(
        rect,
        Point3::xyz(0.0, 0.0, -0.3),
        11.0,
        azimuth,
        elevation,
        Lens::Perspective(0.42),
    );
    panel3(c, rect, cam, |scene| {
        for (spin, fibre, newest) in fibres {
            let pts: Vec<_> = fibre.iter().map(|p| clipped(*p)).collect();
            let (width, strength) = if *newest { (2.6, 1.0) } else { (0.9, 0.75) };
            scene.polyline(&pts, width, (colour(*spin)).faded(strength));
        }
    });
    // Any two fibres link once: the first and the last drawn.
    if fibres.len() > 1 {
        let link = linking(&fibres[0].1, &fibres[fibres.len() - 1].1).s();
        let text = format!("LINKING NUMBER OF THE FIRST AND LAST FIBRE: {link:+.3}");
        let corner = screen.hi + Point2::direction(-14.0, -14.0);
        c.text(&text, corner, 11.0, palette::INK, Align::Right);
    }
    if phase >= SWEEP {
        let text = "FIBRES OVER CIRCLES OF SPINS FILL NESTED TORI";
        let corner = screen.top_right() + Point2::direction(-14.0, 30.0);
        c.text(text, corner, 11.0, palette::INK, Align::Right);
    }
}

/// The sphere of spins, a dot for each fibre's.
fn draw_spins(c: &mut Canvas, rect: Rect, fibres: &[Fibre], (azimuth, elevation): (f32, f32)) {
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        6.0,
        azimuth,
        elevation,
        Lens::Perspective(0.45),
    );
    panel3(c, rect, cam, |scene| {
        scene.sphere_wire(ORIGIN3, 1.0, 24, palette::GRID.faded(0.6));
        for (spin, _, newest) in fibres {
            let size = if *newest { 10.0 } else { 5.0 };
            scene.dot(reach3(position(*spin)), Marker::Dot, size, colour(*spin));
        }
    });
    // The label clear of the caption, beside the sphere's top.
    let label = rect.lo + Point2::direction(14.0, 30.0);
    c.text("SPINS", label, 11.0, palette::INK, Align::Left);
}

/// The lift: a spinor carried around a circle of spins, drawn up to now.
fn draw_lift(c: &mut Canvas, rect: Rect, (azimuth, elevation): (f32, f32), phase: f32) {
    let (_, start, carried) = lift(LIFT_POLAR, LIFT_STEPS);
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        9.0,
        azimuth,
        elevation,
        Lens::Perspective(0.42),
    );
    panel3(c, rect, cam, |scene| {
        let circle: Vec<_> = fibre(start, &turn(LIFT_STEPS))
            .into_iter()
            .map(|r| clipped(stereographic(r)))
            .collect();
        scene.polyline(&circle, 1.2, palette::GRID);
        let upto = ((phase / SWEEP * carried.len() as f32) as usize).clamp(1, carried.len());
        let track: Vec<_> = core::iter::once(start)
            .chain(carried[..upto].iter().copied())
            .map(|r| clipped(stereographic(r)))
            .collect();
        scene.polyline(&track, 1.8, palette::RED);
        let first = clipped(stereographic(start));
        scene.dot(first, Marker::Dot, 8.0, palette::SKY);
        scene.dot(track[track.len() - 1], Marker::Dot, 8.0, palette::RED);
    });
    let down = Point2::direction(0.0, 1.0);
    let label = rect.lo + Point2::direction(14.0, 8.0);
    let ink = palette::INK;
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
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let phase = t / SECONDS;
    let azimuth = -0.96 + 0.6 * wave(phase * core::f32::consts::TAU);
    let view = (azimuth, 0.38);
    let fibres = fibres(phase);
    let (w, h) = (screen.width(), screen.height());
    draw_space(c, Rect::new(0.3 * w, 0.0, w, h), &fibres, view, phase);
    // The sphere of spins on the left, above the lift.
    let split = h * 0.5 + 30.0;
    draw_spins(c, Rect::new(0.0, 60.0, 0.3 * w, split), &fibres, view);
    draw_lift(c, Rect::new(0.0, split, 0.3 * w, h), view, phase);
    caption(
        c,
        "THE HOPF FIBRATION",
        "UNIT SPINORS (VGA3D ROTORS) WITH THE SAME SPIN: LINKED CIRCLES IN SPACE",
    );
}

fn main() {
    run(Anim::new("hopf", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::hopf::*;
    use super::{LIFT_POLAR, LIFT_STEPS, PER_CIRCLE, POLARS, SAMPLES};
    use gax::ApproxEq;
    use gax::vga3d::{Rotor, Scalar, Vector};

    /// Every spinor on each fibre of `tori` has the fibre's spin, to numga's 1e-9.
    fn assert_fibres_have_their_spins(polars: &[f64], per_circle: usize, samples: usize) {
        for polar in polars {
            for k in 0..per_circle {
                let spin = sphere(
                    *polar,
                    core::f64::consts::TAU * k as f64 / per_circle as f64,
                );
                for r in fibre(fibre_start(spin), &turn(samples)) {
                    assert!(hopf().of(r).of(r).approx_eq(&spin, 1e-9));
                }
            }
        }
    }

    /// The spinor carried around the loop at `polar` comes back on its own fibre, turned along
    /// it by half the solid angle the loop encloses: the gap.
    fn lift_gap(polar: f64, steps: usize) -> f64 {
        let (_, start, carried) = lift(polar, steps);
        #[allow(clippy::disallowed_methods)] // the reference it is checked against
        let half_solid_angle = core::f64::consts::PI * (1.0 - polar.cos());
        let turned = start * along_fibre(-half_solid_angle);
        carried[carried.len() - 1].max_abs_diff(&turned)
    }

    /// numga's test: `spin.dot(hopf)` has eigenvalues -1, -1, 1, 1, and both spinors of the top
    /// pair have the spin; turning a spinor on the right in the xy plane keeps it on its fibre.
    #[test]
    fn the_fibre_is_the_top_eigenspace_of_the_paired_form() {
        let spin: Spin = Spin::new(0.3, -0.5, 0.8).normalized();
        let form: Scalar<(Rotor, Rotor)> = spin.scalar_product(hopf().reverse());
        let (values, spinors) = form.eigh(Scalar::identity());
        for (v, want) in values.iter().zip([-1.0, -1.0, 1.0, 1.0]) {
            assert!((v - want).abs() < 1e-12, "{values:?}");
        }
        for s in &spinors[2..] {
            assert!(hopf().of(*s).of(*s).approx_eq(&spin, 1e-12));
        }
        for r in fibre(spinors[3], &[0.4, 2.1]) {
            assert!(hopf().of(r).of(r).approx_eq(&spin, 1e-9));
        }
    }

    /// numga's test: at the test's sizes every spinor on a fibre has the fibre's spin, and three
    /// pairs of fibres (numga's `[0, 0]–[1, 3]`, `[0, 1]–[0, 4]`, `[1, 2]–[1, 0]`) link once. numga
    /// allows 2e-2; the port holds 1e-2 (its numbers are 1.0005, 1.0007, 1.0002).
    #[test]
    fn fibres_have_their_spins_and_link_once() {
        let polars = [0.8, 0.5].map(|f| core::f64::consts::PI * f);
        let (per_circle, samples) = (5, 120);
        assert_fibres_have_their_spins(&polars, per_circle, samples);
        let tori = tori(&polars, per_circle, samples);
        for ((a, b), (c, d)) in [((0, 0), (1, 3)), ((0, 1), (0, 4)), ((1, 2), (1, 0))] {
            let link = linking(&tori[a][b].1, &tori[c][d].1).s();
            assert!((link - 1.0).abs() < 1e-2, "{link}");
        }
    }

    /// numga's scenario checks at its sizes: the tori's fibres have their spins, the first and
    /// a later fibre link once, and the spinor carried around the loop turns by half the solid
    /// angle.
    #[test]
    fn the_scenarios_pass_their_checks() {
        let polars = POLARS.map(|f| core::f64::consts::PI * f);
        assert_fibres_have_their_spins(&polars, PER_CIRCLE, SAMPLES);
        let tori = tori(&polars, PER_CIRCLE, SAMPLES);
        let link = linking(&tori[0][0].1, &tori[2][PER_CIRCLE / 3].1).s();
        assert!((link - 1.0).abs() < 1e-2, "{link}");
        let gap = lift_gap(LIFT_POLAR, LIFT_STEPS);
        assert!(gap < 1e-4, "{gap}");
    }

    /// numga's test: a spinor carried around a loop of 200 steps turns by half the solid angle.
    /// numga allows 1e-3; the port holds the scenario's 1e-4.
    #[test]
    fn a_spinor_carried_around_a_loop_turns_by_half_the_solid_angle() {
        let gap = lift_gap(2.2, 200);
        assert!(gap < 1e-4, "{gap}");
    }

    /// numga's numbers at numga 35797de: a spinor's spin and its stereographic projection, the
    /// pole, and the coordinates the planes x, y, z read from a point.
    #[test]
    fn the_numbers_agree_with_numga() {
        let q: Rotor = Rotor::new(0.5, 0.1, -0.7, 0.3).normalized();
        let want = Spin::new(
            0.9047619047619049,
            -0.38095238095238104,
            -0.19047619047619044,
        );
        assert!(hopf().of(q).of(q).approx_eq(&want, 1e-14));
        let want = Spin::new(0.0705957862696895, -0.49417050388782646, 0.2117873588090685);
        let p = stereographic(q);
        assert!(p.approx_eq(&want, 1e-14));
        assert!(pole().approx_eq(&Spin::new(0.0, 0.0, 1.0), 0.0));
        let planes = [
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ];
        let read = planes.map(|plane| (plane & p).s());
        assert!(position(p).approx_eq(&Vector::new(read[0], read[1], read[2]), 1e-15));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
