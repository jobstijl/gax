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
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Marker, Rgb, Scene3, backdrop, canvas, caption, colormap,
    palette, run,
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

    /// Gauss's linking number of two closed polygons: the volume each pair of segments spans
    /// with the line between them, over the cube of its length, summed and divided by four pi.
    pub fn linking(first: &[V], second: &[V]) -> f64 {
        let mut sum = 0.0;
        for a in first.windows(2) {
            for b in second.windows(2) {
                let separation = (a[1] + a[0] - b[1] - b[0]).gp(0.5);
                let volume = (separation ^ (a[1] - a[0]) ^ (b[1] - b[0])).dual().s();
                sum += volume / separation.norm().powi(3);
            }
        }
        sum / (4.0 * core::f64::consts::PI)
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

fn xyz(v: V) -> [f32; 3] {
    let [x, y, z] = [v.e1(), v.e2(), v.e3()];
    if (x * x + y * y + z * z).sqrt() > LIMIT {
        [f32::NAN; 3]
    } else {
        [x as f32, y as f32, z as f32]
    }
}

/// A direction's colour: its azimuth as the hue, darker towards the south pole.
fn colour(d: V) -> Rgb {
    let hue = (d.e2().atan2(d.e1()) / core::f64::consts::TAU) as f32;
    let base = canvas::mix(colormap::hsv(hue), [1.0; 3], 0.25);
    canvas::scale(base, 0.45 + 0.55 * (1.0 + d.e3() as f32) / 2.0)
}

/// A canvas for a panel at `rect`, its backdrop the matching band of the full backdrop, so that
/// a 3D scene drawn on it with its own camera can be blitted in place.
fn panel(c: &Canvas, rect: [usize; 4]) -> Canvas {
    let mut p = Canvas::new(rect[2] - rect[0], rect[3] - rect[1]);
    let h = c.height as f32 - 1.0;
    p.backdrop(
        canvas::mix(palette::top(), palette::bottom(), rect[1] as f32 / h),
        canvas::mix(palette::top(), palette::bottom(), (rect[3] - 1) as f32 / h),
    );
    p
}

/// The part of the loop spent on the sweep; the rest shows the tori.
const SWEEP: f32 = 0.72;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width, c.height);
    let phase = t / SECONDS;
    let azimuth = -0.96 + 0.6 * (phase * core::f32::consts::TAU).sin();
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
    let seen: Vec<V> = fibres.iter().map(|f| f.0).collect();

    // Space: the fibres so far, the newest heavier.
    let rect = [(w as f32 * 0.3) as usize, 0, w, h];
    let mut space = panel(c, rect);
    let cam = Camera::orbit(
        space.width,
        space.height,
        [0.0, 0.0, -0.3],
        11.0,
        azimuth,
        elevation,
        Lens::Perspective(0.42),
    );
    let mut scene = Scene3::new(cam);
    for (d, fibre, newest) in &fibres {
        let pts: Vec<[f32; 3]> = fibre.iter().map(|v| xyz(*v)).collect();
        let (width, alpha) = if *newest { (2.6, 1.0) } else { (0.9, 0.75) };
        scene.polyline(&pts, width, colour(*d), alpha);
    }
    scene.draw(&mut space);
    c.blit(&space, rect[0], rect[1]);
    // Any two fibres link once: the first and the last drawn.
    let link = linking(&fibres[0].1, &fibres[fibres.len() - 1].1);
    if fibres.len() > 1 {
        c.text(
            &format!("LINKING NUMBER OF THE FIRST AND LAST FIBRE: {link:+.3}"),
            w as f32 - 14.0,
            h as f32 - 14.0,
            11.0,
            palette::ink(),
            Align::Right,
        );
    }
    if phase >= SWEEP {
        c.text(
            "FIBRES OVER CIRCLES OF DIRECTIONS FILL NESTED TORI",
            w as f32 - 14.0,
            30.0,
            11.0,
            palette::ink(),
            Align::Right,
        );
    }

    // The sphere of directions.
    let rect = [0, 60, (w as f32 * 0.3) as usize, h / 2 + 30];
    let mut base = panel(c, rect);
    let cam = Camera::orbit(
        base.width,
        base.height,
        [0.0; 3],
        6.0,
        azimuth,
        elevation,
        Lens::Perspective(0.45),
    );
    let mut scene = Scene3::new(cam);
    scene.sphere_wire([0.0; 3], 1.0, 24, palette::grid(), 0.6);
    for (k, d) in seen.iter().enumerate() {
        let p = [d.e1() as f32, d.e2() as f32, d.e3() as f32];
        let size = if fibres[k].2 { 10.0 } else { 5.0 };
        scene.dot(p, Marker::Dot, size, colour(*d));
    }
    scene.draw(&mut base);
    c.blit(&base, rect[0], rect[1]);
    c.text(
        "DIRECTIONS",
        14.0,
        rect[1] as f32 + 14.0,
        11.0,
        palette::ink(),
        Align::Left,
    );

    // The lift: a spinor carried around a circle of directions, drawn up to now.
    let rect = [0, h / 2 + 30, (w as f32 * 0.3) as usize, h];
    let mut inset = panel(c, rect);
    let (_, start, carried) = lift(LIFT_POLAR, 200);
    let cam = Camera::orbit(
        inset.width,
        inset.height,
        [0.0, 0.0, 0.0],
        9.0,
        azimuth,
        elevation,
        Lens::Perspective(0.42),
    );
    let mut scene = Scene3::new(cam);
    let circle: Vec<[f32; 3]> = fibre(start, &turn(200))
        .into_iter()
        .map(|r| xyz(stereographic(r)))
        .collect();
    scene.polyline(&circle, 1.2, palette::grid(), 1.0);
    let upto = ((phase / SWEEP * carried.len() as f32) as usize).clamp(1, carried.len());
    let track: Vec<[f32; 3]> = core::iter::once(start)
        .chain(carried[..upto].iter().copied())
        .map(|r| xyz(stereographic(r)))
        .collect();
    scene.polyline(&track, 1.8, palette::red(), 1.0);
    scene.dot(xyz(stereographic(start)), Marker::Dot, 8.0, palette::sky());
    scene.dot(track[track.len() - 1], Marker::Dot, 8.0, palette::red());
    scene.draw(&mut inset);
    c.blit(&inset, rect[0], rect[1]);
    c.text(
        "BERRY PHASE: BACK ON THE FIBRE,",
        14.0,
        rect[1] as f32 + 8.0,
        10.0,
        palette::ink(),
        Align::Left,
    );
    c.text(
        "TURNED BY HALF THE SOLID ANGLE",
        14.0,
        rect[1] as f32 + 22.0,
        10.0,
        palette::ink(),
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
    use gax::vga3d::Vector;

    /// The largest coefficient of a difference.
    fn gap(a: R, b: R) -> f64 {
        (a - b).c.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    fn close(a: V, b: V, tol: f64) -> bool {
        (a - b).c.iter().all(|v| v.abs() <= tol)
    }

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
            assert!(close(hopf().of(*s).of(*s), direction, 1e-12));
        }
        for r in fibre(spinors[3], &[0.4, 2.1]) {
            assert!(close(hopf().of(r).of(r), direction, 1e-9));
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
                    assert!(close(hopf().of(r).of(r), d, 1e-9));
                }
            }
        }
        let link = linking(&tori[0][0].1, &tori[1][per_circle / 3].1);
        assert!((link - 1.0).abs() < 1e-2, "{link}");
    }

    /// numga's `lift` check: the spinor comes back on its own fibre, turned along it by half
    /// the solid angle the loop encloses.
    #[test]
    fn the_carried_spinor_turns_by_half_the_solid_angle() {
        let polar = 2.2;
        let (_, start, carried) = lift(polar, 200);
        let half_solid_angle = core::f64::consts::PI * (1.0 - polar.cos());
        let turned = start * along_fibre(-half_solid_angle);
        let last = carried[carried.len() - 1];
        assert!(gap(last, turned) < 1e-4, "{}", gap(last, turned));
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
