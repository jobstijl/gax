//! numga's `optics/photoelasticity`: the stress around a traction-free hole in a stretched
//! plate, read through polarized light.
//!
//! The plate is thin, isotropic and infinite, with uniform stress far from its circular hole.
//! Everything is done in spacetime, STA's `x+y+z+t-`. The in-plane stress maps a cut through the
//! plate to the traction across it, `Stress = Traction <- Cut`: a cut is a plane, a vector,
//! extrinsic, which reads how far a direction crosses it, and positions and tractions are
//! directions in the plate, antivectors (`x* y*`), intrinsic like all geometry. Kirsch's solution
//! resolves the remote stress on the cuts across the local radial and tangential directions and
//! writes the stress back as dyads, each weighting a direction by a cut's reading of another,
//! `radial (Cut ∨ radial)`, with no metric; the metric enters where the cut across a direction
//! is meant, and in the principal stresses, the eigenvectors of the stress form
//! `Cut ∨ stress(Cut)` against it.
//!
//! Light's electric field is a bivector through the observer's time; each coefficient holds the
//! field's oscillation as a pair of amplitudes, the field now and a quarter period later, a
//! complex number, so that a delay turns every pair alike. Each axis of a polarizer, or principal
//! direction of the stress, sweeps a plane through the observer's time, and the field's component
//! along the axis is its pairing with that plane, `F ∨ (axis ∨ time)`. The weak stress-optic law
//! delays the field along each principal direction by a share of the period proportional to the
//! stress along it: the plate is the sum over its principal directions of their polarizers, each
//! with its delay. Light travels normal to the plate; absorption, reflection and ray bending are
//! neglected. (The port before numga 35797de turned the Poincaré sphere instead; numga keeps
//! that view in a second notebook.)
//!
//! The animation: a circular polariscope (a polarizer and a quarter-wave plate an eighth of a
//! turn from it, the plate, the same pair crossed) as the load rises from nothing to full tension
//! and back, the fringes running in from the far field and crowding at the hole's sides (left);
//! the principal stress difference over the applied stress, and two crossed linear polariscopes
//! at the same load, whose dark bands (isoclinics) mark where a principal direction lies along
//! the polarizer.

use gax_numga_examples::points::circle;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, ORIGIN2, Point2, Rect, backdrop, caption, colormap, palette,
    plot, run_with, signal::haversine,
};

mod photoelasticity {
    use core::f64::consts::{FRAC_PI_4, FRAC_PI_8, TAU};

    use gax::sta::{Bivector, Even, Vector, intrinsic};
    use gax::{Complex, Unit, vga2d};

    /// Each coefficient's pair of amplitudes, the field now and a quarter period later.
    pub type C = Complex<f64>;
    /// A direction in the plate (numga's `Planar`): a position, or the traction across a cut;
    /// an antivector, intrinsic, of the blades `yzt zxt`.
    pub type Planar = intrinsic::Trivector;
    /// A cut through the plate: a plane, extrinsic, of the blades `x y`.
    pub type Cut = Vector;
    /// The stress: the traction across each cut.
    pub type Stress = intrinsic::Trivector<(Vector,)>;
    /// The electric field, with its two amplitudes.
    pub type Field = Bivector<(), C>;
    /// An optical component: a map on fields.
    pub type OpticalMap = Bivector<(Bivector,), C>;
    /// A real map on fields, before its delays.
    pub type Projector = Bivector<(Bivector,)>;
    pub type Rotor = Unit<Even>;

    pub fn x() -> Cut {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn y() -> Cut {
        Vector::new(0.0, 1.0, 0.0, 0.0)
    }
    pub fn t() -> Vector {
        Vector::new(0.0, 0.0, 0.0, 1.0)
    }

    /// A direction in the plate from its coordinates along `x*` and `y*`.
    pub fn planar(along_x: f64, along_y: f64) -> Planar {
        x().hodge() * along_x + y().hodge() * along_y
    }

    /// The direction in time of an observer at rest with the plate, `−t*`, which pairs to one
    /// with `t`.
    pub fn time() -> intrinsic::Trivector {
        -t().hodge()
    }

    /// A turn in the plate by `angle`, from `x` towards `y`.
    pub fn turn(angle: f64) -> Rotor {
        ((x() ^ y()) * (-angle / 2.0)).exp()
    }

    /// An eighth and a quarter of a turn in the plate.
    pub fn eighth_turn() -> Rotor {
        turn(FRAC_PI_4)
    }
    pub fn quarter_turn() -> Rotor {
        turn(core::f64::consts::FRAC_PI_2)
    }

    /// The plate's cuts in spacetime: VGA2D's `e1, e2` to `x, y`. The principal stresses are
    /// an eigenproblem of the plate's two cuts, not of all of spacetime's vectors.
    pub fn plate_cuts() -> Vector<(vga2d::Vector,)> {
        Vector::from_images([x(), y()])
    }

    /// The square of a direction, `p p`: its length squared, positive for the plate's.
    pub fn squared(p: Planar) -> f64 {
        p.scalar_product(p).s()
    }

    /// The dyad `a (cut ∨ b)`, the cut left open.
    pub fn dyad(a: Planar, b: Planar) -> Stress {
        a * (Cut::slot() & b)
    }

    /// The exterior stress at a position for a circular hole of the given radius under any
    /// uniform symmetric remote stress. At the rim the radial traction vanishes; far away the
    /// stress is the remote one.
    #[allow(clippy::disallowed_methods)] // the distance, a scalar root
    pub fn kirsch(position: Planar, radius: f64, remote: Stress) -> Stress {
        let distance_squared = squared(position);
        let radial = position * distance_squared.sqrt().recip();
        let tangent = quarter_turn() >> radial;
        let ratio = radius * radius / distance_squared;
        // The cuts across the radial and tangential directions: where the metric enters.
        let (radial_cut, tangent_cut) = (across(radial), across(tangent));
        // The remote traction map resolved on those cuts; the mean is half the stress form's
        // trace, read on the plate's orthonormal cuts.
        let reading = |a: Cut, b: Cut| (a & remote.of(b)).s();
        let mean = (reading(x(), x()) + reading(y(), y())) / 2.0;
        let anisotropy = reading(radial_cut, radial_cut) - mean;
        let shear = reading(radial_cut, tangent_cut);
        let radial_stress =
            mean * (1.0 - ratio) + anisotropy * (1.0 - 4.0 * ratio + 3.0 * ratio * ratio);
        let tangential_stress = mean * (1.0 + ratio) - anisotropy * (1.0 + 3.0 * ratio * ratio);
        let shear_stress = shear * (1.0 + 2.0 * ratio - 3.0 * ratio * ratio);
        dyad(radial, radial) * radial_stress
            + dyad(tangent, tangent) * tangential_stress
            + (dyad(radial, tangent) + dyad(tangent, radial)) * shear_stress
    }

    /// The cut across a direction: numga's `dual_inverse`, the inverse of the Hodge dual.
    pub fn across(direction: Planar) -> Cut {
        direction.hodge_inverse()
    }

    /// The principal stresses, from least to greatest, and their directions: the eigenvectors of
    /// the stress form against the metric, on the plate's cuts, are the principal cuts, and the
    /// directions are across them.
    pub fn principal_stresses(stress: Stress) -> ([f64; 2], [Planar; 2]) {
        let cuts = plate_cuts();
        let form = (Cut::slot() & stress.of(Cut::slot())).of_both(cuts, cuts);
        let metric = (Cut::slot() | Cut::slot()).of_both(cuts, cuts);
        let (principal, principal_cuts) = form.eigh(metric);
        (principal, principal_cuts.map(|c| cuts.of(c).hodge()))
    }

    /// The principal stress difference.
    pub fn principal_difference(stress: Stress) -> f64 {
        let (principal, _) = principal_stresses(stress);
        principal[1] - principal[0]
    }

    /// The plane a direction sweeps through the observer's time, which reads the field along it.
    pub fn swept(direction: Planar) -> intrinsic::Bivector {
        direction & time()
    }

    /// The field along a direction, the dual of the plane the direction sweeps in time: the
    /// vacuum's identification of a direction with the field along it, pairing to one with that
    /// plane for a unit direction.
    pub fn field_along(direction: Planar) -> Bivector {
        swept(direction).hodge()
    }

    /// The field along `a`, weighted by the field's pairing with the plane `b` sweeps in time;
    /// with a unit axis twice, the polarizer that keeps the field's component along the axis.
    pub fn polarizer(a: Planar, b: Planar) -> Projector {
        field_along(a) * (Bivector::slot() & swept(b))
    }

    /// A real map on fields with its two amplitudes.
    pub fn complex(m: Projector) -> OpticalMap {
        m.map_coefs(C::real)
    }

    /// The stressed plate: the field along each principal direction delayed by its principal
    /// stress times `delay_per_stress`, the angle of the period a unit of stress delays the
    /// light by through the plate, 2π times the stress-optic coefficient times the thickness
    /// over the wavelength. The mean stress delays both principal directions alike and changes
    /// no intensity.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
    pub fn retarder(stress: Stress, delay_per_stress: f64) -> OpticalMap {
        let (principal, directions) = principal_stresses(stress);
        principal
            .into_iter()
            .zip(directions)
            .map(|(p, d)| complex(polarizer(d, d)).gp(delay(p * delay_per_stress)))
            .sum()
    }

    /// A delay by an angle of the period: it turns both amplitudes alike.
    pub fn delay(angle: f64) -> C {
        (C::i() * C::real(angle)).exp()
    }

    /// The intensity a field puts through a polarizer along a unit axis, in units of a unit
    /// field's: its component along the axis, squared and summed over both its amplitudes.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
    pub fn intensity(field: Field, axis: Planar) -> f64 {
        (field & swept(axis).map_coefs(C::real)).s().norm_sqr()
    }

    /// A polariscope: the field entering the plate, the optics after it, and the analyser's
    /// axis.
    #[derive(Clone, Copy, Debug)]
    pub struct Polariscope {
        pub entering: Field,
        pub exit: OpticalMap,
        pub analyser: Planar,
    }

    impl Polariscope {
        /// Crossed linear polarizers, the first along `axis`.
        pub fn linear(axis: Planar) -> Polariscope {
            Polariscope {
                entering: field_along(axis).map_coefs(C::real),
                exit: complex(Bivector::slot()),
                analyser: quarter_turn() >> axis,
            }
        }

        /// Crossed circular polarizers: a linear polarizer along `axis` and a quarter-wave plate
        /// an eighth of a turn from it, the plate, and the same pair crossed. The two
        /// quarter-wave plates undo each other, so the view is dark without stress.
        pub fn circular(axis: Planar) -> Polariscope {
            let fast = eighth_turn() >> axis;
            let slow = quarter_turn() >> fast;
            // A quarter-wave plate delays the field along its slow axis by a quarter period;
            // crossed, its fast and slow axes trade places.
            let quarter_wave = |fast, slow| {
                complex(polarizer(fast, fast)) + complex(polarizer(slow, slow)).gp(C::i())
            };
            Polariscope {
                entering: quarter_wave(fast, slow).of(field_along(axis).map_coefs(C::real)),
                exit: quarter_wave(slow, fast),
                analyser: quarter_turn() >> axis,
            }
        }

        /// The share of light through a plate.
        #[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
        pub fn through(&self, plate: OpticalMap) -> f64 {
            intensity(self.exit.of(plate.of(self.entering)), self.analyser)
        }

        /// What a stressed plate passes at any load, read once: per principal direction, its
        /// stress and the amplitude the polariscope reads through its polarizer. Scaling the
        /// load keeps the principal directions, so only the delays grow.
        pub fn reading(&self, stress: Stress) -> Reading {
            let (principal, directions) = principal_stresses(stress);
            let amplitudes = directions.map(|d| {
                let leaving = self.exit.of(complex(polarizer(d, d)).of(self.entering));
                (leaving & swept(self.analyser).map_coefs(C::real)).s()
            });
            Reading {
                principal,
                amplitudes,
            }
        }
    }

    /// A polariscope's reading of one stressed plate: see [`Polariscope::reading`].
    #[derive(Clone, Copy, Debug)]
    pub struct Reading {
        principal: [f64; 2],
        amplitudes: [C; 2],
    }

    impl Reading {
        /// The share of light through the plate at `load` times its stress.
        pub fn at(&self, load: f64, delay_per_stress: f64) -> f64 {
            let [a, b] = [0, 1]
                .map(|k| self.amplitudes[k] * delay(self.principal[k] * load * delay_per_stress));
            (a + b).norm_sqr()
        }
    }

    // --- the scenes -------------------------------------------------------------------

    /// The hole's radius, mm.
    pub const RADIUS: f64 = 10.0;
    /// Half the width of the window into the infinite plate, mm.
    pub const EXTENT: f64 = 30.0;
    /// Pixels across the window, each way.
    pub const PIXELS: usize = 200;
    /// The remote tension along `x`, MPa.
    pub const TENSION: f64 = 30.0;
    /// The plate's thickness, mm.
    const THICKNESS: f64 = 1.0;
    /// The light's wavelength, mm.
    const WAVELENGTH: f64 = 550e-6;
    /// The stress-optic coefficient, per MPa: an illustrative sensitivity.
    const STRESS_OPTIC: f64 = 5.5e-5;
    /// The angle of the period a unit of stress delays the light by.
    pub const DELAY_PER_STRESS: f64 = TAU * STRESS_OPTIC * THICKNESS / WAVELENGTH;

    /// A tension along `x`: the traction along `x*` on each cut, by how far `x*` crosses it.
    pub fn uniaxial(tension: f64) -> Stress {
        let pull = x().hodge();
        dyad(pull, pull) * tension
    }

    /// The remote stress.
    pub fn remote() -> Stress {
        uniaxial(TENSION)
    }

    /// The pixel centres of the window, row by row from the bottom.
    pub fn window() -> Vec<Planar> {
        let centre = |i: usize| (i as f64 + 0.5) * (2.0 * EXTENT / PIXELS as f64) - EXTENT;
        (0..PIXELS * PIXELS)
            .map(|k| planar(centre(k % PIXELS), centre(k / PIXELS)))
            .collect()
    }

    /// The exterior stress at each pixel centre; the pixels inside the hole carry the formula's
    /// values and are covered when drawn.
    pub fn stress_field() -> Vec<Stress> {
        window()
            .into_iter()
            .map(|p| kirsch(p, RADIUS, remote()))
            .collect()
    }

    /// The linear polariscopes' polarizer angles in the plate: along `x`, and an eighth of a
    /// quarter turn from it.
    pub const POLARIZER_ANGLES: [f64; 2] = [0.0, FRAC_PI_8];

    /// Two crossed linear polariscopes and a circular one: without stress, every view is dark.
    pub fn polariscopes() -> [Polariscope; 3] {
        let axis = |angle: f64| turn(angle) >> x().hodge();
        [
            Polariscope::linear(axis(POLARIZER_ANGLES[0])),
            Polariscope::linear(axis(POLARIZER_ANGLES[1])),
            Polariscope::circular(x().hodge()),
        ]
    }
}

use photoelasticity::*;

/// One loop: from no load to full tension and back, numga's 180 frames of 40 ms.
const SECONDS: f32 = 7.2;

/// What the frames share: the three polariscopes' readings of the plate, and the principal
/// stress difference over the applied stress, at each pixel.
struct Plate {
    readings: Vec<[Reading; 3]>,
    difference: Vec<f64>,
    /// The largest difference over the plate, outside the hole.
    largest: f64,
}

fn plate() -> Plate {
    let stress = stress_field();
    let difference: Vec<f64> = stress
        .iter()
        .map(|s| principal_difference(*s) / TENSION)
        .collect();
    let outside = window().into_iter().map(|p| squared(p) >= RADIUS * RADIUS);
    let largest = difference
        .iter()
        .zip(outside)
        .filter(|(_, out)| *out)
        .fold(0.0f64, |m, (d, _)| m.max(*d));
    Plate {
        readings: stress
            .iter()
            .map(|s| polariscopes().map(|view| view.reading(*s)))
            .collect(),
        difference,
        largest,
    }
}

/// The load at time `t`: a smooth cycle from nothing to the full tension and back.
fn load(t: f32) -> f64 {
    let turn = f64::from(t / SECONDS) * core::f64::consts::TAU;
    haversine(turn)
}

/// A value of the window's pixel grid at a point of the plate (mm), interpolated between the
/// pixel centres.
fn sample(values: &[f64], p: Point2) -> f64 {
    let [x, y] = p.to_euclidean();
    let index = |v: f32| (f64::from(v) + EXTENT) / (2.0 * EXTENT) * PIXELS as f64 - 0.5;
    plot::bilinear(values, PIXELS, index(x), index(y))
}

/// An image of the plate's window in `rect`, `shade` of each value, with the hole drawn over it
/// and a title above.
fn plate_panel(
    c: &mut Canvas,
    rect: Rect,
    title: &str,
    values: &[f64],
    shade: impl Fn(f64) -> Light + Sync,
) {
    let ax = Axes::equal(rect, ORIGIN2, EXTENT as f32);
    ax.image(c, 1, |p| Some(shade(sample(values, p))));
    let hole = circle(ORIGIN2, RADIUS as f32, 96);
    ax.fill(c, &hole, palette::BOTTOM, 1.0);
    ax.polyline(c, &hole, 1.0, palette::GRID.faded(2.0));
    let r = ax.rect;
    c.polyline(
        &[r.lo, r.top_right(), r.hi, r.bottom_left()],
        1.0,
        palette::GRID,
        true,
    );
    let size = (rect.height() / 14.0).clamp(5.0, 12.0);
    let above = r.lo + Point2::direction(0.0, -size * 0.6);
    c.text(title, above, size, palette::INK, Align::Left);
}

/// Transmitted light as a light: white-blue on the dark plate, its intensity the share passed.
fn light_through(share: f64) -> Light {
    palette::INK.faded(share.clamp(0.0, 1.0) as f32 * 0.85)
}

/// The colour scale of the stress difference, from none to its largest value, as a bar at the
/// left of `rect` with its ends labelled.
fn colour_bar(c: &mut Canvas, rect: Rect, largest: f64, size: f32) {
    let bar = rect.part(0.08, 0.05, 0.2, 0.95);
    let ax = Axes::new(bar, [0.0, 1.0], [0.0, largest as f32]);
    ax.image(c, 1, |p| {
        let [_, v] = p.to_euclidean();
        Some(colormap::viridis(v / largest as f32))
    });
    let beside = |fy: f32| ax.px(ax.at(1.0, fy)) + Point2::direction(size * 0.6, size * 0.4);
    let ink = palette::INK;
    c.text(
        &format!("{largest:.1}, AT THE RIM"),
        beside(1.0),
        size,
        ink,
        Align::Left,
    );
    c.text(
        "1, FAR AWAY",
        beside(1.0 / largest as f32),
        size,
        ink,
        Align::Left,
    );
    c.text("0", beside(0.0), size, ink, Align::Left);
}

fn draw(c: &mut Canvas, p: &Plate, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let load = load(t);
    let views: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            p.readings
                .iter()
                .map(|r| r[k].at(load, DELAY_PER_STRESS))
                .collect()
        })
        .collect();
    let top = h * 0.18;
    let side = (h - top - h * 0.06).min(w * 0.5);
    let left = w * 0.03;
    let big = Rect::new(left, top, left + side, top + side);
    plate_panel(
        c,
        big,
        "CIRCULAR POLARISCOPE: FRINGES",
        &views[2],
        light_through,
    );
    let readout = format!("LOAD {:3.0}% OF {TENSION:.0} MPA", 100.0 * load);
    let below = big.bottom_middle() + Point2::direction(0.0, h * 0.045);
    c.text(
        &readout,
        below,
        (h / 40.0).clamp(6.0, 13.0),
        palette::YELLOW,
        Align::Center,
    );

    let right = Rect::new(left + side + w * 0.04, top, w * 0.98, top + side);
    let cell = |i: usize, j: usize| {
        right
            .part(
                0.5 * i as f32,
                0.5 * j as f32,
                0.5 * (i + 1) as f32,
                0.5 * (j + 1) as f32,
            )
            .inset(8.0, 24.0, 8.0, 4.0)
    };
    let difference = |v: f64| colormap::viridis((v / p.largest) as f32);
    plate_panel(
        c,
        cell(0, 0),
        "STRESS DIFFERENCE / TENSION",
        &p.difference,
        difference,
    );
    colour_bar(c, cell(1, 0), p.largest, (h / 44.0).clamp(5.0, 12.0));
    let names = ["LINEAR AT 0 DEG", "LINEAR AT 22.5 DEG"];
    for (k, name) in names.into_iter().enumerate() {
        plate_panel(c, cell(k, 1), name, &views[k], light_through);
    }
    caption(
        c,
        "PHOTOELASTICITY: STRESS AROUND A HOLE IN POLARIZED LIGHT",
        "KIRSCH STRESS FROM CUTS TO TRACTIONS; THE PLATE A SUM OF DELAYED POLARIZERS (STA)",
    );
}

fn main() {
    run_with(
        Anim::new("photoelasticity", SECONDS).size(960, 540),
        plate,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::photoelasticity::*;
    use core::f64::consts::{PI, TAU};
    use gax::ApproxEq;
    use gax::sta::{Bivector, Vector};
    use gax_numga_examples::signal::wave;

    const ROUND_OFF: f64 = 1e-10;

    /// A uniform symmetric stress from its normal stresses along `x` and `y` and its shear.
    fn uniform_stress(along_x: f64, along_y: f64, shear: f64) -> Stress {
        let (x, y) = (x().hodge(), y().hodge());
        dyad(x, x) * along_x + dyad(y, y) * along_y + (dyad(x, y) + dyad(y, x)) * shear
    }

    /// The largest difference between two complex maps' coefficients.
    fn gap(a: OpticalMap, b: OpticalMap) -> f64 {
        a.c.iter()
            .flatten()
            .zip(b.c.iter().flatten())
            .map(|(p, q)| (*p - *q).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn free_rim_and_uniaxial_stress_concentration() {
        let (radius, tension, samples) = (1.4, 2.3, 73);
        let remote = uniform_stress(1.7, -0.4, 0.6);
        // The cuts along the rim, facing out, and the directions across them.
        for k in 0..samples {
            let cut = turn(TAU * k as f64 / samples as f64) >> x();
            let normal = cut.hodge();
            let traction = kirsch(normal * radius, radius, remote).of(cut);
            assert!(traction.max_abs_diff(&Planar::zero()) < ROUND_OFF);
        }
        // The hoop stress at the rim: minus the tension where the load leaves it, three times
        // it at the sides.
        for (end_cut, expected) in [(x(), -tension), (y(), 3.0 * tension)] {
            let tangent_cut = quarter_turn() >> end_cut;
            let rim = kirsch(end_cut.hodge() * radius, radius, uniaxial(tension));
            assert!(((tangent_cut & rim.of(tangent_cut)).s() - expected).abs() < ROUND_OFF);
        }
    }

    #[test]
    fn stress_rotates_with_load_and_positions_and_recovers_remote_field() {
        let (radius, far_distance) = (0.8, 1e7 * 0.8);
        let turn = turn(0.74);
        let remote = uniform_stress(2.1, 0.3, -0.5);
        let moved_remote = turn >> remote.of(turn << Cut::slot());
        let cuts = [x() + y() * 0.3, y() - x() * 1.2];
        for p in [planar(1.2, 0.4), planar(-1.6, 0.7), planar(0.3, -2.1)] {
            let stress = kirsch(p, radius, remote);
            let moved = kirsch(turn >> p, radius, moved_remote);
            let rotated = turn >> stress.of(turn << Cut::slot());
            assert!(moved.max_abs_diff(&rotated) < ROUND_OFF);
            // Symmetric: the stress form reads two cuts the same either way round.
            let form = |a: Vector, b: Vector| (a & stress.of(b)).s();
            assert!((form(cuts[0], cuts[1]) - form(cuts[1], cuts[0])).abs() < ROUND_OFF);
            #[allow(clippy::disallowed_methods)] // a distance, a scalar root
            let far_position = p * (far_distance / squared(p).sqrt());
            let far = kirsch(far_position, radius, remote);
            assert!(far.max_abs_diff(&remote) < ROUND_OFF);
        }
    }

    #[test]
    fn stress_has_zero_divergence_away_from_the_hole() {
        let (radius, step) = (0.7, 1e-4 * 0.7);
        let remote = uniform_stress(1.8, -0.2, 0.4);
        // Steps along x* and y*, and the cuts that read them, paired with no metric.
        for p in [planar(1.1, 0.3), planar(-1.3, 0.8), planar(0.4, -1.7)] {
            let divergence = [x(), y()].into_iter().fold(Planar::zero(), |sum, cut| {
                let d = cut.hodge();
                let change =
                    kirsch(p + d * step, radius, remote) - kirsch(p - d * step, radius, remote);
                sum + change.of(cut) * (1.0 / (2.0 * step))
            });
            // This tolerance tests the centred derivative's truncation error.
            assert!(divergence.max_abs_diff(&Planar::zero()) < 1e-7);
        }
    }

    #[test]
    fn mean_stress_delays_every_direction_alike() {
        let delay_per_stress = 83.0 * PI + 0.41;
        let mean = 3.7;
        let base = uniform_stress(2.4, -0.8, 0.5);
        // An isotropic stress: the mean times the metric, the traction straight across every
        // cut.
        let isotropic = Cut::slot().hodge() * mean;
        let plates = [base, base + isotropic, isotropic].map(|s| retarder(s, delay_per_stress));
        let delayed = delay(delay_per_stress * mean);
        let (x, y) = (x().hodge(), y().hodge());
        // The polarizers along both directions of the plate together pass every field in it.
        let unstressed = complex(polarizer(x, x) + polarizer(y, y));

        // The mean stress only delays the whole field, and alone leaves its polarization as it
        // was.
        assert!(gap(plates[1], plates[0].gp(delayed)) < ROUND_OFF);
        assert!(gap(plates[2], unstressed.gp(delayed)) < ROUND_OFF);
    }

    #[test]
    fn linear_and_circular_dark_fields_match_the_retardance_law() {
        let orientations = 19;
        let retardances = [0.0, 0.01, 0.83, PI, TAU, 7.1, 80.0 * PI + 0.4];
        let axis = x().hodge();
        let (linear, circular) = (Polariscope::linear(axis), Polariscope::circular(axis));
        for k in 0..orientations {
            let angle = PI * k as f64 / orientations as f64;
            let turn = turn(angle);
            let base = uniform_stress(1.0, 0.0, 0.0);
            let stress = turn >> base.of(turn << Cut::slot());
            for retardance in retardances {
                let plate = retarder(stress * retardance, 1.0);
                let phase_intensity = wave(retardance / 2.0).powi(2);
                let expected_linear = wave(2.0 * angle).powi(2) * phase_intensity;
                assert!((linear.through(plate) - expected_linear).abs() < ROUND_OFF);
                assert!((circular.through(plate) - phase_intensity).abs() < ROUND_OFF);
            }
        }
    }

    #[test]
    fn loading_cycle_returns_to_dark_and_matches_the_loaded_plate() {
        let (radius, delay_per_stress) = (0.8, 13.7);
        let loads = [0.0, 0.5, 1.0, 0.5, 0.0];
        let remote = uniform_stress(2.1, 0.3, -0.5);
        let circular = Polariscope::circular(x().hodge());
        for p in [planar(1.2, 0.4), planar(-1.6, 0.7), planar(0.3, -2.1)] {
            let stress = kirsch(p, radius, remote);
            let frames = loads.map(|l| circular.through(retarder(stress * l, delay_per_stress)));
            // Recompute the stress at every remote load, rather than scaling the reference one.
            let expected = loads.map(|l| {
                let scaled = kirsch(p, radius, remote * l);
                circular.through(retarder(scaled, delay_per_stress))
            });
            // The animation's reading, the amplitudes read once and delayed per load.
            let reading = circular.reading(stress);
            for (k, f) in frames.iter().enumerate() {
                assert!((f - expected[k]).abs() < ROUND_OFF);
                assert!((f - frames[loads.len() - 1 - k]).abs() < ROUND_OFF);
                assert!((f - reading.at(loads[k], delay_per_stress)).abs() < ROUND_OFF);
            }
            assert!(frames[0].abs() < ROUND_OFF);
        }
    }

    /// The optics' pieces: a polarizer keeps what it passes, the field along a unit direction
    /// pairs to one with the plane it sweeps, a direction's cut is across it, and the plate's
    /// directions square to their lengths.
    #[test]
    fn polarizers_project_and_directions_pair() {
        let axis = turn(0.3) >> x().hodge();
        let p = polarizer(axis, axis);
        assert!(p.of(p).max_abs_diff(&p) < ROUND_OFF);
        assert!(((field_along(axis) & swept(axis)).s() - 1.0).abs() < ROUND_OFF);
        let cut = across(axis);
        assert!(cut.hodge().max_abs_diff(&axis) < ROUND_OFF);
        assert!(((cut & axis).s() - 1.0).abs() < ROUND_OFF);
        assert!((squared(planar(3.0, 4.0)) - 25.0).abs() < ROUND_OFF);
        // A field across the axis does not pass.
        let across_field = field_along(quarter_turn() >> axis);
        assert!(p.of(across_field).max_abs_diff(&Bivector::zero()) < ROUND_OFF);
    }

    /// The scene against numga's numbers: the principal stress difference and the three
    /// polariscopes at the pixel centre `(21.15, 0.15)` mm (numga 35797de, as before it).
    #[test]
    fn scene_matches_numga() {
        let p = planar(170.5 * 0.3 - EXTENT, 100.5 * 0.3 - EXTENT);
        let stress = kirsch(p, RADIUS, remote());
        assert!((principal_difference(stress) - 14.387127181521555).abs() < 1e-9);
        let plate = retarder(stress, DELAY_PER_STRESS);
        let numga = [
            5.636241310714593e-04,
            5.049874138896087e-01,
            9.633843147899731e-01,
        ];
        for (view, expected) in polariscopes().into_iter().zip(numga) {
            assert!((view.through(plate) - expected).abs() < 1e-9);
            assert!((view.reading(stress).at(1.0, DELAY_PER_STRESS) - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::plate(), 0.5);
    }
}
