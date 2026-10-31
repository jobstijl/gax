//! numga's `orientation/wu_experiment`: the electrons from polarized cobalt-60, the mirror
//! image of the experiment, and that mirror image built for real, in VGA3D.
//!
//! In 1957 Chien-Shiung Wu polarized cobalt-60 nuclei with the current in a coil and counted the
//! electrons their decay emits: more left against the nuclear spin than along it. A spin is an
//! angular momentum, intrinsic like every momentum (ADR-051): the patch the coil's current
//! circulates around, the dual of the coil's plane, an intrinsic vector (numga's antibivector),
//! which reverses with the current. A direction is an intrinsic bivector (numga's antivector).
//! The rate of electrons along a unit direction is one plus the decay's asymmetry times the
//! spin's pairing with the direction, `spin & direction`. Both are intrinsic, so their pairing
//! is an intrinsic scalar, and for the rate to be an ordinary number the asymmetry is an
//! intrinsic scalar too: a constant with a sense of its own, measured against a handedness.
//!
//! A mirror held upright beside the apparatus, the plane `x = 0`, reflects every part of the
//! apparatus with `>>`, and shows electrons leaving along the spin. Built for real, the mirrored
//! apparatus keeps the decay law, whose constant no mirror of the apparatus reverses, and sends
//! them against its spin, as the original does: the decay tells the two apart. Reflecting the
//! asymmetry along with the apparatus (`mirror >> asymmetry`, which negates an intrinsic scalar)
//! would give the mirror image back.
//!
//! The animation shows numga's three panels, the apparatus, its mirror image and the mirrored
//! apparatus built for real: the rate along each direction as a polar curve in the plane that
//! holds the coil's axis, the coil where that plane cuts it (its current going into the page
//! through the wire with the cross, out through the one with the dot), and electrons streaming
//! out along each direction as often as the rate says.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, ORIGIN2, Point2, Rect, backdrop, caption, label_under,
    palette, run_with,
};

mod wu {
    use gax::vga3d::{Bivector, Vector, intrinsic};

    /// A direction: an intrinsic bivector, numga's antivector.
    pub type Direction = intrinsic::Bivector;
    /// A spin, the patch its current circulates around: an intrinsic vector, numga's
    /// antibivector.
    pub type Spin = intrinsic::Vector;
    /// A constant of the decay with a sense of its own: an intrinsic scalar.
    pub type Asymmetry = intrinsic::Scalar;
    /// A mirror, by its normal.
    pub type Mirror = Vector;

    /// The decay's asymmetry: minus one for cobalt-60, times the electrons' speed over light's,
    /// taken as 0.6, measured against the lab's handedness `x ^ y ^ z`.
    pub const DECAY_ASYMMETRY: f64 = -0.6;
    /// The directions around the plane holding the spin's axis, the last closing the loop on
    /// the first.
    pub const DIRECTIONS: usize = 181;

    /// The decay's asymmetry.
    pub fn asymmetry() -> Asymmetry {
        Asymmetry::new(DECAY_ASYMMETRY)
    }

    /// Straight up: the direction along `z`, the dual of the plane `z = 0`.
    pub fn up() -> Direction {
        Vector::new(0.0, 0.0, 1.0).dual()
    }

    /// The patch the coil's current circulates around, counterclockwise seen from above: the
    /// dual of the plane `xy`.
    pub fn spin() -> Spin {
        Bivector::new(0.0, 0.0, 1.0).dual()
    }

    /// The mirror: the plane `x = 0`, holding the coil's axis.
    pub fn mirror() -> Mirror {
        Vector::new(1.0, 0.0, 0.0)
    }

    /// The direction at `angle` from up, turned from `z` toward `x`, around the plane that holds
    /// the spin's axis.
    pub fn direction(angle: f64) -> Direction {
        Bivector::new(0.0, 1.0, 0.0).gp(-angle / 2.0).exp() >> up()
    }

    /// The rate of electrons along a unit direction: that of unpolarized nuclei, one, departed
    /// from by the asymmetry times the spin's pairing with the direction.
    pub fn rate(asymmetry: Asymmetry, spin: Spin, direction: Direction) -> f64 {
        1.0 + (asymmetry * (spin & direction)).s()
    }

    /// The three cases: the apparatus, its mirror image, and the mirrored apparatus built for
    /// real.
    pub struct Experiment {
        /// Each case's spin.
        pub spins: [Spin; 3],
        /// The directions around the plane, the last closing the loop on the first.
        pub directions: Vec<Direction>,
        /// The rate along each direction, `[case][direction]`.
        pub rates: [Vec<f64>; 3],
    }

    /// What each case sends along a direction: the apparatus; its mirror image, which is what
    /// the apparatus sends along the direction's mirror image; and the mirrored apparatus, its
    /// coil the mirror image, the decay law the same.
    pub fn case_rate(case: usize, direction: Direction) -> f64 {
        let (asymmetry, spin, mirror) = (asymmetry(), spin(), mirror());
        match case {
            0 => rate(asymmetry, spin, direction),
            1 => rate(asymmetry, spin, mirror >> direction),
            _ => rate(asymmetry, mirror >> spin, direction),
        }
    }

    /// numga's scenario: the three cases along each direction around the plane.
    pub fn experiment() -> Experiment {
        let spin = spin();
        let directions: Vec<Direction> = (0..DIRECTIONS)
            .map(|k| core::f64::consts::TAU * k as f64 / (DIRECTIONS - 1) as f64)
            .map(direction)
            .collect();
        let rates = [0, 1, 2].map(|case| directions.iter().map(|d| case_rate(case, *d)).collect());
        Experiment {
            spins: [spin, mirror() >> spin, mirror() >> spin],
            directions,
            rates,
        }
    }
}

use gax::vga3d::Vector;
use wu::*;

/// One loop of the electrons' streams.
const SECONDS: f32 = 6.0;
/// The coil's radius, in the panels' units of rate.
const COIL_RADIUS: f32 = 0.55;
/// The streams of electrons drawn, around the plane.
const STREAMS: usize = 36;
/// Electrons along a stream per loop for a rate of one.
const ELECTRONS: f64 = 6.0;

/// A direction or a polar curve's point in the panel's plane: across along `x`, up along `z`,
/// read by pairing it with the planes `x = 0` and `z = 0`.
fn planar(d: Direction) -> Point2 {
    let read = |n: Vector| (n & d).s() as f32;
    Point2::xy(
        read(Vector::new(1.0, 0.0, 0.0)),
        read(Vector::new(0.0, 0.0, 1.0)),
    )
}

/// The sense of a spin's current: its pairing with up, one where it circulates
/// counterclockwise seen from above.
fn sense(spin: Spin) -> f32 {
    (spin & up()).s() as f32
}

/// One case in `rect`: its emission curve, the coil's axis, the coil's wires, and the electrons
/// streaming out at time `t`.
fn case_panel(c: &mut Canvas, rect: Rect, ex: &Experiment, k: usize, t: f32, reach: f32) {
    let ax = Axes::equal(rect, ORIGIN2, reach);
    let curve: Vec<Point2> = ex
        .directions
        .iter()
        .zip(&ex.rates[k])
        .map(|(d, r)| planar(d.gp(*r)))
        .collect();
    ax.dashed(
        c,
        &[Point2::xy(0.0, -reach), Point2::xy(0.0, reach)],
        1.0,
        5.0,
        palette::GRID.faded(2.5),
    );
    ax.fill(c, &curve, palette::ORANGE, 0.25);
    ax.polyline(c, &curve, 1.8, palette::ORANGE);
    // The electrons: along each stream, as many per loop as the rate there says, moving out at
    // one speed, so that they crowd where the rate is high.
    let electrons: Vec<Point2> = (0..STREAMS)
        .flat_map(|s| {
            let angle = core::f64::consts::TAU * s as f64 / STREAMS as f64;
            let d = direction(angle);
            let count = (case_rate(k, d) * ELECTRONS).round().max(1.0);
            let phase = (f64::from(t / SECONDS) * count).fract();
            (0..count as usize).map(move |j| d.gp((j as f64 + phase) / count * f64::from(reach)))
        })
        .map(planar)
        .filter(|p| p.norm() > COIL_RADIUS * 0.4)
        .collect();
    ax.scatter(c, &electrons, Marker::Dot, 2.4, palette::SKY.faded(0.6));
    // With x to the right and z up, +y runs into the page: a current counterclockwise seen from
    // above goes into the page at +x and comes out of it at -x.
    let inward = sense(ex.spins[k]).signum() * COIL_RADIUS;
    let wire = 0.08 * reach;
    let scale = ax.scale();
    for at in [Point2::xy(inward, 0.0), Point2::xy(-inward, 0.0)] {
        c.disk(ax.px(at), wire * scale, palette::BOTTOM);
        c.ring(ax.px(at), wire * scale, 1.6, palette::INK);
    }
    gax_numga_examples::plot::mark(
        c,
        ax.px(Point2::xy(inward, 0.0)),
        Marker::Cross,
        wire * scale * 1.1,
        palette::INK,
    );
    c.disk(
        ax.px(Point2::xy(-inward, 0.0)),
        wire * scale * 0.3,
        palette::INK,
    );
}

fn draw(c: &mut Canvas, ex: &Experiment, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let largest = ex
        .rates
        .iter()
        .flatten()
        .fold(0.0f64, |m, r| m.max(r.abs()));
    let reach = 1.25 * largest as f32;
    let names = [
        "THE APPARATUS",
        "ITS MIRROR IMAGE",
        "THE MIRRORED APPARATUS, BUILT",
    ];
    for (k, name) in names.into_iter().enumerate() {
        let rect = screen
            .part(k as f32 / 3.0, 0.13, (k + 1) as f32 / 3.0, 0.84)
            .inset(10.0, 4.0, 10.0, 4.0);
        case_panel(c, rect, ex, k, t, reach);
        label_under(c, rect, name, 20.0, 0.9);
    }
    let rates = |k: usize| {
        let [up, down] = [up(), -up()].map(|d| case_rate(k, d));
        format!("UP {up:.1}, DOWN {down:.1}")
    };
    for k in 0..3 {
        let rect = screen.part(k as f32 / 3.0, 0.9, (k + 1) as f32 / 3.0, 1.0);
        let at = rect.top_middle() + Point2::direction(0.0, 22.0);
        c.text(&rates(k), at, 11.0, palette::YELLOW, Align::Center);
    }
    caption(
        c,
        "THE WU EXPERIMENT",
        "SPIN & DIRECTION IS INTRINSIC, SO THE ASYMMETRY IS TOO: NO MIRROR OF THE APPARATUS REVERSES IT (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("wu experiment", SECONDS).size(960, 540),
        experiment,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::wu::*;
    use gax_numga_examples::testing::{close, near};

    /// numga's test: around the circle the rate averages to that of unpolarized nuclei (the
    /// closing sample repeats the first), the mirror image is the original read backwards, and
    /// the mirrored apparatus differs from the mirror image by more than one somewhere.
    #[test]
    fn emission_and_its_mirror_image() {
        let ex = experiment();
        let [original, image, built] = &ex.rates;
        let mean = original[..DIRECTIONS - 1].iter().sum::<f64>() / (DIRECTIONS - 1) as f64;
        assert!(close(mean, 1.0, 1e-12), "{mean}");
        for (i, r) in image.iter().zip(original.iter().rev()) {
            assert!(close(*i, *r, 1e-12));
        }
        let largest = built
            .iter()
            .zip(image)
            .fold(0.0f64, |m, (b, i)| m.max((b - i).abs()));
        assert!(largest > 1.0, "{largest}");
    }

    /// numga's scenario checks: with the asymmetry mirrored along, the mirrored apparatus gives
    /// the mirror image; as it is, it sends electrons where the mirror image does not; the
    /// original sends them mostly downward, against its spin, the mirrored apparatus upward.
    #[test]
    fn the_decay_law_holds_an_intrinsic_constant() {
        let ex = experiment();
        let [_, image, built] = &ex.rates;
        let (m, spin) = (mirror(), spin());
        for ((d, i), b) in ex.directions.iter().zip(image).zip(built) {
            let mirrored_law = rate(m >> asymmetry(), m >> spin, *d);
            assert!(close(mirrored_law, *i, 1e-12));
            assert!(close(b + i, 2.0, 1e-12));
        }
        let [original_down, original_up] = [-up(), up()].map(|d| rate(asymmetry(), spin, d));
        let [built_down, built_up] = [-up(), up()].map(|d| rate(asymmetry(), m >> spin, d));
        assert!(original_down > original_up && built_up > built_down);
    }

    /// numga's numbers: the spins `[1, -1, -1]` along `z`, the directions and the rates.
    #[test]
    fn rates_are_numgas() {
        let ex = experiment();
        for (spin, z) in ex.spins.iter().zip([1.0, -1.0, -1.0]) {
            assert!(near(*spin, Spin::new(0.0, 0.0, z), 1e-12), "{spin:?}");
        }
        for (k, rates) in [
            (0, [0.4, 0.4, 1.6]),
            (10, [0.43618443, 0.43618443, 1.56381557]),
            (45, [1.0, 1.0, 1.0]),
            (90, [1.6, 1.6, 0.4]),
            (135, [1.0, 1.0, 1.0]),
        ] {
            for (case, r) in rates.into_iter().enumerate() {
                assert!(close(ex.rates[case][k], r, 1e-8), "{k} {case}");
            }
        }
        let numga = [
            (0, [0.0, 1.0]),
            (10, [0.342020143, 0.939692621]),
            (45, [1.0, 0.0]),
        ];
        for (k, [yz, xy]) in numga {
            assert!(near(ex.directions[k], Direction::new(yz, 0.0, xy), 1e-9));
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &experiment(), 1.0);
    }
}
