//! numga's `orientation/mechanical_work`: a crank driven by a push at its handle, beside its
//! mirror image, in PGA2D.
//!
//! The handle is a point and the push a direction, both intrinsic (ADR-051); the crank's rate
//! of turning is a `Twist`, the extrinsic point where two lines meet, its pivot. The handle's
//! velocity is its commutator with the rate. Joining the handle to the push gives the load on
//! the crank, a `Forque` (`Point & Point` is an intrinsic line), and pairing the load with the
//! rate, `forque & twist`, gives the power, an extrinsic scalar.
//!
//! The mirror turns the crank the other way, so its twist changes sign, while the handle and
//! the push are carried over as they are. gax's `>>` gives the two orientations opposite signs
//! under a reflection (the guide's "Versors respect every product, the join included"), so the
//! load built from the reflected handle and push is the reflected load, and the reflected
//! power is the power: the same positive `1.35` on both sides, with no sign to fix by hand.
//!
//! The animation turns both cranks once: velocities in cyan, pushes in orange with the load's
//! line of action, each twist's sense of turning in green, the mirror dashed, and each crank's
//! power under it.

use gax::pga2d::Point;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Point2, backdrop, caption, palette, run,
};

mod work {
    use gax::pga2d::{Forque, Line, Point, Scalar, Twist};

    /// numga's crank: the pivot's distance from the mirror, the arm's length, the rate of
    /// turning and the strength of the push.
    pub const SEPARATION: f64 = 1.65;
    pub const RADIUS: f64 = 0.9;
    pub const SPEED: f64 = 1.0;
    pub const PUSH: f64 = 1.5;

    /// The mirror: the line `x = 0`.
    pub fn mirror() -> Line {
        Line::new(1.0, 0.0, 0.0)
    }

    /// A crank: its pivot, its handle at the start, the push on the handle at the start (a
    /// direction), and its rate of turning.
    #[derive(Clone, Copy, Debug)]
    pub struct Crank {
        pub pivot: Point,
        pub handle: Point,
        pub push: Point,
        pub rate: Twist,
    }

    /// The crank left of the mirror: the handle to the right of the pivot, pushed upwards,
    /// turning counterclockwise about the meet of the lines `x = -1.65` and `y = 0`.
    pub fn crank() -> Crank {
        let pivot_lines = Line::new(1.0, 0.0, SEPARATION) ^ Line::new(0.0, 1.0, 0.0);
        Crank {
            pivot: Point::xy(-SEPARATION, 0.0),
            handle: Point::xy(-SEPARATION + RADIUS, 0.0),
            push: Point::direction(0.0, PUSH),
            rate: pivot_lines.gp(SPEED),
        }
    }

    /// The crank reflected in a mirror: every ingredient by the same `>>`.
    pub fn reflected(mirror: Line, c: Crank) -> Crank {
        Crank {
            pivot: mirror >> c.pivot,
            handle: mirror >> c.handle,
            push: mirror >> c.push,
            rate: mirror >> c.rate,
        }
    }

    /// The crank after turning for `time`: the handle, its velocity, the push (which turns
    /// with the handle), the load on the crank and the power it supplies.
    #[derive(Clone, Copy, Debug)]
    pub struct State {
        pub handle: Point,
        pub velocity: Point,
        pub push: Point,
        pub load: Forque,
        pub power: Scalar,
    }

    pub fn turn(c: Crank, time: f64) -> State {
        let pose = c.rate.gp(-time / 2.0).exp();
        let (handle, push) = (pose >> c.handle, pose >> c.push);
        // The handle's velocity follows from the rate; the load from where the push acts.
        let velocity = handle.commutator(c.rate);
        let load: Forque = handle & push;
        State {
            handle,
            velocity,
            push,
            load,
            power: load & c.rate,
        }
    }

    /// The time of one turn.
    pub fn period() -> f64 {
        core::f64::consts::TAU / SPEED
    }
}

use work::*;

/// numga's animation: 36 frames of 85 milliseconds for one turn.
const SECONDS: f32 = 36.0 * 0.085;
/// The samples of a handle's path.
const TRAIL: usize = 96;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let cranks = [crank(), reflected(mirror(), crank())];
    let time = period() * f64::from(t / SECONDS);
    let ax = Axes::equal(
        screen.part(0.03, 0.14, 0.97, 0.9),
        Point2::xy(0.0, 0.0),
        1.45,
    );
    let (lo, hi) = (ax.at(0.5, 0.0), ax.at(0.5, 1.0));
    ax.dashed(c, &[lo, hi], 1.5, 7.0, palette::INK.faded(0.35));
    // numga's arrow scale: the velocity, `RADIUS * SPEED` long, drawn at 0.55 of the arm.
    let scale = 0.55 / SPEED;
    for (k, crank) in cranks.into_iter().enumerate() {
        // The handle's path, sampled by the same turn.
        let trail: Vec<Point> = (0..=TRAIL)
            .map(|i| turn(crank, period() * i as f64 / TRAIL as f64).handle)
            .collect();
        ax.polyline(c, &trail, 1.1, palette::GRID.faded(2.5));
        let s = turn(crank, time);
        // The load's line of action, the forque, and the twist's sense of turning about the
        // pivot: three quarters of a turn of a point near it, by the crank's own motion.
        ax.axline(
            c,
            s.handle,
            s.load.direction(),
            1.0,
            palette::ORANGE.faded(0.12),
        );
        let near_pivot = crank.pivot + Point::direction(0.0, -0.28);
        let arc: Vec<Point> = (0..=24)
            .map(|i| crank.rate.gp(-0.75 * period() * f64::from(i) / 48.0).exp() >> near_pivot)
            .collect();
        ax.polyline(c, &arc[..23], 1.6, palette::GREEN);
        ax.arrow(c, arc[22], arc[24], 1.6, 8.0, palette::GREEN);
        ax.line(c, crank.pivot, s.handle, 5.0, palette::INK.faded(0.45));
        ax.scatter(c, &[crank.pivot], Marker::Ring, 14.0, palette::INK);
        ax.scatter(c, &[s.handle], Marker::Dot, 9.0, palette::INK);
        // The arrows a little apart, across the arm, so equal directions both show.
        let radial = (s.handle - crank.pivot).gp(0.055 / RADIUS);
        for (v, light, side) in [
            (s.velocity, palette::SKY, -1.0),
            (s.push, palette::ORANGE, 1.0),
        ] {
            let from = s.handle + radial.gp(side);
            let to = from + Point::direction(v.e20(), v.e01()).gp(scale);
            ax.arrow(c, from, to, 2.2, 9.0, light);
        }
        let below = Point::xy(crank.pivot.to_euclidean()[0], -1.15);
        let reading = format!("POWER {:+.3}", s.power.s());
        let side = ["CRANK", "ITS MIRROR IMAGE"][k];
        ax.text(c, below, side, 11.0, palette::INK.faded(0.6), Align::Center);
        let under = below - Point::direction(0.0, 0.17);
        ax.text(c, under, &reading, 12.0, palette::YELLOW, Align::Center);
    }
    let legend = [
        ("VELOCITY", palette::SKY),
        ("PUSH, ITS LINE THE LOAD", palette::ORANGE),
        ("TWIST", palette::GREEN),
    ];
    ax.legend(c, &legend);
    caption(
        c,
        "MECHANICAL WORK IN A MIRROR",
        "THE LOAD HANDLE & PUSH IS A FORQUE; PAIRED WITH THE CRANK'S TWIST IT IS THE POWER, \
         THE SAME IN THE MIRROR (PGA2D)",
    );
}

fn main() {
    run(Anim::new("mechanical work", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::work::*;
    use gax::pga2d::{Forque, Point, Twist};
    use gax_numga_examples::testing::{close, near};

    /// numga's test: a unit crank about the origin pushed with strength two, and its mirror
    /// image. The load and the velocity built from the reflected handle and push are the
    /// reflected load and velocity, the power is two on both sides, and the mirror leaves the
    /// power, an extrinsic scalar, as it is.
    #[test]
    fn reflected_work_agrees_with_rebuilding_the_load() {
        let unit = Crank {
            pivot: Point::xy(0.0, 0.0),
            handle: Point::xy(1.0, 0.0),
            push: Point::direction(0.0, 2.0),
            rate: Twist::new(0.0, 0.0, 1.0),
        };
        let m = mirror();
        let image = reflected(m, unit);
        for k in 0..13 {
            let time = core::f64::consts::TAU * f64::from(k) / 12.0;
            let (a, b) = (turn(unit, time), turn(image, time));
            assert!(
                near(b.load, m >> a.load, 1e-12),
                "{:?} {:?}",
                b.load,
                a.load
            );
            assert!(near(b.velocity, m >> a.velocity, 1e-12));
            assert!(close(a.power.s(), 2.0, 1e-12) && close(b.power.s(), 2.0, 1e-12));
            assert!(near(m >> a.power, b.power, 1e-12));
        }
    }

    /// numga's scene, against its numbers: the cranks supply the same power, `1.35` (the
    /// push times the arm times the rate), at every frame; and the fifth frame's handles,
    /// velocities, pushes and loads.
    #[test]
    fn the_mirrored_cranks_turn_with_numgas_numbers() {
        let cranks = [crank(), reflected(mirror(), crank())];
        // numga: the mirror keeps the push and reverses the rate, the reflected pivot at
        // weight -1 (`e20 e01 e12` = numga's `yw wx xy`, at `-1.65, 0, -1`).
        assert!(near(cranks[1].rate, Twist::new(-1.65, 0.0, -1.0), 1e-12));
        assert!(near(cranks[1].push, cranks[0].push, 1e-12));
        for k in 0..36 {
            let time = period() * f64::from(k) / 36.0;
            for crank in cranks {
                assert!(close(turn(crank, time).power.s(), 1.35, 1e-12));
            }
        }
        let time = period() * 4.0 / 36.0;
        let [a, b] = cranks.map(|crank| turn(crank, time));
        assert!(
            near(a.handle, Point::xy(-0.960560, 0.578509), 1e-6),
            "{:?}",
            a.handle
        );
        assert!(near(b.handle, Point::xy(0.960560, 0.578509), 1e-6));
        assert!(near(
            a.velocity,
            Point::direction(-0.578509, 0.689440),
            1e-6
        ));
        assert!(near(b.velocity, Point::direction(0.578509, 0.689440), 1e-6));
        assert!(near(a.push, Point::direction(-0.964181, 1.149067), 1e-6));
        assert!(near(b.push, Point::direction(0.964181, 1.149067), 1e-6));
        // numga's loads, `x y w` = `e1 e2 e0`: the join of the same points in the same order.
        let loads = [
            Forque::new(-1.149067, -0.964181, -0.545960),
            Forque::new(-1.149067, 0.964181, 0.545960),
        ];
        assert!(near(a.load, loads[0], 1e-6), "{:?}", a.load);
        assert!(near(b.load, loads[1], 1e-6), "{:?}", b.load);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
