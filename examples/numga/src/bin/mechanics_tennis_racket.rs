//! numga's `mechanics/tennis_racket`: the tennis racket theorem in two to five dimensions.
//!
//! A rigid body with distinct principal moments spins stably about its major and minor axes but
//! tumbles about its intermediate one (the Dzhanibekov effect). The body is a box with half
//! sides 1, 2, 3, ... in p dimensions, its inertia the sum of `p & [p, ·]` over its corners;
//! one body is spun in each bivector plane with a tiny perturbation, and the Lie-group steppers
//! of the shared `mechanics_lie` module carry them, the same lines for the rotors of every
//! dimension. In three dimensions exactly one spin, the medial one, flips; in four and five
//! some medial planes wander chaotically and some stabilize.
//!
//! The animation shows three boxes in 3D spun about their major, medial and minor axes (the
//! middle one tumbles over and back), the rates of the 3D and 4D bodies as they evolve, and
//! the world-momentum drift of the three steppers in 3, 4 and 5 dimensions: RKMK4 conserves it
//! to fourth order, Verlet and RK4 to first.

use gax_numga_examples::{Anim, Axes, Canvas, Scene3, backdrop, caption, palette, plot, run};

#[path = "../shared/mechanics_lie.rs"]
mod lie;

gax::algebra! {
    algebra vga4d "Euclidean 4-space, R(4,0,0).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4];
    kind Bivector = [e12, e13, e14, e23, e24, e34];
    kind Trivector = [e234, e134, e124, e123];
    kind Pseudoscalar = [e1234];
    versor Even = [1, e12, e13, e14, e23, e24, e34, e1234];
}

gax::algebra! {
    algebra vga5d "Euclidean 5-space, R(5,0,0).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, e5];
    kind Bivector = [e12, e13, e14, e15, e23, e24, e25, e34, e35, e45];
    kind Trivector = [e123, e124, e125, e134, e135, e145, e234, e235, e245, e345];
    kind Quadvector = [e2345, e1345, e1245, e1235, e1234];
    kind Pseudoscalar = [e12345];
    versor Even = [1, e12, e13, e14, e15, e23, e24, e25, e34, e35, e45,
        e2345, e1345, e1245, e1235, e1234];
}

/// A small xorshift generator: numga's NumPy streams cannot be reproduced, and the racket's
/// perturbation only has to be tiny and generic.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Uniform in `(0, 1]`.
    pub fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Standard normal, by Box and Muller.
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    }
}

/// The racket in the dimension of the kinds in scope (numga instantiates its `core` per
/// algebra; here the same text expands per algebra, after the Lie steppers).
macro_rules! racket {
    () => {
        /// Batched bodies, one per bivector plane, sharing one inertia.
        pub struct Body {
            pub motor: Vec<M>,
            pub rate: Vec<R>,
            pub inertia: Inertia,
            pub inertia_inv: InertiaInv,
        }

        /// The dimension.
        pub fn dimension() -> usize {
            Vector::<(), f64>::zero().c.len()
        }

        /// The corners of the box with half sides 1, 2, ..., p, as vectors.
        pub fn corners() -> Vec<Vector<(), f64>> {
            let p = dimension();
            (0..1usize << p)
                .map(|bits| {
                    Vector::from_coeffs(core::array::from_fn(|i| {
                        let sign = if bits >> i & 1 == 1 { 1.0 } else { -1.0 };
                        sign * (i as f64 + 1.0)
                    }))
                })
                .collect()
        }

        /// One body spinning in each independent bivector plane, with a small perturbation.
        pub fn racket(seed: u64) -> Body {
            // The corners as mass points: the duals of the vectors.
            let points: Vec<P> = corners().into_iter().map(|v| v.dual()).collect();
            let (inertia, inertia_inv) = inertia_from_points(&points);
            let planes = R::zero().c.len();
            let mut rng = crate::Rng::new(seed);
            let rate: Vec<R> = (0..planes)
                .map(|i| {
                    R::from_coeffs(core::array::from_fn(|j| {
                        f64::from(u8::from(i == j)) + 1e-5 * rng.normal()
                    }))
                })
                .collect();
            Body {
                motor: vec![R::zero().exp(); planes],
                rate,
                inertia,
                inertia_inv,
            }
        }

        /// Torque-free motion: motors and rates over time (`[step][body]`), starting with the
        /// initial state.
        pub fn simulate(
            body: &Body,
            step: Step,
            dt: f64,
            steps: usize,
        ) -> (Vec<Vec<M>>, Vec<Vec<R>>) {
            let mut motors = vec![body.motor.clone()];
            let mut rates = vec![body.rate.clone()];
            for _ in 0..steps {
                let (m, r): (Vec<M>, Vec<R>) = motors[motors.len() - 1]
                    .iter()
                    .zip(&rates[rates.len() - 1])
                    .map(|(m, r)| step(*m, *r, body.inertia, body.inertia_inv, dt, &free))
                    .unzip();
                motors.push(m);
                rates.push(r);
            }
            (motors, rates)
        }

        /// The relative drift of the world-frame angular momentum, per step and body. The
        /// squared magnitude of a momentum is the scalar part of `p ~p`, in any dimension.
        pub fn momentum_drift(
            motors: &[Vec<M>],
            rates: &[Vec<R>],
            inertia: Inertia,
        ) -> Vec<Vec<f64>> {
            let world = |k: usize, b: usize| motors[k][b] >> inertia.of(rates[k][b]);
            (0..motors.len())
                .map(|k| {
                    (0..motors[k].len())
                        .map(|b| {
                            let start = world(0, b);
                            ((world(k, b) - start).norm_squared() / start.norm_squared()).sqrt()
                        })
                        .collect()
                })
                .collect()
        }

        /// A stepper by numga's name.
        pub fn stepper(name: &str) -> Step {
            match name {
                "verlet" => explicit_verlet,
                "rk4" => explicit_rk4,
                _ => explicit_rkmk4,
            }
        }

        /// The worst world-momentum drift over all bodies and steps of a run of `runtime`.
        pub fn worst_drift(name: &str, dt: f64, runtime: f64) -> f64 {
            let body = racket(42);
            let (m, r) = simulate(&body, stepper(name), dt, (runtime / dt).round() as usize);
            momentum_drift(&m, &r, body.inertia)
                .iter()
                .flatten()
                .fold(0.0, |a: f64, &b| a.max(b))
        }
    };
}

/// The plane: one spin plane, nothing to tumble.
pub mod d2 {
    use gax::vga2d::{Bivector as Rate, Rotor as Motor, Scalar as Forque, Vector, Vector as Point};
    crate::lie::integrators!(commutative);
    racket!();
}

/// Space: three spin planes, the classical theorem.
pub mod d3 {
    use gax::vga3d::{
        Bivector as Rate, Bivector as Point, Rotor as Motor, Vector, Vector as Forque,
    };
    crate::lie::integrators!();
    racket!();
}

/// Four dimensions: six spin planes.
pub mod d4 {
    use crate::vga4d::{
        Bivector as Rate, Bivector as Forque, Even as Motor, Trivector as Point, Vector,
    };
    crate::lie::integrators!();
    racket!();
}

/// Five dimensions: ten spin planes.
pub mod d5 {
    use crate::vga5d::{
        Bivector as Rate, Even as Motor, Quadvector as Point, Trivector as Forque, Vector,
    };
    crate::lie::integrators!();
    racket!();
}

/// The simulated time the loop shows, and the loop's length in seconds.
const SPAN: f64 = 105.0;
const LOOP: f32 = 21.0;
/// The display step (finer than numga's 0.25, for smooth motion).
const SHOW_DT: f64 = 0.05;

/// Everything the frames show, simulated once.
struct Scenes {
    /// The three 3D boxes spun in the planes e23, e31 and e12 (`[step][body]`).
    motors3: Vec<Vec<d3::M>>,
    rates3: Vec<Vec<d3::R>>,
    /// The six 4D bodies' rates.
    rates4: Vec<Vec<d4::R>>,
    /// numga's integrator comparison: the worst world-momentum drift over the bodies per step,
    /// per dimension and stepper, at dt = 0.25 over 100.
    drift: Vec<(usize, &'static str, Vec<f64>)>,
}

/// The comparison's step and run time (numga's `integrator_comparison((3, 4, 5), 0.25, 100.0)`).
const COMPARE_DT: f64 = 0.25;
const COMPARE_RUN: f64 = 100.0;

fn scenes() -> &'static Scenes {
    static SCENES: std::sync::OnceLock<Scenes> = std::sync::OnceLock::new();
    SCENES.get_or_init(|| {
        let steps = (SPAN / SHOW_DT).round() as usize;
        let body3 = d3::racket(42);
        let (motors3, rates3) = d3::simulate(&body3, d3::explicit_rkmk4, SHOW_DT, steps);
        let body4 = d4::racket(42);
        let (_, rates4) = d4::simulate(&body4, d4::explicit_rkmk4, SHOW_DT, steps);
        let worst = |d: Vec<Vec<f64>>| -> Vec<f64> {
            d.iter()
                .map(|b| b.iter().fold(0.0, |a: f64, &x| a.max(x)))
                .collect()
        };
        let n = (COMPARE_RUN / COMPARE_DT).round() as usize;
        let mut drift = Vec::new();
        for name in ["verlet", "rk4", "rkmk4"] {
            let b = d3::racket(42);
            let (m, r) = d3::simulate(&b, d3::stepper(name), COMPARE_DT, n);
            drift.push((3, name, worst(d3::momentum_drift(&m, &r, b.inertia))));
            let b = d4::racket(42);
            let (m, r) = d4::simulate(&b, d4::stepper(name), COMPARE_DT, n);
            drift.push((4, name, worst(d4::momentum_drift(&m, &r, b.inertia))));
            let b = d5::racket(42);
            let (m, r) = d5::simulate(&b, d5::stepper(name), COMPARE_DT, n);
            drift.push((5, name, worst(d5::momentum_drift(&m, &r, b.inertia))));
        }
        Scenes {
            motors3,
            rates3,
            rates4,
            drift,
        }
    })
}

fn v3(v: gax::vga3d::Vector<(), f64>) -> [f32; 3] {
    v.c.map(|x| x as f32)
}

/// One box of the 3D racket, turned by its rotor, drawn with its spin axis about pixel `centre`
/// at `scale` pixels per unit.
fn draw_box(c: &mut Canvas, rotor: d3::M, axis: usize, centre: [f32; 2], scale: f32) {
    use gax_numga_examples::{Camera, Lens};
    // A parallel camera whose canvas centre falls on `centre`.
    let half = centre[1] / scale;
    let cam = Camera::looking(
        (2.0 * centre[0]) as usize,
        (2.0 * centre[1]) as usize,
        [10.0, -16.0, 9.0],
        [0.0, 0.0, 0.0],
        Lens::Parallel(half),
    );
    let mut s = Scene3::new(cam);
    let corners: Vec<[f32; 3]> = d3::corners().into_iter().map(|v| v3(rotor >> v)).collect();
    let face_colours = [palette::orange(), palette::sky(), palette::green()];
    for a in 0..3 {
        let (b, d) = ((a + 1) % 3, (a + 2) % 3);
        for side in [0, 1] {
            let at = |u: usize, v: usize| corners[side << a | u << b | v << d];
            let (p, q, r, t) = (at(0, 0), at(1, 0), at(1, 1), at(0, 1));
            let col = s.lit(p, q, r, face_colours[a]);
            s.quad(p, q, r, t, col, 0.92);
            for (x, y) in [(p, q), (q, r), (r, t), (t, p)] {
                s.seg(x, y, 1.0, palette::bottom(), 0.5);
            }
        }
    }
    // The axis the body was spun about, carried with it.
    let mut e = [0.0; 3];
    e[axis] = 4.6;
    let tip = v3(rotor >> gax::vga3d::Vector::from_coeffs(e.map(f64::from)));
    s.seg(tip.map(|x| -x), tip, 2.5, palette::yellow(), 1.0);
    s.dot(tip, gax_numga_examples::Marker::Dot, 7.0, palette::yellow());
    s.draw(c);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let sc = scenes();
    let (w, h) = (c.width as f32, c.height as f32);
    let u = h / 540.0;
    let time = f64::from(t.rem_euclid(LOOP) / LOOP) * SPAN;
    let k = ((time / SHOW_DT) as usize).min(sc.rates3.len() - 1);
    caption(
        c,
        "TENNIS RACKET: THE MEDIAL AXIS TUMBLES",
        "BOXES OF HALF SIDES 1, 2, 3, ... IN 2 TO 5 DIMENSIONS",
    );
    // The three 3D boxes.
    let labels = [
        "MAJOR AXIS: STABLE",
        "MEDIAL AXIS: TUMBLES",
        "MINOR AXIS: STABLE",
    ];
    let scale = 15.5 * u;
    for (i, label) in labels.iter().enumerate() {
        let centre = [w * (0.09 + 0.16 * i as f32), h * 0.33];
        draw_box(c, sc.motors3[k][i], i, centre, scale);
        c.text(
            label,
            centre[0],
            h * 0.53,
            10.0 * u,
            palette::ink(),
            gax_numga_examples::Align::Center,
        );
    }
    let span = SPAN as f32;
    let growing = |rates: &dyn Fn(usize) -> f32| -> Vec<[f32; 2]> {
        (0..=k)
            .map(|j| [(j as f64 * SHOW_DT) as f32, rates(j)])
            .collect()
    };
    // The 3D rates: each body's spin in its own plane.
    let ax = Axes::new(
        plot::inset(
            [0.0, h * 0.55, w * 0.5, h],
            46.0 * u,
            34.0 * u,
            16.0 * u,
            34.0 * u,
        ),
        [0.0, span],
        [-1.25, 1.25],
    );
    ax.frame(c, "3D: EACH BODY'S RATE IN ITS OWN PLANE", "TIME", "");
    for i in 0..3 {
        let pts = growing(&|j| sc.rates3[j][i].c[i] as f32);
        ax.polyline(c, &pts, 1.6, palette::series(i), 1.0);
    }
    ax.legend(
        c,
        &[
            ("MAJOR", palette::series(0)),
            ("MEDIAL", palette::series(1)),
            ("MINOR", palette::series(2)),
        ],
    );
    // The 4D rates: six planes, some medial spins wander.
    let ax = Axes::new(
        plot::inset(
            [w * 0.5, h * 0.12, w, h * 0.52],
            46.0 * u,
            34.0 * u,
            16.0 * u,
            30.0 * u,
        ),
        [0.0, span],
        [-1.25, 1.25],
    );
    ax.frame(c, "4D: SIX SPIN PLANES, RKMK4", "", "");
    for i in 0..6 {
        let pts = growing(&|j| sc.rates4[j][i].c[i] as f32);
        ax.polyline(c, &pts, 1.4, palette::series(i), 1.0);
    }
    // The steppers' world-momentum drift, per dimension.
    let shown = ((time.min(COMPARE_RUN) / COMPARE_DT) as usize).max(1);
    let colours = [palette::orange(), palette::sky(), palette::green()];
    for (p, dim) in [3, 4, 5].into_iter().enumerate() {
        let x0 = w * 0.5 + p as f32 * w * 0.5 / 3.0;
        let rect = plot::inset(
            [x0, h * 0.55, x0 + w * 0.5 / 3.0, h],
            if p == 0 { 40.0 } else { 14.0 } * u,
            34.0 * u,
            8.0 * u,
            34.0 * u,
        );
        let ax = Axes::new(rect, [0.0, COMPARE_RUN as f32], [1e-12, 1.0]).log_y();
        let title = format!("{dim}D DRIFT");
        ax.frame(c, &title, "TIME", "");
        for (s, (_, name, curve)) in sc.drift.iter().filter(|d| d.0 == dim).enumerate() {
            let pts: Vec<[f32; 2]> = (1..=shown.min(curve.len() - 1))
                .map(|j| [(j as f64 * COMPARE_DT) as f32, curve[j].max(1e-30) as f32])
                .collect();
            if *name == "rk4" {
                ax.dashed(c, &pts, 1.6, 5.0, colours[s], 1.0);
            } else {
                ax.polyline(c, &pts, 1.6, colours[s], 1.0);
            }
        }
        if p == 2 {
            ax.legend(
                c,
                &[
                    ("VERLET", colours[0]),
                    ("RK4", colours[1]),
                    ("RKMK4", colours[2]),
                ],
            );
        }
    }
    // The time cursor on the rate plots.
    for rect in [
        plot::inset(
            [0.0, h * 0.55, w * 0.5, h],
            46.0 * u,
            34.0 * u,
            16.0 * u,
            34.0 * u,
        ),
        plot::inset(
            [w * 0.5, h * 0.12, w, h * 0.52],
            46.0 * u,
            34.0 * u,
            16.0 * u,
            30.0 * u,
        ),
    ] {
        let ax = Axes::new(rect, [0.0, span], [-1.25, 1.25]);
        ax.line(
            c,
            [time as f32, -1.25],
            [time as f32, 1.25],
            1.0,
            palette::grid(),
            1.0,
        );
    }
}

fn main() {
    run(Anim::new("tennis racket", LOOP).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sandwich by `h.exp()` on bivectors equals the exponential of the commutator with
    /// `h`, on the ten-dimensional bivectors of five dimensions. The exponential of the map is
    /// its Taylor series, composed with `of`.
    #[test]
    fn adjoint_is_the_open_commutator_in_five_dimensions() {
        use vga5d::Bivector;
        let mut rng = Rng::new(0);
        let h = Bivector::<(), f64>::from_coeffs(core::array::from_fn(|_| 0.3 * rng.normal()));
        let slot = Bivector::<(), f64>::slot();
        // gax's commutator, like numga's, is half the bracket.
        let ad = h.commutator(slot) * 2.0;
        let (mut term, mut series) = (slot, slot);
        for k in 1..40 {
            term = ad.of(term) * (1.0 / f64::from(k));
            series += term;
        }
        let sandwich = h.exp() >> slot;
        for (a, b) in series.c.iter().flatten().zip(sandwich.c.iter().flatten()) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    /// All steppers run the same RK4 on the autonomous body-frame rate, so energies agree.
    #[test]
    fn energy_is_blind_to_the_motor_step() {
        let body = d4::racket(42);
        let histories: Vec<Vec<f64>> = ["verlet", "rk4", "rkmk4"]
            .iter()
            .map(|name| {
                let (_, rates) = d4::simulate(&body, d4::stepper(name), 0.25, 40);
                rates
                    .iter()
                    .flatten()
                    .map(|r| d4::kinetic_energy(*r, body.inertia))
                    .collect()
            })
            .collect();
        for h in &histories[1..] {
            for (a, b) in h.iter().zip(&histories[0]) {
                assert!((a - b).abs() <= 1e-11 * b.abs(), "{a} vs {b}");
            }
        }
    }

    /// Halving dt cuts RKMK4's momentum drift about sixteenfold; Verlet and RK4 only halve it.
    #[test]
    fn rkmk4_conserves_world_momentum_to_fourth_order() {
        let checks: [fn(&str, f64, f64) -> f64; 2] = [d3::worst_drift, d4::worst_drift];
        for drift in checks {
            let coarse = drift("rkmk4", 0.25, 20.0);
            let fine = drift("rkmk4", 0.125, 20.0);
            assert!(coarse < 1e-4, "{coarse}");
            assert!(
                10.0 < coarse / fine && coarse / fine < 24.0,
                "{}",
                coarse / fine
            );
            for first_order in ["verlet", "rk4"] {
                let ratio = drift(first_order, 0.25, 20.0) / drift(first_order, 0.125, 20.0);
                assert!(1.7 < ratio && ratio < 2.4, "{first_order}: {ratio}");
            }
        }
    }

    /// The same stepper integrates Spin(5) rotors with fourth-order momentum conservation.
    #[test]
    fn rkmk4_in_five_dimensions() {
        let coarse = d5::worst_drift("rkmk4", 0.25, 20.0);
        let fine = d5::worst_drift("rkmk4", 0.125, 20.0);
        assert!(coarse < 0.01, "{coarse}");
        assert!(
            10.0 < coarse / fine && coarse / fine < 24.0,
            "{}",
            coarse / fine
        );
        assert!(d5::worst_drift("verlet", 0.25, 20.0) > 1e-2);
    }

    /// In 3D the spin about the medial axis flips sign; the major and minor axes stay put.
    #[test]
    fn intermediate_axis_tumbles_and_others_do_not() {
        let body = d3::racket(42);
        let (_, rates) = d3::simulate(&body, d3::explicit_rkmk4, 0.25, 800);
        let flips = (0..3)
            .filter(|&i| rates.iter().any(|r| r[i].c[i] * rates[0][i].c[i] < 0.0))
            .count();
        assert_eq!(flips, 1);
    }

    /// In the plane rotations commute: the spin never changes, and every stepper conserves the
    /// momentum exactly.
    #[test]
    fn the_plane_has_nothing_to_tumble() {
        let body = d2::racket(42);
        let (motors, rates) = d2::simulate(&body, d2::explicit_rkmk4, 0.25, 40);
        assert!((rates[40][0].c[0] - rates[0][0].c[0]).abs() < 1e-15);
        let drift = d2::momentum_drift(&motors, &rates, body.inertia);
        assert!(drift.iter().flatten().all(|d| *d < 1e-14));
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
