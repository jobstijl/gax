//! numga's `mechanics/tennis_racket`: the tennis racket theorem in two to five dimensions.
//!
//! A rigid body with distinct principal moments spins stably about its major and minor axes but
//! tumbles about its intermediate one (the Dzhanibekov effect). The body is a box with half
//! sides 1, 2, 3, ... in p dimensions, its inertia the sum of `p & [p, ·]` over its corners;
//! one body is spun in each bivector plane with a tiny perturbation, and the Lie-group steppers
//! of the shared `mechanics_lie` module carry them, the same generic functions over
//! `gax::motions::Motions` for the rotors of every dimension (VGA4D and VGA5D are declared here
//! and join in with `gax::motions!`). In three dimensions exactly one spin, the medial one,
//! flips; in four and five some medial planes wander chaotically and some stabilize.
//!
//! The animation shows three boxes in 3D spun about their major, medial and minor axes (the
//! middle one tumbles over and back), the rates of the 3D and 4D bodies as they evolve, and
//! the world-momentum drift of the three steppers in 3, 4 and 5 dimensions: RKMK4 conserves it
//! to fourth order, Verlet and RK4 to first.

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, Scene3,
    backdrop, caption, palette, run,
};

/// A point of a chart: (time, value).
type Chart = gax::pga2d::Point<(), f64>;

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

use gax::{Extensor, Kind, Of};
use gax_numga_examples::rng::{Draw, rng};
use lie::{Inertia, InertiaInv, Lie, M, R, Step};

/// Batched bodies, one per bivector plane, sharing one inertia.
pub struct Body<G: Lie> {
    pub motor: Vec<M<G>>,
    pub rate: Vec<R<G>>,
    pub inertia: Inertia<G>,
    pub inertia_inv: InertiaInv<G>,
}

/// The corners of the box with half sides 1, 2, ..., p.
pub fn corners<G: Lie>() -> Vec<G::Coords> {
    (0..1usize << G::DIM)
        .map(|bits| {
            G::coords_from_fn(|i| {
                let sign = if bits >> i & 1 == 1 { 1.0 } else { -1.0 };
                sign * (i as f64 + 1.0)
            })
        })
        .collect()
}

/// One body spinning in each independent bivector plane, with a small perturbation.
pub fn racket<G: Lie>(seed: u64) -> Body<G> {
    let points: Vec<_> = corners::<G>().into_iter().map(G::point).collect();
    let (inertia, inertia_inv) = G::inertia_from_points(&points);
    type Planes<G> = <R<G> as Extensor>::Kind;
    let planes = <Planes<G> as Kind>::N;
    let mut rng = rng(seed);
    let rate: Vec<R<G>> = (0..planes)
        .map(|i| {
            R::<G>::from_coeffs(<Planes<G> as Kind>::arr_from_fn(|j| {
                f64::from(u8::from(i == j)) + 1e-5 * rng.normal()
            }))
        })
        .collect();
    Body {
        motor: vec![G::identity(); planes],
        rate,
        inertia,
        inertia_inv,
    }
}

/// Motors and rates over time, `[step][body]`.
pub type History<G> = (Vec<Vec<M<G>>>, Vec<Vec<R<G>>>);

/// Torque-free motion: motors and rates over time (`[step][body]`), starting with the initial
/// state.
pub fn simulate<G: Lie>(body: &Body<G>, step: Step<G>, dt: f64, steps: usize) -> History<G> {
    let mut motors = vec![body.motor.clone()];
    let mut rates = vec![body.rate.clone()];
    for _ in 0..steps {
        let (m, r): (Vec<M<G>>, Vec<R<G>>) = motors[motors.len() - 1]
            .iter()
            .zip(&rates[rates.len() - 1])
            .map(|(m, r)| step(*m, *r, body.inertia, body.inertia_inv, dt, &G::free))
            .unzip();
        motors.push(m);
        rates.push(r);
    }
    (motors, rates)
}

/// The relative drift of the world-frame angular momentum, per step and body: the norm of the
/// change over the norm of the start (in a vector algebra, the momentum bivector's norm is its
/// magnitude).
pub fn momentum_drift<G: Lie>(
    motors: &[Vec<M<G>>],
    rates: &[Vec<R<G>>],
    inertia: Inertia<G>,
) -> Vec<Vec<f64>>
where
    lie::F<G>: gax::Norm,
{
    use gax::Norm;
    let world = |k: usize, b: usize| motors[k][b] >> inertia.of(rates[k][b]);
    (0..motors.len())
        .map(|k| {
            (0..motors[k].len())
                .map(|b| {
                    let start = world(0, b);
                    (world(k, b) - start).norm() / start.norm()
                })
                .collect()
        })
        .collect()
}

/// The steppers by numga's names.
pub const STEPPERS: [&str; 3] = ["verlet", "rk4", "rkmk4"];

/// A stepper by numga's name.
pub fn stepper<G: Lie>(name: &str) -> Step<G> {
    match name {
        "verlet" => G::explicit_verlet,
        "rk4" => G::explicit_rk4,
        _ => G::explicit_rkmk4,
    }
}

/// The worst world-momentum drift over all bodies, per step of a run of `runtime`.
pub fn drift_per_step<G: Lie>(name: &str, dt: f64, runtime: f64) -> Vec<f64>
where
    lie::F<G>: gax::Norm,
{
    let body = racket::<G>(42);
    let steps = (runtime / dt).round() as usize;
    let (m, r) = simulate(&body, stepper::<G>(name), dt, steps);
    momentum_drift::<G>(&m, &r, body.inertia)
        .iter()
        .map(|bodies| bodies.iter().fold(0.0, |a: f64, &b| a.max(b)))
        .collect()
}

/// The worst world-momentum drift over all bodies and steps of a run of `runtime`.
pub fn worst_drift<G: Lie>(name: &str, dt: f64, runtime: f64) -> f64
where
    lie::F<G>: gax::Norm,
{
    drift_per_step::<G>(name, dt, runtime)
        .into_iter()
        .fold(0.0, |a: f64, b| a.max(b))
}

/// The plane: one spin plane, nothing to tumble.
pub use gax::motions::Vga2d as D2;
/// Space: three spin planes, the classical theorem.
pub use gax::motions::Vga3d as D3;

gax::motions! {
    /// Four dimensions: six spin planes.
    pub struct D4 in crate::vga4d {
        Motor = Even, Twist = Bivector, Forque = Bivector, Point = Trivector,
        Coords = [T; 4],
        point(c) = Vector::<(), T>::from_coeffs(c).dual(),
        coords(p) = p.undual().c,
    }
}

gax::motions! {
    /// Five dimensions: ten spin planes.
    pub struct D5 in crate::vga5d {
        Motor = Even, Twist = Bivector, Forque = Trivector, Point = Quadvector,
        Coords = [T; 5],
        point(c) = Vector::<(), T>::from_coeffs(c).dual(),
        coords(p) = p.undual().c,
    }
}

/// The simulated time the loop shows, and the loop's length in seconds.
const SPAN: f64 = 105.0;
const LOOP: f32 = 21.0;
/// The display step (finer than numga's 0.25, for smooth motion).
const SHOW_DT: f64 = 0.05;

/// Everything the frames show, simulated once.
struct Scenes {
    /// The three 3D boxes spun in the planes e23, e31 and e12 (`[step][body]`).
    motors3: Vec<Vec<M<D3>>>,
    rates3: Vec<Vec<R<D3>>>,
    /// The six 4D bodies' rates.
    rates4: Vec<Vec<R<D4>>>,
    /// numga's integrator comparison: the worst world-momentum drift over the bodies per step,
    /// in 3, 4 and 5 dimensions, per stepper, at dt = 0.25 over 100.
    drift: [[Vec<f64>; 3]; 3],
}

/// The comparison's step and run time (numga's `integrator_comparison((3, 4, 5), 0.25, 100.0)`).
const COMPARE_DT: f64 = 0.25;
const COMPARE_RUN: f64 = 100.0;

fn scenes() -> &'static Scenes {
    static SCENES: std::sync::OnceLock<Scenes> = std::sync::OnceLock::new();
    SCENES.get_or_init(|| {
        let steps = (SPAN / SHOW_DT).round() as usize;
        let body3 = racket::<D3>(42);
        let (motors3, rates3) = simulate::<D3>(&body3, D3::explicit_rkmk4, SHOW_DT, steps);
        let body4 = racket::<D4>(42);
        let (_, rates4) = simulate::<D4>(&body4, D4::explicit_rkmk4, SHOW_DT, steps);
        let drift = [
            STEPPERS.map(|name| drift_per_step::<D3>(name, COMPARE_DT, COMPARE_RUN)),
            STEPPERS.map(|name| drift_per_step::<D4>(name, COMPARE_DT, COMPARE_RUN)),
            STEPPERS.map(|name| drift_per_step::<D5>(name, COMPARE_DT, COMPARE_RUN)),
        ];
        Scenes {
            motors3,
            rates3,
            rates4,
            drift,
        }
    })
}

/// The point a vector of the body reaches from its centre, which stays at the origin.
fn placed(v: gax::vga3d::Vector<(), f64>) -> gax::pga3d::Point<(), f64> {
    gax::pga3d::Point::xyz(0.0, 0.0, 0.0) + gax::pga3d::Point::direction(v.e1(), v.e2(), v.e3())
}

/// One box of the 3D racket, turned by its rotor, drawn with its spin axis about pixel `centre`
/// at `scale` pixels per unit.
fn draw_box(c: &mut Canvas, rotor: M<D3>, axis: usize, centre: Point2, scale: f32) {
    use gax::vga3d::Vector;
    // A parallel camera drawing into the square five units either way of `centre`.
    let reach = Point2::direction(5.0, 5.0).gp(scale);
    let view = Rect {
        lo: centre - reach,
        hi: centre + reach,
    };
    let cam = Camera::looking(
        view.width() as usize,
        view.height() as usize,
        Point3::xyz(10.0, -16.0, 9.0),
        ORIGIN3,
        Lens::Parallel(5.0),
    )
    .viewport(view);
    let mut s = Scene3::new(cam);
    let corners: Vec<_> = corners::<D3>()
        .into_iter()
        .map(|c| placed(rotor >> Vector::from_coeffs(c)))
        .collect();
    let face_colours = [palette::orange(), palette::sky(), palette::green()];
    for a in 0..3 {
        let (b, d) = ((a + 1) % 3, (a + 2) % 3);
        for side in [0, 1] {
            let at = |u: usize, v: usize| corners[side << a | u << b | v << d];
            let (p, q, r, t) = (at(0, 0), at(1, 0), at(1, 1), at(0, 1));
            // The faces cover what is behind them: a dimmer light than the glowing strokes.
            let col = s.lit(p, q, r, face_colours[a].faded(0.5));
            s.quad(p, q, r, t, col, 0.92);
            for (x, y) in [(p, q), (q, r), (r, t), (t, p)] {
                s.seg(x, y, 1.0, palette::bottom().faded(0.5));
            }
        }
    }
    // The axis the body was spun about, carried with it.
    let along = core::array::from_fn(|i| if i == axis { 4.6 } else { 0.0 });
    let tip = rotor >> Vector::<(), f64>::from_coeffs(along);
    s.seg(placed(-tip), placed(tip), 2.5, palette::yellow());
    s.dot(placed(tip), Marker::Dot, 7.0, palette::yellow());
    s.draw(c);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let sc = scenes();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let u = c.unit();
    let time = f64::from(t.rem_euclid(LOOP) / LOOP) * SPAN;
    let k = ((time / SHOW_DT) as usize).min(sc.rates3.len() - 1);
    caption(
        c,
        "TENNIS RACKET: THE MEDIAL AXIS TUMBLES",
        "BOXES OF HALF SIDES 1, 2, 3, ... IN 2 TO 5 DIMENSIONS",
    );
    // The three 3D boxes.
    let labels = [
        ("MAJOR AXIS", "STABLE"),
        ("MEDIAL AXIS", "TUMBLES"),
        ("MINOR AXIS", "STABLE"),
    ];
    let scale = 15.5 * u;
    let down = Point2::direction(0.0, 13.0 * u);
    for (i, (axis, fate)) in labels.iter().enumerate() {
        let across = w * (0.09 + 0.16 * i as f32);
        draw_box(c, sc.motors3[k][i], i, Point2::xy(across, h * 0.33), scale);
        let label = Point2::xy(across, h * 0.51);
        for (line, text) in [axis, fate].into_iter().enumerate() {
            let at = label + down.gp(line as f32);
            c.text(text, at, 10.0 * u, palette::ink(), Align::Center);
        }
    }
    // The rates, each body's spin in its own plane: the three 3D bodies, and the six 4D ones,
    // some of whose medial spins wander.
    let rates3 = Axes::new(
        Rect::new(0.0, h * 0.55, w * 0.5, h).inset(46.0 * u, 34.0 * u, 16.0 * u, 34.0 * u),
        [0.0, SPAN as f32],
        [-1.25, 1.25],
    );
    rates3.frame(c, "3D: EACH BODY'S RATE IN ITS OWN PLANE", "TIME", "");
    rate_curves(c, &rates3, 3, k, 1.6, |j, i| sc.rates3[j][i].c[i]);
    rates3.legend(
        c,
        &[
            ("MAJOR", palette::series(0)),
            ("MEDIAL", palette::series(1)),
            ("MINOR", palette::series(2)),
        ],
    );
    let rates4 = Axes::new(
        Rect::new(w * 0.5, h * 0.12, w, h * 0.52).inset(46.0 * u, 34.0 * u, 16.0 * u, 30.0 * u),
        [0.0, SPAN as f32],
        [-1.25, 1.25],
    );
    rates4.frame(c, "4D: SIX SPIN PLANES, RKMK4", "", "");
    rate_curves(c, &rates4, 6, k, 1.4, |j, i| sc.rates4[j][i].c[i]);
    // The steppers' world-momentum drift, per dimension.
    let shown = ((time.min(COMPARE_RUN) / COMPARE_DT) as usize).max(1);
    let colours = [palette::orange(), palette::sky(), palette::green()];
    let drifts = Rect::new(w * 0.5, h * 0.55, w, h);
    for (p, curves) in sc.drift.iter().enumerate() {
        // Each plot keeps room on its left for its decade labels, clear of the time labels of
        // the plot before it.
        let rect = drifts
            .column(p, 3)
            .inset(40.0 * u, 34.0 * u, 12.0 * u, 34.0 * u);
        let ax = Axes::new(rect, [0.0, COMPARE_RUN as f32], [1e-12, 1.0]).log_y();
        ax.frame(c, &format!("{}D DRIFT", p + 3), "TIME", "");
        for (s, (name, curve)) in STEPPERS.iter().zip(curves).enumerate() {
            let pts: Vec<Chart> = (1..=shown.min(curve.len() - 1))
                .map(|j| Chart::xy(j as f64 * COMPARE_DT, curve[j].max(1e-30)))
                .collect();
            if *name == "rk4" {
                ax.dashed(c, &pts, 1.6, 5.0, colours[s]);
            } else {
                ax.polyline(c, &pts, 1.6, colours[s]);
            }
        }
        if p == 2 {
            let entries: Vec<(&str, Light)> = STEPPERS
                .iter()
                .zip(colours)
                .map(|(name, colour)| (*name, colour))
                .collect();
            ax.legend(c, &entries);
        }
    }
    // The time cursor on the rate plots.
    for ax in [rates3, rates4] {
        let at = |rate: f64| Chart::xy(time, rate);
        ax.line(c, at(-1.25), at(1.25), 1.0, palette::grid());
    }
}

/// The rates of `bodies` bodies up to step `k`, `rate(step, body)`, one curve each.
fn rate_curves(
    c: &mut Canvas,
    ax: &Axes,
    bodies: usize,
    k: usize,
    width: f32,
    rate: impl Fn(usize, usize) -> f64,
) {
    for i in 0..bodies {
        let pts: Vec<Chart> = (0..=k)
            .map(|j| Chart::xy(j as f64 * SHOW_DT, rate(j, i)))
            .collect();
        ax.polyline(c, &pts, width, palette::series(i));
    }
}

fn main() {
    run(Anim::new("tennis racket", LOOP).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gax::ApproxEq;

    /// The sandwich by `h.exp()` on bivectors equals the exponential of the commutator with
    /// `h`, on the ten-dimensional bivectors of five dimensions. The exponential of the map is
    /// its Taylor series, composed with `of`.
    #[test]
    fn adjoint_is_the_open_commutator_in_five_dimensions() {
        use vga5d::Bivector;
        let mut rng = rng(0);
        let h = Bivector::<(), f64>::from_coeffs(core::array::from_fn(|_| 0.3 * rng.normal()));
        let slot = Bivector::<(), f64>::slot();
        // gax's commutator, like numga's, is half the bracket.
        let ad = h.commutator(slot) * 2.0;
        let (mut term, mut series) = (slot, slot);
        for k in 1..40 {
            term = ad.of(term) / f64::from(k);
            series += term;
        }
        let sandwich = h.exp() >> slot;
        let error = series.max_abs_diff(&sandwich);
        assert!(error < 1e-10, "{error}");
    }

    /// All steppers run the same RK4 on the autonomous body-frame rate, so energies agree.
    #[test]
    fn energy_is_blind_to_the_motor_step() {
        let body = racket::<D4>(42);
        let histories: Vec<Vec<f64>> = STEPPERS
            .iter()
            .map(|name| {
                let (_, rates) = simulate::<D4>(&body, stepper::<D4>(name), 0.25, 40);
                rates
                    .iter()
                    .flatten()
                    .map(|r| D4::kinetic_energy(*r, body.inertia))
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
        let checks: [fn(&str, f64, f64) -> f64; 2] = [worst_drift::<D3>, worst_drift::<D4>];
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
        let coarse = worst_drift::<D5>("rkmk4", 0.25, 20.0);
        let fine = worst_drift::<D5>("rkmk4", 0.125, 20.0);
        assert!(coarse < 0.01, "{coarse}");
        assert!(
            10.0 < coarse / fine && coarse / fine < 24.0,
            "{}",
            coarse / fine
        );
        assert!(worst_drift::<D5>("verlet", 0.25, 20.0) > 1e-2);
    }

    /// In 3D the spin about the medial axis flips sign; the major and minor axes stay put.
    #[test]
    fn intermediate_axis_tumbles_and_others_do_not() {
        let body = racket::<D3>(42);
        let (_, rates) = simulate::<D3>(&body, D3::explicit_rkmk4, 0.25, 800);
        let flips = (0..3)
            .filter(|&i| rates.iter().any(|r| r[i].c[i] * rates[0][i].c[i] < 0.0))
            .count();
        assert_eq!(flips, 1);
    }

    /// In the plane rotations commute: the spin never changes, and every stepper conserves the
    /// momentum exactly.
    #[test]
    fn the_plane_has_nothing_to_tumble() {
        let body = racket::<D2>(42);
        let (motors, rates) = simulate::<D2>(&body, D2::explicit_rkmk4, 0.25, 40);
        assert!((rates[40][0].c[0] - rates[0][0].c[0]).abs() < 1e-15);
        let drift = momentum_drift::<D2>(&motors, &rates, body.inertia);
        assert!(drift.iter().flatten().all(|d| *d < 1e-14));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
