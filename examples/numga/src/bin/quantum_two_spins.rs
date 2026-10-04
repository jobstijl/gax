//! numga's `quantum/two_spins`: two spins entangled by their exchange interaction, in the
//! algebra of two copies of space, R(6,0). One spin lives in the directions `x y z` (`e1 e2 e3`),
//! the other in `X Y Z` (`e4 e5 e6`). The state of one spin is an even multivector of its own
//! space, as in the Pauli algebra; the state of the pair is a product of the two, taken in the
//! ideal of the correlator `C = (1 - xy XY) / 2`. In that ideal, multiplying on the right by either
//! spin's `xy` plane is the same: `C xy` is the pair's imaginary unit.
//!
//! The exchange couples each plane of one spin to the same plane of the other,
//! `K = yz YZ + zx ZX + xy XY`, with `K² = 3 + 2 K`: `(1 + K) / 4` and `(3 - K) / 4` are
//! idempotents that split every state into its singlet part (on which `K` is 3) and its triplet
//! part (on which it is -1), and each part turns by its own rotor, multiplied on the right.
//!
//! The animation runs the exchange from spin up and spin down to the swapped pair and back: each
//! spin's Bloch vector shrinks to nothing as the pair becomes fully entangled and grows again
//! turned over, while the correlation ellipsoid (the image of the second spin's unit directions
//! under the correlation map) swells from a needle to a sphere; below, the largest Bell (CHSH)
//! combination rises from 2 to `2 sqrt 2` and falls back.

use gax_light::fade;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, palette, reach3, run,
};
use std::sync::OnceLock;

gax::algebra! {
    algebra pair "Two copies of Euclidean space, R(6,0): one spin's directions e1 e2 e3, the other's e4 e5 e6.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1;
    kind Scalar = [1];
    kind First = [e1, e2, e3];
    kind Second = [e4, e5, e6];
    kind FirstPlanes = [e23, e31, e12];
    kind SecondPlanes = [e56, e64, e45];
    kind FirstVolume = [e123];
    kind SecondVolume = [e456];
    kind FirstSpinor = [1, e23, e31, e12];
    kind Spinor = [1, e23, e31, e12, e56, e64, e45,
        e2356, e2364, e2345, e3156, e3164, e3145, e1256, e1264, e1245];
}

mod spins {
    use super::pair::*;

    /// A direction of the first spin, of the second, and a state of the pair.
    pub type F = First<(), f64>;
    pub type G = Second<(), f64>;
    pub type Sp = Spinor<(), f64>;
    /// The correlations as a map from the second spin's directions to the first's.
    pub type Correlation = First<(Second,), f64>;

    /// The first spin's directions and planes.
    pub fn x() -> F {
        First::new(1.0, 0.0, 0.0)
    }
    pub fn y() -> F {
        First::new(0.0, 1.0, 0.0)
    }
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn z() -> F {
        First::new(0.0, 0.0, 1.0)
    }
    pub fn yz() -> FirstPlanes<(), f64> {
        FirstPlanes::new(1.0, 0.0, 0.0)
    }
    pub fn zx() -> FirstPlanes<(), f64> {
        FirstPlanes::new(0.0, 1.0, 0.0)
    }
    pub fn xy() -> FirstPlanes<(), f64> {
        FirstPlanes::new(0.0, 0.0, 1.0)
    }
    /// The second spin's directions and planes.
    pub fn big_x() -> G {
        Second::new(1.0, 0.0, 0.0)
    }
    pub fn big_y() -> G {
        Second::new(0.0, 1.0, 0.0)
    }
    pub fn big_z() -> G {
        Second::new(0.0, 0.0, 1.0)
    }
    pub fn big_yz() -> SecondPlanes<(), f64> {
        SecondPlanes::new(1.0, 0.0, 0.0)
    }
    pub fn big_zx() -> SecondPlanes<(), f64> {
        SecondPlanes::new(0.0, 1.0, 0.0)
    }
    pub fn big_xy() -> SecondPlanes<(), f64> {
        SecondPlanes::new(0.0, 0.0, 1.0)
    }
    /// Each spin's pseudoscalar.
    pub fn i_first() -> FirstVolume<(), f64> {
        FirstVolume::new(1.0)
    }
    pub fn i_second() -> SecondVolume<(), f64> {
        SecondVolume::new(1.0)
    }

    /// One, as a state of the pair.
    pub fn one() -> Sp {
        Scalar::new(1.0).cast::<Spinor>()
    }
    /// The correlator `(1 - xy XY) / 2`.
    pub fn correlator() -> Sp {
        (one() - xy() * big_xy()) * 0.5
    }
    /// The pair's imaginary unit, `C xy`.
    pub fn imaginary() -> Sp {
        correlator() * xy()
    }
    /// The exchange coupling `yz YZ + zx ZX + xy XY`.
    pub fn coupling() -> Sp {
        yz() * big_yz() + zx() * big_zx() + xy() * big_xy()
    }
    /// The idempotents of the singlet and the triplet.
    pub fn singlet_part() -> Sp {
        (one() + coupling()) * 0.25
    }
    pub fn triplet_part() -> Sp {
        (one() * 3.0 - coupling()) * 0.25
    }

    /// The state after the exchange has acted for `angle` (coupling strength times time): its
    /// singlet part turned by three times the angle, its triplet part back by the angle.
    pub fn exchange(state: Sp, angle: f64) -> Sp {
        singlet_part() * state * (xy() * (3.0 * angle)).exp().into_inner()
            + triplet_part() * state * (xy() * -angle).exp().into_inner()
    }

    /// Each spin's Bloch vector: the part of `2 state C xy ~state` in its own planes, read as a
    /// vector of its space.
    pub fn bloch(state: Sp) -> (F, G) {
        let spin = state * imaginary() * state.reverse() * 2.0;
        let first = i_first().inverse() * spin.cast::<FirstPlanes>();
        let second = i_second().inverse() * spin.cast::<SecondPlanes>();
        (first, second)
    }

    /// The correlations as a map from the second spin's directions to the first's:
    /// `a | correlation(b)` is the expected product of the two spins' values along `a` and `b`.
    /// The density `2 state ~state` times the second spin's plane of `b`, its part in the first
    /// spin's planes, read as a vector.
    pub fn correlation(state: Sp) -> Correlation {
        let density = state * state.reverse() * 2.0;
        let planes = i_second() * Second::slot();
        i_first().inverse() * (density * planes).cast::<FirstPlanes>()
    }

    /// The largest Bell combination over all directions, `2 sqrt(s1² + s2²)` from the two
    /// largest singular values of the correlation map.
    pub fn bell(correlations: Correlation) -> f64 {
        let s = correlations.svdvals();
        2.0 * s[0].hypot(s[1])
    }

    /// The Bell combination along two directions of each spin as a multivector acting on states
    /// from the left: `(I a)(I' b)` is minus the product of the two spins' values along a and b.
    pub fn bell_element(a: F, a2: F, b: G, b2: G) -> Sp {
        -((i_first() * a) * (i_second() * (b + b2)) + (i_first() * a2) * (i_second() * (b - b2)))
    }

    /// The expectation of a multivector acting from the left: `2 <~state element state>`.
    pub fn expectation(element: Sp, state: Sp) -> f64 {
        2.0 * (state.reverse() * element * state).s()
    }

    /// The first spin up along z, the second down: a half turn in its ZX plane turns Z over.
    pub fn up_down() -> Sp {
        -big_zx() * correlator()
    }

    /// The exchange from spin up and spin down over the angle that swaps them: the angles, both
    /// Bloch vectors and the largest Bell combinations.
    pub struct Swap {
        pub angles: Vec<f64>,
        pub first: Vec<F>,
        pub second: Vec<G>,
        pub bells: Vec<f64>,
    }

    pub fn swap(frames: usize) -> Swap {
        let angles: Vec<f64> = (0..frames)
            .map(|k| core::f64::consts::FRAC_PI_4 * k as f64 / (frames - 1) as f64)
            .collect();
        let states: Vec<Sp> = angles.iter().map(|a| exchange(up_down(), *a)).collect();
        let (first, second) = states.iter().map(|s| bloch(*s)).unzip();
        let bells = states.iter().map(|s| bell(correlation(*s))).collect();
        Swap {
            angles,
            first,
            second,
            bells,
        }
    }

    /// The singlet (the singlet part of spin up and spin down, normalized), the Bell element along
    /// the directions at which it reaches `2 sqrt 2`, and its value there.
    pub fn singlet() -> (Sp, Sp, f64) {
        let state = singlet_part() * up_down() * core::f64::consts::SQRT_2;
        // The first spin's directions a quarter turn apart; the second spin's halfway between
        // them, reversed, since the singlet's spins disagree along every direction.
        let h = core::f64::consts::FRAC_1_SQRT_2;
        let element = bell_element(x(), y(), (big_x() + big_y()) * -h, (big_x() - big_y()) * -h);
        (state, element, expectation(element, state))
    }
}

use spins::*;

/// The exchange over its whole range, computed once.
fn swapped() -> &'static Swap {
    static S: OnceLock<Swap> = OnceLock::new();
    S.get_or_init(|| swap(121))
}

/// The drawing's space.
type Space = gax::vga3d::Vector<(), f64>;

/// The unit directions of space.
fn axes() -> [Space; 3] {
    [
        Space::new(1.0, 0.0, 0.0),
        Space::new(0.0, 1.0, 0.0),
        Space::new(0.0, 0.0, 1.0),
    ]
}

/// Each spin's directions drawn as the directions of space: maps from the pair's algebra.
fn first_in_space() -> gax::vga3d::Vector<(pair::First,), f64> {
    gax::vga3d::Vector::from_images(axes())
}
fn second_in_space() -> gax::vga3d::Vector<(pair::Second,), f64> {
    gax::vga3d::Vector::from_images(axes())
}

/// The second spin's direction for each direction of space: the other way round.
fn space_in_second() -> pair::Second<(gax::vga3d::Vector,), f64> {
    pair::Second::from_images([big_x(), big_y(), big_z()])
}

/// The unit reach3, faintly, with its three axes.
fn draw_ball(s: &mut Scene3) {
    s.sphere_wire(ORIGIN3, 1.0, 18, fade(palette::grid(), 0.55));
    for a in axes() {
        s.seg(reach3(-a), reach3(a), 1.0, palette::grid());
    }
}

/// Axis names just beyond the ends of the axes.
fn axis_names(c: &mut Canvas, cam: &Camera) {
    for (name, a) in ["X", "Y", "Z"].iter().zip(axes()) {
        if let Some(p) = cam.px(reach3(a * 1.18)) {
            let at = p + Point2::direction(0.0, 4.0);
            c.text(name, at, 11.0, palette::grid(), Align::Center);
        }
    }
}

/// The direction of space at longitude `lon` and latitude `lat`: `x` raised towards `z`, then
/// turned about `z`.
fn on_sphere(lon: f64, lat: f64) -> Space {
    use gax::vga3d::Bivector;
    let turn = (Bivector::new(0.0, 0.0, 1.0) * (-lon / 2.0)).exp()
        * (Bivector::new(0.0, 1.0, 0.0) * (lat / 2.0)).exp();
    turn >> Space::new(1.0, 0.0, 0.0)
}

/// The image of the second spin's unit sphere under the correlation map, as a map of space.
fn ellipsoid(s: &mut Scene3, corr: Correlation, colour: Light) {
    let shown = first_in_space().of(corr.of(space_in_second()));
    s.surface(
        |u, v| {
            let lon = core::f64::consts::TAU * f64::from(u);
            let lat = core::f64::consts::PI * (f64::from(v) - 0.5);
            reach3(shown.of(on_sphere(lon, lat)))
        },
        28,
        14,
        |_, _| colour,
        0.35,
        Some((fade(colour, 0.6), 0.8)),
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let (up, down) = (Point2::direction(0.0, -1.0), Point2::direction(0.0, 1.0));
    let tau = core::f64::consts::TAU;
    let phase = f64::from(t) / 8.0;
    // From up-down to swapped and back, easing in and out.
    let angle = core::f64::consts::FRAC_PI_8 * (1.0 - (tau * phase).cos());
    let state = exchange(up_down(), angle);
    let data = swapped();
    let (first, second) = bloch(state);
    let corr = correlation(state);
    let value = bell(corr);
    let azimuth = (-55.0f32).to_radians() + 0.35 * (tau * phase).sin() as f32;
    let elevation = 18.0f32.to_radians();
    // The band of the three balls, in thirds.
    let band = Rect::new(0.0, h * 92.0 / 540.0, w, h * 392.0 / 540.0);
    let colours: [Light; 3] = [palette::red(), palette::purple(), palette::sky()];
    let titles = ["FIRST SPIN", "CORRELATIONS", "SECOND SPIN"];
    // Each Bloch vector in space, with its tip's path over the whole exchange.
    let blochs: [(Space, Vec<Space>); 2] = [
        (
            first_in_space().of(first),
            data.first.iter().map(|v| first_in_space().of(*v)).collect(),
        ),
        (
            second_in_space().of(second),
            data.second
                .iter()
                .map(|v| second_in_space().of(*v))
                .collect(),
        ),
    ];
    for k in 0..3 {
        let rect = band.column(k, 3);
        let cam = Camera::orbit(
            rect.width() as usize,
            rect.height() as usize,
            ORIGIN3,
            4.4,
            azimuth,
            elevation,
            Lens::Perspective(0.62),
        );
        let drawn = panel3(c, rect, cam, |s| {
            draw_ball(s);
            match k {
                1 => ellipsoid(s, corr, colours[1]),
                _ => {
                    let (now, path) = &blochs[k / 2];
                    let tips: Vec<_> = path.iter().map(|r| reach3(*r)).collect();
                    s.polyline(&tips, 2.0, fade(colours[k], 0.35));
                    s.arrow(ORIGIN3, *now, 3.0, 11.0, colours[k]);
                }
            }
        });
        axis_names(c, &drawn);
        let title = rect.top_middle() + up.gp(4.0);
        c.text(titles[k], title, 13.0, colours[k], Align::Center);
    }
    // The lengths of the Bloch vectors under their balls.
    for (k, (now, _)) in [0, 2].into_iter().zip(&blochs) {
        let note = (band.column(k, 3)).bottom_middle() + down.gp(2.0);
        let text = format!("LENGTH {:.2}", now.norm());
        c.text(&text, note, 11.0, palette::ink(), Align::Center);
    }
    // The largest Bell combination along the exchange.
    let quarter = core::f32::consts::FRAC_PI_4;
    let below = Rect {
        lo: band.bottom_left(),
        hi: screen.hi,
    };
    let plot = below.inset(w * 0.073, h * 34.0 / 540.0, w * 0.03, h * 42.0 / 540.0);
    let ax = Axes::new(plot, [0.0, quarter], [1.9, 2.95]);
    ax.frame(c, "", "EXCHANGE ANGLE (RAD)", "LARGEST BELL VALUE");
    let root8 = 2.0 * core::f32::consts::SQRT_2;
    for (level, dash) in [(2.0, 6.0), (root8, 2.0)] {
        let across = [Point2::xy(0.0, level), Point2::xy(quarter, level)];
        ax.dashed(c, &across, 1.0, dash, palette::grid());
    }
    let small = 9.0;
    let note = Point2::xy(quarter / 2.0, 2.04);
    let text = "EACH SPIN ITS OWN ANSWERS";
    ax.text(c, note, text, small, palette::grid(), Align::Center);
    let note = Point2::xy(0.01, root8 + 0.04);
    ax.text(c, note, "2 SQRT 2", small, palette::grid(), Align::Left);
    let curve: Vec<Point2> = data
        .angles
        .iter()
        .zip(&data.bells)
        .map(|(a, b)| Point2::xy(*a as f32, *b as f32))
        .collect();
    ax.polyline(c, &curve, 1.6, colours[1]);
    let now = Point2::xy(angle as f32, value as f32);
    ax.scatter(c, &[now], Marker::Dot, 8.0, colours[1]);
    // The singlet's Bell combination along the directions that reach the bound, and the
    // state's norm, which the exchange keeps.
    let (_, _, reached) = singlet();
    let norm = expectation(one(), state);
    caption(
        c,
        "TWO SPINS: EXCHANGE AND ENTANGLEMENT",
        &format!("R(6,0)  ANGLE {angle:.3}  BELL {value:.3}  NORM {norm:.3}  SINGLET {reached:.3}"),
    );
}

fn main() {
    run(Anim::new("two spins", 8.0).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::pair::*;
    use super::spins::*;
    use gax_numga_examples::rng::{Draw, rng};

    fn small(s: Sp, tol: f64) -> bool {
        s.c.iter().all(|v| v.abs() <= tol)
    }

    /// The singlet and triplet parts are idempotent and add up to one; the coupling squares to
    /// `3 + 2 K`.
    #[test]
    fn the_coupling_splits_states_into_singlet_and_triplet() {
        let k = coupling();
        assert!(small(k * k - one() * 3.0 - k * 2.0, 1e-12));
        let s = singlet_part();
        assert!(small(s * s - s, 1e-12));
        assert!(small(s + triplet_part() - one(), 1e-12));
    }

    /// numga's `swap` checks: the exchange keeps the state normalized and both Bloch vectors
    /// equally long; halfway the spins are fully entangled, with no Bloch vector and the Bell
    /// combination at `2 sqrt 2`; at the end they have swapped.
    #[test]
    fn the_exchange_swaps_up_and_down_through_a_fully_entangled_state() {
        for frames in [121, 21] {
            let s = swap(frames);
            for (angle, (a, b)) in s.angles.iter().zip(s.first.iter().zip(&s.second)) {
                let state = exchange(up_down(), *angle);
                assert!((expectation(one(), state) - 1.0).abs() < 1e-7);
                assert!(((*a | *a).s() - (*b | *b).s()).abs() < 1e-7);
            }
            let middle = frames / 2;
            assert!(s.first[middle].c.iter().all(|v| v.abs() < 1e-7));
            let last = frames - 1;
            for (i, want) in [(0, 2.0), (middle, 2.0 * 2f64.sqrt()), (last, 2.0)] {
                assert!((s.bells[i] - want).abs() < 1e-7, "{i}: {}", s.bells[i]);
            }
            assert!((s.first[last] + z()).c.iter().all(|v| v.abs() < 1e-7));
            assert!((s.second[last] - big_z()).c.iter().all(|v| v.abs() < 1e-7));
        }
    }

    /// numga's `singlet` checks: normalized, the coupling acts on it as three, no Bloch vector,
    /// every direction anticorrelated with the same direction of the other spin; the Bell element
    /// squares to `4 - 4 (a ^ a')(b ^ b')` for any unit directions, and the singlet reaches the
    /// bound `2 sqrt 2` this sets.
    #[test]
    fn the_singlet_reaches_the_bell_bound() {
        let (state, _, value) = singlet();
        assert!((expectation(one(), state) - 1.0).abs() < 1e-12);
        assert!(small(coupling() * state - state * 3.0, 1e-12));
        let (first, _) = bloch(state);
        assert!(first.c.iter().all(|v| v.abs() < 1e-12));
        let corr = correlation(state);
        for (across, along) in [(big_x(), x()), (big_y(), y()), (big_z(), z())] {
            assert!((corr.of(across) + along).c.iter().all(|v| v.abs() < 1e-12));
        }
        // Any unit directions, drawn at random.
        let mut rng = rng(0x5eed);
        for _ in 0..20 {
            let mut f = || First::from_coeffs(rng.direction());
            let (a, a2) = (f(), f());
            let mut g = || Second::from_coeffs(rng.direction());
            let (b, b2) = (g(), g());
            let e = bell_element(a, a2, b, b2);
            let want = one() * 4.0 - (a ^ a2) * (b ^ b2) * 4.0;
            assert!(small(e * e - want, 1e-12));
        }
        assert!((value - 2.0 * 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let anim = gax_numga_examples::Anim::new("t", 8.0).size(480, 270);
        let a = gax_numga_examples::app::frame(&anim, 0.5, &mut draw);
        let b = gax_numga_examples::app::frame(&anim, 3.0, &mut draw);
        assert!(gax_light::luma(a.mean()) > 0.0);
        assert!(a.mean() != b.mean());
    }
}
