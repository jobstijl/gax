//! numga's `quantum/neutrino_isospin`: neutrino flavour oscillations as an open quantum system, in
//! the Pauli algebra of space (VGA3D). A neutrino is made and detected in a flavour but travels
//! as a mix of mass states, so its flavour turns as it goes. Its state is `(1 + P) / 2`, a scalar
//! plus its isospin vector `P` (VGA3D's `Paravector`): `+z` is the electron flavour, `-z` the
//! muon flavour, and the chance of finding an electron neutrino is `(1 + P · z) / 2`.
//!
//! The Hamiltonian is a bivector: the phase plane `xy` turned by twice the mixing angle in the
//! `xz` plane, by a rotor. It turns states by its commutator, `-[H, rho]`, a map on states;
//! the separation of the mass states' wave packets adds a Lindblad dissipator, the half turn `l`
//! in the mass basis's phase plane, `l rho ~l - rho` (as numga; a vector `l` along the mass axis
//! would reflect, which turns the state's vector part over). The transfer over a step is the
//! exponential of the rates, to fifth order as a sum of powers of the map, and a journey is
//! these maps applied one after another.
//!
//! The animation runs three journeys side by side: vacuum oscillations, where the isospin
//! precesses about the Hamiltonian's axis; wave packet decoherence, where it spirals in onto
//! the mass axis and the electron flavour settles at `(1 + (l · z)²) / 2`; and the solar MSW
//! effect, where the electron density falls away from the Sun's core, the Hamiltonian's plane
//! tilts from near the phase plane to the vacuum one, and the isospin follows it adiabatically
//! from electron to muon flavour. Below, the two flavours' probabilities along the way.

use gax::vga3d::Bivector;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::{frame, wave};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, palette, reach3, run,
};
use std::sync::OnceLock;

mod neutrino {
    use gax::Unit;
    use gax::vga3d::{Bivector, Paravector, Pseudoscalar, Rotor, Vector};
    use gax_numga_examples::signal::linspace;

    /// A state `(1 + P) / 2`, and its isospin vector.
    pub type St = Paravector;
    /// A rate of change of states, and a transfer over a stretch of the journey: maps on states.
    pub type Rates = Paravector<(Paravector,)>;
    pub type Transfer = Paravector<(Paravector,)>;

    /// The order of the exponential series of a step's transfer.
    const SERIES_ORDER: u32 = 5;

    pub fn one() -> St {
        Paravector::new(1.0, 0.0, 0.0, 0.0)
    }
    pub fn x() -> Vector {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn y() -> Vector {
        Vector::new(0.0, 1.0, 0.0)
    }
    /// The electron flavour, `+z`, and the muon flavour, `-z`.
    pub fn electron() -> Vector {
        Vector::new(0.0, 0.0, 1.0)
    }
    pub fn muon() -> Vector {
        -electron()
    }
    /// The plane in which the quantum phase accumulates.
    pub fn phase_plane() -> Bivector {
        x() ^ y()
    }
    /// The plane in which the flavour basis turns into the mass basis.
    fn mixing_plane() -> Bivector {
        x() ^ electron()
    }

    /// The open state.
    fn open() -> Rates {
        Paravector::slot()
    }

    /// The state with isospin vector `p`.
    pub fn state(p: Vector) -> St {
        (one() + p.cast::<Paravector>()) * 0.5
    }

    /// The isospin vector of a state: twice its vector part.
    pub fn isospin(rho: St) -> Vector {
        rho.cast::<Vector>() * 2.0
    }

    /// The chances of the electron and the muon flavour: the isospin's projection on the flavour
    /// axis, from -1 to 1, read as a probability.
    pub fn flavour_probabilities(p: Vector) -> (f64, f64) {
        let along = (p | electron()).s();
        (0.5 * (1.0 + along), 0.5 * (1.0 - along))
    }

    /// The rotor that turns the flavour basis into the mass basis: twice the mixing angle in the
    /// `xz` plane, since a rotor turns by twice its angle.
    pub fn mixing_rotor(theta: f64) -> Unit<Rotor> {
        (mixing_plane() * -theta).exp()
    }

    /// The Hamiltonian in vacuum: the phase plane, reversed, turned by the mixing rotor.
    pub fn vacuum_hamiltonian(omega: f64, theta: f64) -> Bivector {
        (mixing_rotor(theta) >> -phase_plane()) * omega
    }

    /// The Hamiltonian in matter: the electrons' potential adds to the phase plane.
    pub fn matter_hamiltonian(vacuum: Bivector, potential: f64) -> Bivector {
        vacuum + phase_plane() * potential
    }

    /// The potential at which the matter Hamiltonian has no part in the phase plane, so that it
    /// turns the flavours fully into each other: the MSW resonance, `omega cos 2 theta`.
    pub fn resonance_potential(vacuum: Bivector) -> f64 {
        vacuum.scalar_product(phase_plane()).s()
    }

    /// The turning by a Hamiltonian: minus its commutator with the state.
    pub fn coherent_generator(plane: Bivector) -> Rates {
        -plane.commutator(open()).cast::<Paravector>()
    }

    /// The Lindblad dissipator of a process `l`, `l rho ~l - {~l l, rho} / 2`. For a bivector
    /// `~l l` is its squared norm, so the anticommutator is plain scaling. The sandwich is the
    /// bivector's as a rotor (a half turn, for a unit one), and it keeps a state a state; gax
    /// types it as a multivector, so the empty grades are dropped.
    pub fn relaxation(process: Bivector) -> Rates {
        (Rotor::from(process) >> open()).cast::<Paravector>() - open() * process.norm_squared()
    }

    /// The decoherence of the wave packets' separation at `gamma`: dephasing by the half turn in
    /// the given phase plane.
    pub fn dephasing_generator(plane: Bivector, gamma: f64) -> Rates {
        relaxation(plane) * gamma
    }

    /// The transfer over `dx`: the exponential of the rates times dx, its powers by composition.
    pub fn evolution(rates: Rates, dx: f64) -> Transfer {
        let small = rates * dx;
        let mut term = open();
        let mut total = term;
        for order in 1..=SERIES_ORDER {
            term = small.of(term) / f64::from(order);
            total += term;
        }
        total
    }

    /// The isospin before each of the transfers, applied one after another to `initial`.
    pub fn propagate(initial: St, transfers: impl IntoIterator<Item = Transfer>) -> Vec<Vector> {
        let mut rho = initial;
        transfers
            .into_iter()
            .map(|transfer| {
                let p = isospin(rho);
                rho = transfer.of(rho);
                p
            })
            .collect()
    }

    /// A journey: the distances, the isospin at each and the Hamiltonian's plane there.
    pub struct Journey {
        pub distances: Vec<f64>,
        pub isospins: Vec<Vector>,
        pub planes: Vec<Bivector>,
    }

    /// `steps` distances evenly from zero to `end`, and the step between them.
    fn distances(end: f64, steps: usize) -> (Vec<f64>, f64) {
        (linspace(0.0, end, steps), end / (steps - 1) as f64)
    }

    /// The distance over which the electron flavour's chance in vacuum comes back,
    /// `2 pi / omega`: the chance goes as `sin²(omega L / 2)`.
    pub fn oscillation_length(omega: f64) -> f64 {
        core::f64::consts::TAU / omega
    }

    /// An electron neutrino oscillating in vacuum over four oscillation lengths (numga's two
    /// "wavelengths" of `4 pi / omega`): one transfer map, applied over and over.
    pub fn vacuum_oscillations(theta_deg: f64, omega: f64, steps: usize) -> Journey {
        let plane = vacuum_hamiltonian(omega, theta_deg.to_radians());
        let (distances, dx) = distances(4.0 * oscillation_length(omega), steps);
        let transfer = evolution(coherent_generator(plane), dx);
        Journey {
            distances,
            isospins: propagate(state(electron()), vec![transfer; steps]),
            planes: vec![plane; steps],
        }
    }

    /// The length of the decoherence journey and of the solar one.
    pub const JOURNEY: f64 = 100.0;

    /// An electron neutrino whose wave packets separate as it oscillates: it dephases onto the
    /// mass axis. The journey, and the electron flavour's chance it settles at: the mass axis's
    /// part along the flavour axis, `(1 + (l · z)²) / 2 = 1 - sin²(2 theta) / 2`.
    pub fn wavepacket_decoherence(
        theta_deg: f64,
        omega: f64,
        gamma: f64,
        steps: usize,
    ) -> (Journey, f64) {
        let theta = theta_deg.to_radians();
        let plane = vacuum_hamiltonian(omega, theta);
        let mass_plane = mixing_rotor(theta) >> phase_plane();
        let (distances, dx) = distances(JOURNEY, steps);
        let rates = coherent_generator(plane) + dephasing_generator(mass_plane, gamma);
        let transfer = evolution(rates, dx);
        let journey = Journey {
            distances,
            isospins: propagate(state(electron()), vec![transfer; steps]),
            planes: vec![plane; steps],
        };
        // The mass axis's part along the flavour axis: the volume the mass plane spans with it.
        let along = (mass_plane ^ electron()).norm();
        (journey, 0.5 * (1.0 + along * along))
    }

    /// The scale over which the Sun's electron density falls by a factor e, and the density at
    /// the core, in units of the resonance's.
    const DENSITY_SCALE: f64 = 25.0;
    const CORE_DENSITY: f64 = 3.0;

    /// An electron neutrino made in the Sun's core, where the electron density is three times
    /// the resonance's, falling off outward: each step's transfer from the matter Hamiltonian
    /// there. The journey, and the potentials along it.
    pub fn solar_adiabatic(theta_deg: f64, omega: f64, steps: usize) -> (Journey, Vec<f64>) {
        let vacuum = vacuum_hamiltonian(omega, theta_deg.to_radians());
        let resonance = resonance_potential(vacuum);
        let (distances, dx) = distances(JOURNEY, steps);
        let potentials: Vec<f64> = distances
            .iter()
            .map(|d| CORE_DENSITY * resonance * (-d / DENSITY_SCALE).exp())
            .collect();
        let planes: Vec<Bivector> = potentials
            .iter()
            .map(|v| matter_hamiltonian(vacuum, *v))
            .collect();
        let transfers = planes
            .iter()
            .map(|plane| evolution(coherent_generator(*plane), dx));
        let isospins = propagate(state(electron()), transfers);
        let journey = Journey {
            distances,
            isospins,
            planes,
        };
        (journey, potentials)
    }

    /// The turning axis of a Hamiltonian, perpendicular to its plane: its dual.
    pub fn axis(plane: Bivector) -> Vector {
        Pseudoscalar::new(1.0) * plane
    }

    /// The unit circle in a plane, in `count` steps: a direction in the plane (the plane's
    /// contraction with the axis it leans on least) turned about its normal.
    pub fn great_circle(plane: Bivector, count: usize) -> Vec<Vector> {
        let unit = plane.normalized();
        let start = [x(), y(), electron()]
            .map(|a| a.lc(unit))
            .into_iter()
            .max_by(|a, b| a.norm_squared().total_cmp(&b.norm_squared()))
            .expect("three directions")
            .normalized();
        (0..=count)
            .map(|k| {
                let angle = core::f64::consts::TAU * k as f64 / count as f64;
                (unit * (-angle / 2.0)).exp() >> start
            })
            .collect()
    }
}

use neutrino::*;

const SECONDS: f32 = 12.0;
/// The mixing angles (degrees) and the frequency of the journeys, numga's.
const THETA_VACUUM: f64 = 30.0;
const THETA_SOLAR: f64 = 10.0;
const OMEGA: f64 = 1.0;
const GAMMA: f64 = 0.05;
/// The journeys' places in the panels, and their steps, numga's.
const VACUUM: usize = 0;
const DECOHERENCE: usize = 1;
const SOLAR: usize = 2;
const STEPS: [usize; 3] = [300, 400, 600];

/// The three journeys, computed once.
struct Data {
    journeys: [Journey; 3],
    asymptote: f64,
    potentials: Vec<f64>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let vacuum = vacuum_oscillations(THETA_VACUUM, OMEGA, STEPS[VACUUM]);
        let (decoherence, asymptote) =
            wavepacket_decoherence(THETA_VACUUM, OMEGA, GAMMA, STEPS[DECOHERENCE]);
        let (solar, potentials) = solar_adiabatic(THETA_SOLAR, OMEGA, STEPS[SOLAR]);
        Data {
            journeys: [vacuum, decoherence, solar],
            asymptote,
            potentials,
        }
    })
}

/// The panels' titles and colours.
const TITLES: [&str; 3] = [
    "VACUUM OSCILLATIONS",
    "WAVE PACKET DECOHERENCE",
    "SOLAR MSW CONVERSION",
];
fn colours() -> [Light; 3] {
    [palette::ORANGE, palette::PURPLE, palette::GREEN]
}

/// The isospin sphere: a faint wireframe, the axes, and the two flavours at the poles.
fn draw_sphere(s: &mut Scene3) {
    s.sphere_wire(ORIGIN3, 1.0, 16, palette::GRID.faded(0.5));
    for a in [x(), y(), electron()] {
        s.seg(reach3(-a), reach3(a), 1.0, palette::GRID);
    }
    s.dot(reach3(electron()), Marker::Dot, 8.0, palette::SKY);
    s.dot(reach3(muon()), Marker::Dot, 8.0, palette::RED);
}

/// A Hamiltonian's plane as its great circle, and its turning axis through the sphere.
fn draw_plane(s: &mut Scene3, plane: Bivector, colour: Light) {
    let circle: Vec<_> = great_circle(plane, 96).into_iter().map(reach3).collect();
    s.polyline(&circle, 1.4, colour);
    let n = axis(plane).normalized() * 1.15;
    s.seg(reach3(-n), reach3(n), 1.0, colour.faded(0.6));
}

/// The flavour names beside the poles.
fn pole_names(c: &mut Canvas, cam: &Camera) {
    let beside = Point2::direction(8.0, 4.0);
    for (name, pole, colour) in [
        ("NU E", electron(), palette::SKY),
        ("NU MU", muon(), palette::RED),
    ] {
        if let Some(p) = cam.px(reach3(pole * 1.12)) {
            c.text(name, p + beside, 10.0, colour, Align::Left);
        }
    }
}

/// One journey on its sphere up to step `now`: the Hamiltonian's plane (where it changes, the
/// first faintly), the path so far and the isospin.
fn draw_journey(s: &mut Scene3, journey: &Journey, now: usize, colour: Light) {
    draw_sphere(s);
    let (first, current) = (journey.planes[0], journey.planes[now]);
    if (current - first).norm() > 1e-9 {
        draw_plane(s, first, palette::INK.faded(0.1));
    }
    draw_plane(s, current, palette::INK.faded(0.2));
    let path: Vec<_> = journey.isospins[..=now]
        .iter()
        .map(|p| reach3(*p))
        .collect();
    s.polyline(&path, 1.6, colour.faded(0.8));
    s.arrow(ORIGIN3, journey.isospins[now], 2.6, 10.0, colour);
}

/// The two flavours' chances along a journey: faint over the whole of it, bright up to step
/// `now`, with markers at the step.
fn draw_probabilities(c: &mut Canvas, ax: &Axes, journey: &Journey, now: usize) {
    let (electron, muon): (Vec<Point2>, Vec<Point2>) = journey
        .distances
        .iter()
        .zip(&journey.isospins)
        .map(|(x, p)| {
            let (e, m) = flavour_probabilities(*p);
            let x = *x as f32;
            (Point2::xy(x, e as f32), Point2::xy(x, m as f32))
        })
        .unzip();
    let flavours = [(electron, palette::SKY), (muon, palette::RED)];
    for (curve, colour) in &flavours {
        ax.polyline(c, curve, 1.0, colour.faded(0.25));
    }
    for (curve, colour) in &flavours {
        ax.polyline(c, &curve[..=now], 1.8, *colour);
        ax.scatter(c, &[curve[now]], Marker::Dot, 6.0, *colour);
    }
}

/// The marks particular to each journey's plot: the level decoherence settles at, and the
/// Sun's electron density (scaled to its value at the core).
fn draw_references(c: &mut Canvas, ax: &Axes, k: usize, d: &Data) {
    let journey = &d.journeys[k];
    let end = *journey.distances.last().expect("a step") as f32;
    let small = 9.0;
    match k {
        DECOHERENCE => {
            let level = d.asymptote as f32;
            let across = [Point2::xy(0.0, level), Point2::xy(end, level)];
            ax.dashed(c, &across, 1.0, 5.0, palette::INK.faded(0.5));
            let note = Point2::xy(end * 0.98, level + 0.05);
            let text = format!("SETTLES AT {level:.3}");
            ax.text(c, note, &text, small, palette::INK, Align::Right);
        }
        SOLAR => {
            let core = d.potentials[0];
            let density: Vec<Point2> = journey
                .distances
                .iter()
                .zip(&d.potentials)
                .map(|(x, v)| Point2::xy(*x as f32, (v / core) as f32))
                .collect();
            ax.dashed(c, &density, 1.0, 3.0, palette::YELLOW.faded(0.6));
            let note = Point2::xy(end * 0.4, 0.5);
            let text = "ELECTRON DENSITY";
            ax.text(c, note, text, small, palette::YELLOW, Align::Left);
        }
        _ => {}
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let screen = c.rect();
    let w = screen.width();
    let up = Point2::direction(0.0, -1.0);
    let progress = (t / SECONDS).fract();
    let tau = core::f32::consts::TAU;
    let azimuth = (-60.0f32).to_radians() + 0.25 * wave(tau * progress);
    let band = Rect::new(0.0, 84.0, w, 330.0);
    let plots = Rect::new(0.0, 360.0, w, screen.height() - 36.0);
    for (k, (journey, colour)) in d.journeys.iter().zip(colours()).enumerate() {
        let steps = journey.isospins.len();
        let now = frame(progress, steps);
        let rect = band.column(k, 3);
        let cam = Camera::orbit(
            rect,
            ORIGIN3,
            4.4,
            azimuth,
            20.0f32.to_radians(),
            Lens::Perspective(0.6),
        );
        let drawn = panel3(c, rect, cam, |s| draw_journey(s, journey, now, colour));
        pole_names(c, &drawn);
        let title = rect.top_middle() + up.gp(2.0);
        c.text(TITLES[k], title, 12.0, colour, Align::Center);
        let end = *journey.distances.last().expect("a step") as f32;
        let plot = plots.column(k, 3).inset(42.0, 8.0, 14.0, 0.0);
        let ax = Axes::new(plot, [0.0, end], [-0.05, 1.05]);
        ax.frame(c, "", "DISTANCE", "FLAVOUR CHANCE");
        draw_references(c, &ax, k, d);
        draw_probabilities(c, &ax, journey, now);
    }
    let legend = Axes::new(
        plots.column(0, 3).inset(42.0, 8.0, 14.0, 0.0),
        [0.0, 1.0],
        [0.0, 1.0],
    );
    legend.legend(c, &[("NU E", palette::SKY), ("NU MU", palette::RED)]);
    caption(
        c,
        "NEUTRINO FLAVOUR ISOSPIN",
        "VGA3D: STATES (1 + P) / 2, TRANSFERS EXP(RATES DX) AS MAPS STATE <- STATE",
    );
}

fn main() {
    run(Anim::new("neutrino isospin", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::neutrino::*;

    /// The chance of the electron flavour along a journey.
    fn electron_chances(journey: &Journey) -> Vec<f64> {
        journey
            .isospins
            .iter()
            .map(|p| flavour_probabilities(*p).0)
            .collect()
    }

    /// numga's `test_flavor_probabilities`: the pure flavours are certain.
    #[test]
    fn pure_flavours_have_certain_probabilities() {
        for (flavour, want) in [(electron(), (1.0, 0.0)), (muon(), (0.0, 1.0))] {
            let (e, m) = flavour_probabilities(flavour);
            assert!((e - want.0).abs() < 1e-12 && (m - want.1).abs() < 1e-12);
        }
    }

    /// numga's `test_vacuum_oscillation_transfer_map`: the transfer maps match the two-flavour
    /// formula `1 - sin²(2 theta) sin²(omega L / 2)`.
    #[test]
    #[allow(clippy::disallowed_methods)] // the analytic reference it is checked against
    fn vacuum_oscillations_match_the_two_flavour_formula() {
        let (theta_deg, omega) = (30.0f64, 1.2);
        let journey = vacuum_oscillations(theta_deg, omega, 200);
        let mixing = (2.0 * theta_deg.to_radians()).sin().powi(2);
        for (got, l) in electron_chances(&journey).iter().zip(&journey.distances) {
            let want = 1.0 - mixing * (0.5 * omega * l).sin().powi(2);
            assert!((got - want).abs() < 1e-5, "{l}: {got} vs {want}");
        }
    }

    /// numga's `test_wavepacket_decoherence_asymptotic`: dephasing damps the oscillations to
    /// `cos⁴ theta + sin⁴ theta`.
    #[test]
    #[allow(clippy::disallowed_methods)] // the analytic reference it is checked against
    fn decoherence_settles_at_the_mixture_of_mass_states() {
        let theta_deg = 30.0f64;
        let (journey, asymptote) = wavepacket_decoherence(theta_deg, 1.0, 0.08, 400);
        let (c, s) = (theta_deg.to_radians().cos(), theta_deg.to_radians().sin());
        assert!((asymptote - (c.powi(4) + s.powi(4))).abs() < 1e-12);
        let last = *electron_chances(&journey).last().expect("a step");
        assert!((last - asymptote).abs() < 1e-3, "{last} vs {asymptote}");
    }

    /// numga's `test_solar_msw_transfer_channel`: over 95% of the electron neutrinos made in the
    /// core leave as muon neutrinos.
    #[test]
    fn the_sun_converts_electron_neutrinos() {
        let (journey, _) = solar_adiabatic(10.0, 1.0, 600);
        let chances = electron_chances(&journey);
        assert!((chances[0] - 1.0).abs() < 1e-12);
        assert!(*chances.last().expect("a step") < 0.05);
    }

    /// numga's `test_trace_conservation`: the state's scalar part, its trace, stays a half
    /// under turning and dephasing alike.
    #[test]
    fn the_trace_is_kept() {
        let theta = 30f64.to_radians();
        let mass_plane = mixing_rotor(theta) >> phase_plane();
        let rates = coherent_generator(vacuum_hamiltonian(1.0, theta))
            + dephasing_generator(mass_plane, 0.1);
        let step = evolution(rates, 0.05);
        let mut rho = state(electron());
        for _ in 0..50 {
            rho = step.of(rho);
            assert!((rho.s() - 0.5).abs() < 1e-12, "{rho:?}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
