//! numga's `relativity/dirac`: the Dirac electron in the spacetime algebra (STA). An electron's
//! state at a point is an even multivector `ψ = exp(I β / 2) R √ρ`: a density, an angle `β` and
//! a Lorentz rotor. Its sandwich on an open vector slot, `ψ >> Vector::slot()`, is a map on
//! spacetime that carries the observer's frame to the electron's: the image of the time axis
//! is the current, that of the `z` axis the spin. `β` drops out on vectors and shows on
//! bivectors.
//!
//! For a plane wave the Dirac equation is a linear map on even multivectors, the Hamiltonian
//! `H(ψ) = p t ψ + m t ψ t`. Paired with the Hermitian product `⟨(t ψ̃ t) φ⟩` it is a symmetric
//! form, and the generalized eigenproblem gives its energies `±√(p² + m²)`, each fourfold: the
//! mass shell. A mixture of positive and negative energy makes the current circulate at twice
//! the energy, the trembling motion (Zitterbewegung). The animation traces the paths of three
//! such electrons, with their spins, beside the two sheets of the mass shell.

use std::sync::OnceLock;

use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, Scene3,
    backdrop, caption, palette, run, signal::wave,
};

mod dirac {
    use gax::sta::{Bivector, Even, Phasor, Pseudoscalar, Scalar, Vector};
    use gax::vga3d;

    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    /// A Dirac spinor is an even multivector.
    pub type Spinor = Even<(), f64>;
    /// A vector of the observer's space.
    pub type Space = vga3d::Vector<(), f64>;
    /// A point of the observer's space.
    pub type Place = gax::pga3d::Point<(), f64>;
    /// The spinor's map on spacetime.
    pub type Frame = Vector<(Vector,), f64>;
    /// The plane-wave Dirac equation, a map on spinors.
    pub type Hamiltonian = Even<(Even,), f64>;

    /// The observer's time axis.
    pub fn time() -> V {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }

    /// The `x` axis.
    pub fn x_axis() -> V {
        Vector::new(0.0, 1.0, 0.0, 0.0)
    }

    /// The `z` axis.
    pub fn z_axis() -> V {
        Vector::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The spin plane `y ^ x`: right multiplication by it squares to minus one.
    pub fn spin_plane() -> B {
        Bivector::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    }

    /// A scalar as a spinor.
    pub fn scalar(s: f64) -> Spinor {
        Scalar::new(s).into()
    }

    /// `exp(I angle)`, with `I = e0123` squaring to minus one: `cos + I sin`, a phasor.
    pub fn phase(angle: f64) -> Phasor<(), f64> {
        Pseudoscalar::new(angle).exp()
    }

    // Used by the tests (part of numga's core).
    #[cfg_attr(not(test), allow(dead_code))]
    /// The spinor with the given density, angle `β` and Lorentz rotor.
    #[allow(clippy::disallowed_methods)] // the amplitude √ρ of a density, not a length
    pub fn spinor(density: f64, beta: f64, rotor: Spinor) -> Spinor {
        phase(beta / 2.0) * rotor * density.sqrt()
    }

    /// The spinor's map on spacetime: the density times its Lorentz transformation. `β` drops
    /// out, because the pseudoscalar anticommutes with vectors.
    pub fn frame(psi: Spinor) -> Frame {
        psi >> Vector::slot()
    }

    // Used by the tests (part of numga's core).
    #[cfg_attr(not(test), allow(dead_code))]
    /// `ψ ψ̃ = exp(I β) ρ`: a scalar plus a pseudoscalar (its bivector part vanishes).
    pub fn invariants(psi: Spinor) -> Phasor<(), f64> {
        (psi * psi.reverse()).cast::<Phasor>()
    }

    /// The spinor scaled to unit density, `|⟨ψ ψ̃⟩| = 1`, keeping its angle `β`: divided by its
    /// norm (gax's `normalized` would also remove `β`, making `ψ ψ̃ = 1`).
    pub fn unit_density(psi: Spinor) -> Spinor {
        psi * psi.norm().recip()
    }

    // Used by the tests (part of numga's core).
    #[cfg_attr(not(test), allow(dead_code))]
    /// What the spinor does to bivectors beyond what its frame does: its own sandwich on
    /// bivectors, composed with the inverse of the frame's extension to bivectors. It is
    /// multiplication by `exp(I β) / ρ`.
    pub fn duality(psi: Spinor) -> Bivector<(Bivector,), f64> {
        (psi >> Bivector::slot()).of(frame(psi).outermorphism::<Bivector>().inverse())
    }

    /// The electron's current: the frame's image of the time axis.
    pub fn current(psi: Spinor) -> V {
        frame(psi).of(time())
    }

    /// The electron's spin, in units of half the reduced Planck constant: the frame's image of
    /// the `z` axis.
    pub fn spin(psi: Spinor) -> V {
        frame(psi).of(z_axis())
    }

    /// The velocity a current carries, as the observer sees it: its relative vector (the part
    /// across the time axis) over its density (the part along it).
    pub fn velocity(flow: V) -> B {
        (flow ^ time()) * (1.0 / (flow | time()).s())
    }

    /// A relative vector (a bivector `eₖ e0` across the observer's time axis) as a vector of
    /// the observer's space: the inverse of gax's spacetime split, `sta::Bivector::from` a
    /// `vga3d::Vector`, on its image.
    pub fn relative(b: B) -> Space {
        Space::new(b.e10(), b.e20(), b.e30())
    }

    /// The part of a spacetime vector across the observer's time axis, in the observer's space.
    pub fn spatial(v: V) -> Space {
        relative(v ^ time())
    }

    /// The Dirac equation for a plane wave of the given spatial momentum, as a map on spinors:
    /// the momentum as a relative vector on the left, and the mass through the reflection in
    /// the time axis.
    pub fn hamiltonian(momentum: V, mass: f64) -> Hamiltonian {
        let t = time();
        (momentum * t) * Even::slot() + (t * Even::slot() * t) * mass
    }

    /// The Hermitian product of spinors, `⟨(t ψ̃ t) φ⟩`: positive definite, the identity in
    /// the blade basis. The Hamiltonian is symmetric under it.
    pub fn hermitian() -> gax::sta::Scalar<(Even, Even), f64> {
        form(Even::slot())
    }

    /// The energy of the plane wave, `√(m² + p²)`: a spatial vector squares to minus its
    /// length squared in this signature.
    #[allow(clippy::disallowed_methods)] // the energy-momentum relation: a physical law
    pub fn energy(momentum: V, mass: f64) -> f64 {
        (mass * mass - momentum.norm_squared()).sqrt()
    }

    /// The plane wave's spinor at the given times. The projector onto positive energy,
    /// `(ψ + H(ψ) / E) / 2`, splits it into its positive- and negative-energy parts, which turn
    /// in the spin plane at the energy, in opposite senses.
    pub fn evolve(momentum: V, mass: f64, psi: Spinor, times: &[f64]) -> Vec<Spinor> {
        let e = energy(momentum, mass);
        let positive = (psi + hamiltonian(momentum, mass).of(psi) * (1.0 / e)) * 0.5;
        times
            .iter()
            .map(|t| {
                let turn = (spin_plane() * (e * t)).exp().into_inner();
                positive * turn.reverse() + (psi - positive) * turn
            })
            .collect()
    }

    /// A displacement in the observer's space as a direction of its points.
    pub fn along(d: Space) -> Place {
        Place::direction(d.e1(), d.e2(), d.e3())
    }

    /// The points a velocity carries a point through, from the origin, one step at a time:
    /// each step the relative vector of the velocity times the step.
    pub fn path(velocities: &[B], dt: f64) -> Vec<Place> {
        let mut at = Place::xyz(0.0, 0.0, 0.0);
        let mut out = vec![at];
        for v in &velocities[..velocities.len() - 1] {
            at += along(relative(*v * dt));
            out.push(at);
        }
        out
    }

    // --- the scenes -------------------------------------------------------------------

    pub const MASS: f64 = 1.0;
    /// The share of negative energy in each electron.
    pub const MIXTURES: [f64; 3] = [0.1, 0.25, 0.5];

    /// A spatial momentum.
    pub fn momentum(x: f64, y: f64, z: f64) -> V {
        Vector::new(0.0, x, y, z)
    }

    /// The Hamiltonian's energies, ascending, and its eigenstates, over a grid of momenta in
    /// the `xy` plane: `(momentum, energies, states)` row by row.
    pub fn mass_shell(extent: f64, count: usize) -> Vec<(V, [f64; 8], [Spinor; 8])> {
        let along = |i: usize| -extent + 2.0 * extent * i as f64 / (count - 1) as f64;
        (0..count * count)
            .map(|k| {
                let p = momentum(along(k % count), along(k / count), 0.0);
                let (values, states) = form(hamiltonian(p, MASS)).eigh_with(hermitian());
                (p, values, states)
            })
            .collect()
    }

    /// The form `⟨(t ψ̃ t) H(φ)⟩` of a map on spinors: symmetric when the map is Hermitian.
    pub fn form(h: Hamiltonian) -> gax::sta::Scalar<(Even, Even), f64> {
        let t = time();
        (t * Even::slot().reverse() * t).scalar_product(h)
    }

    /// Electrons of the given momentum with a share of negative energy mixed in, followed in
    /// time: the times, their spinors, and the paths their currents trace.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::disallowed_methods)] // a superposition's amplitudes √p from its probabilities
    pub fn trembling(
        momentum: V,
        seconds: f64,
        count: usize,
    ) -> (Vec<f64>, Vec<Vec<Spinor>>, Vec<Vec<Place>>) {
        let h = hamiltonian(momentum, MASS);
        let e = energy(momentum, MASS);
        // A positive-energy state with spin along z, and a negative-energy state that turns the
        // other way (from the plane `t ∧ x`): the projections onto positive and negative
        // energy, `(1 ± H / E) / 2`.
        let tx: Spinor = (time() ^ x_axis()).into();
        let electron = unit_density((scalar(1.0) + h.of(scalar(1.0)) * (1.0 / e)) * 0.5);
        let positron = unit_density((tx - h.of(tx) * (1.0 / e)) * 0.5);
        let times: Vec<f64> = (0..count)
            .map(|i| seconds * i as f64 / (count - 1) as f64)
            .collect();
        let dt = times[1] - times[0];
        let mut spinors = Vec::new();
        let mut paths = Vec::new();
        for share in MIXTURES {
            let psi = evolve(
                momentum,
                MASS,
                electron * (1.0 - share).sqrt() + positron * share.sqrt(),
                &times,
            );
            let velocities: Vec<B> = psi.iter().map(|p| velocity(current(*p))).collect();
            paths.push(path(&velocities, dt));
            spinors.push(psi);
        }
        (times, spinors, paths)
    }
}

use dirac::*;

const SECONDS: f32 = 10.0;
/// The time the trembling paths are followed for.
const DURATION: f64 = 12.0;
const SAMPLES: usize = 600;

/// What the frames share: the mass shell's grid and sheets (momentum across, energy up), and
/// the trembling paths with the spins along them, in the observer's space.
struct Scene {
    shell: (usize, Vec<Place>, Vec<Place>),
    paths: Vec<Vec<Place>>,
    spins: Vec<Vec<Space>>,
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let n = 21;
        let shell = mass_shell(2.0, n);
        // Each momentum (in the xy plane, from the origin) raised by its `k`-th energy along z.
        let origin = Place::xyz(0.0, 0.0, 0.0);
        let sheet = |k: usize| -> Vec<Place> {
            shell
                .iter()
                .map(|(p, e, _)| origin + along(spatial(*p) + Space::new(0.0, 0.0, e[k])))
                .collect()
        };
        let (_, spinors, paths) = trembling(momentum(0.0, 0.0, 0.3), DURATION, SAMPLES);
        Scene {
            shell: (n, sheet(7), sheet(0)),
            paths,
            spins: spinors
                .iter()
                .map(|s| s.iter().map(|p| spatial(spin(*p))).collect())
                .collect(),
        }
    })
}

fn colours() -> [Light; 3] {
    [
        palette::sky(),
        palette::purple(),
        Light::from_srgb(0.95, 0.42, 0.33, 1.8),
    ]
}

/// A parallel camera drawing into `view`, `half` world units from its middle to its top.
fn camera(view: Rect, target: Point3, azimuth: f32, elevation: f32, half: f32) -> Camera {
    Camera::orbit(view, target, 20.0, azimuth, elevation, Lens::Parallel(half))
}

/// The rectangle `height` pixels tall (and as wide as `rect`) centred `drop` pixels below the
/// centre of `rect`.
fn view_of(rect: Rect, drop: f32, height: f32) -> Rect {
    let centre = rect.centre() + Point2::direction(0.0, drop);
    let half = Point2::direction(rect.width() * 0.5, height * 0.5);
    Rect {
        lo: centre - half,
        hi: centre + half,
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scene();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let size = (h / 36.0).clamp(7.0, 14.0);
    let phase = t / SECONDS;
    let colours = colours();
    let down = Point2::direction(0.0, 1.0);
    let (left, right) = (
        Rect::new(0.0, 0.0, w * 0.6, h),
        Rect::new(w * 0.6, 0.0, w, h),
    );

    // The trembling paths, traced as time goes on, with each electron's spin where it is.
    let upto = ((phase * SAMPLES as f32) as usize).clamp(2, SAMPLES);
    let azimuth = -1.05 + 0.5 * wave(phase * core::f32::consts::TAU);
    let view = view_of(left, h * 0.04, h * 1.08);
    let cam = camera(view, Point3::xyz(0.0, 0.0, 1.7), azimuth, 0.2, 2.0);
    let mut scene3 = Scene3::new(cam);
    // A floor grid and the z axis.
    let g = palette::grid().faded(0.7);
    for k in 0..=6 {
        let a = -0.6 + 0.2 * k as f32;
        let (x0, x1) = (Point3::xyz(a, -0.6, 0.0), Point3::xyz(a, 0.6, 0.0));
        let (y0, y1) = (Point3::xyz(-0.6, a, 0.0), Point3::xyz(0.6, a, 0.0));
        scene3.seg(x0, x1, 1.0, g);
        scene3.seg(y0, y1, 1.0, g);
    }
    let up = Point3::direction(0.0, 0.0, 1.0);
    let axis_light = palette::grid().faded(0.9);
    scene3.seg(ORIGIN3, ORIGIN3 + up.gp(3.4), 1.0, axis_light);
    for (k, (path, spins)) in s.paths.iter().zip(&s.spins).enumerate() {
        scene3.polyline(&path[..upto], 1.6, colours[k]);
        let here = path[upto - 1];
        let axis = spins[upto - 1].normalized().into_inner() * 0.4;
        scene3.arrow(here, axis, 2.0, 8.0, colours[k]);
        scene3.dot(here, Marker::Dot, 6.0, colours[k]);
    }
    c.clip(left);
    scene3.draw(c);
    c.unclip();
    let at = f64::from(phase) * DURATION;
    let key = left.lo + Point2::direction(w * 0.04, h * 0.2);
    // The clock under the key, clear of the paths below.
    let clock = key + down.gp(MIXTURES.len() as f32 * size * 1.6 + size * 0.6);
    let text = format!("T = {at:4.1} H/MC2");
    c.text(&text, clock, size, palette::ink(), Align::Left);
    for (k, share) in MIXTURES.iter().enumerate() {
        c.text(
            &format!("{:.0}% NEGATIVE ENERGY", share * 100.0),
            key + down.gp(k as f32 * size * 1.6),
            size * 0.85,
            colours[k],
            Align::Left,
        );
    }

    // The mass shell: the two sheets of energies over the momentum plane, turning.
    let (n, top, bottom) = &s.shell;
    let azimuth = -0.9 + phase * core::f32::consts::TAU;
    let view = view_of(right, h * 0.06, h * 1.12);
    let cam = camera(view, ORIGIN3, azimuth, 0.3, 4.6);
    let mut scene3 = Scene3::new(cam);
    for (sheet, colour) in [(top, palette::red()), (bottom, palette::sky())] {
        let m = *n - 1;
        let at = |u: f32, v: f32| {
            let (i, j) = (
                (u * m as f32).round() as usize,
                (v * m as f32).round() as usize,
            );
            sheet[j * n + i]
        };
        let edges = Some((colour.faded(0.6), 0.6));
        scene3.surface(at, m, m, |_, _| colour, 0.6, edges);
    }
    let reach = up.gp(3.2);
    scene3.seg(
        ORIGIN3 - reach,
        ORIGIN3 + reach,
        1.0,
        palette::ink().faded(0.6),
    );
    c.clip(right);
    scene3.draw(c);
    c.unclip();
    let (top_middle, bottom_middle) = (right.top_middle(), right.bottom_middle());
    // The title on two lines, to fit the panel.
    let title = top_middle + down.gp(h * 0.17);
    for (k, line) in ["THE MASS SHELL", "E = +-SQRT(P2 + M2)"]
        .into_iter()
        .enumerate()
    {
        c.text(
            line,
            title + down.gp(k as f32 * size * 1.4),
            size * 0.85,
            palette::ink(),
            Align::Center,
        );
    }
    c.text(
        "POSITIVE ENERGY",
        title + down.gp(size * 3.0),
        size * 0.8,
        palette::red(),
        Align::Center,
    );
    c.text(
        "NEGATIVE ENERGY, GAP 2 MC2",
        bottom_middle - down.gp(h * 0.05),
        size * 0.8,
        palette::sky(),
        Align::Center,
    );
    caption(
        c,
        "THE DIRAC ELECTRON: ZITTERBEWEGUNG",
        "STA SPINORS: THE CURRENT CIRCLES AT TWICE THE ENERGY, SPIN ALONG Z",
    );
}

fn main() {
    run(Anim::new("dirac", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::dirac::*;
    use gax::ApproxEq;
    use gax::sta::{Bivector, Even, Vector};
    use gax::vga3d;

    use gax_numga_examples::rng::{Draw, rng};

    /// `ψ >> v == ρ (R >> v)` for vectors; on bivectors the spinor's sandwich is its frame's
    /// extension times `exp(I β) / ρ`.
    #[test]
    fn a_spinor_is_rho_times_a_lorentz_map_and_beta_acts_only_on_bivectors() {
        // numga's `(tx * 0.4 + xy * 0.9 + yz * -0.3) * 0.5`, with `tx = -e10`.
        let rotor = Bivector::new(-0.2, 0.0, 0.0, -0.15, 0.0, 0.45)
            .exp()
            .into_inner();
        let psi = spinor(2.0, 0.7, rotor);
        let v = Vector::new(0.3, -1.2, 0.5, 2.0);
        assert!(frame(psi).of(v).max_abs_diff(&((rotor >> v) * 2.0)) < 1e-12);
        // A phasor times a bivector is a bivector: the duality rotation stays in the kind.
        let b = Bivector::new(0.2, -0.4, 1.1, 0.3, -0.7, 0.5);
        let expected: Bivector<(), f64> = phase(0.7) * b * 0.5;
        assert!(duality(psi).of(b).max_abs_diff(&expected) < 1e-9);
    }

    /// The spacetime split: a vector of space as a relative vector and back.
    #[test]
    fn relative_vectors_are_the_observer_s_space() {
        let s = vga3d::Vector::new(0.3, -1.2, 0.5);
        assert_eq!(relative(Bivector::from(s)), s);
        // A spatial vector's part across the time axis is itself.
        let v = Vector::new(0.7, 0.3, -1.2, 0.5);
        assert!(spatial(v).max_abs_diff(&s) < 1e-15);
    }

    /// Each momentum has the energies `-E` and `+E`, each fourfold; the positive-energy states
    /// have `β = 0` and the negative-energy states `β = π`, where `ψ ψ̃` has a negative scalar
    /// part.
    #[test]
    fn mass_shell_energies_are_fourfold_and_beta_tells_their_sign() {
        for (p, values, states) in mass_shell(1.0, 5) {
            let e = energy(p, MASS);
            for (k, value) in values.iter().enumerate() {
                let expected = if k < 4 { -e } else { e };
                assert!((value - expected).abs() < 1e-12, "{values:?} vs {e}");
                let sign = invariants(states[k]).s().signum();
                assert_eq!(sign, if k < 4 { -1.0 } else { 1.0 });
            }
        }
    }

    /// The density the current carries is constant in time, and the Hamiltonian commutes with
    /// right multiplication by the spin plane.
    #[test]
    fn trembling_density_is_constant_and_the_spin_plane_commutes() {
        let p = momentum(0.0, 0.0, 0.3);
        let (_, spinors, paths) = trembling(p, 3.0, 60);
        for psi in &spinors {
            let first = (current(psi[0]) | time()).s();
            for s in psi {
                let density = (current(*s) | time()).s();
                assert!((density - first).abs() <= 1e-10 * first.abs());
            }
        }
        assert_eq!(paths.len(), 3);
        let mut rng = rng(1);
        let probe = Even::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.normal()));
        let h = hamiltonian(p, MASS);
        let unit = Even::slot() * spin_plane();
        assert!(h.of(unit.of(probe)).max_abs_diff(&unit.of(h.of(probe))) < 1e-12);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
