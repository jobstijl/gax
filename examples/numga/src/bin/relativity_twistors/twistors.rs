//! numga's `twistors/core.py`: the conformal algebra of spacetime, twistors as its spinors,
//! the light rays they carry, the Robinson congruence and the Hopfion it carries.

// numga's API in full: the tests use parts the animation does not.
#![cfg_attr(not(test), allow(dead_code))]

use std::sync::OnceLock;

gax::algebra! {
    algebra cl42 "Cl(4,2): spacetime's x y z t (t squaring to minus one) and the conformal pair u (plus one) and v (minus one).";
    basis ex = 1, ey = 1, ez = 1, et = -1, eu = 1, ev = -1;
    kind Scalar = [1];
    kind Vector = [ex, ey, ez, et, eu, ev];
    kind Bivector = [exy, exz, eyz, ext, eyt, ezt, exu, eyu, ezu, etu, exv, eyv, ezv, etv, euv];
    kind Even = [1, exy, exz, eyz, ext, eyt, ezt, exu, eyu, ezu, etu, exv, eyv, ezv, etv, euv, exyzt, exyzu, exyzv, exytu, exytv, exyuv, exztu, exztv, exzuv, extuv, eyztu, eyztv, eyzuv, eytuv, eztuv, exyztuv];
    kind Pseudoscalar = [exyztuv];
    kind Event = [ex, ey, ez, et];
    kind Faraday = [exy, exz, eyz, ext, eyt, ezt];
    kind Phasor = [1, exyzt];
    kind SpatialVolume = [exyz];
    kind Twistor = [1, ey, ez, eu, eyz, eyu, ezu, eyzu];
}

/// A conformal vector.
pub type V = cl42::Vector<(), f64>;
/// An event of spacetime.
pub type Event = cl42::Event<(), f64>;
pub type B = cl42::Bivector<(), f64>;
/// An even multivector: the conformal rotors. The algebra declares no versors (their sandwich
/// kernels would slow its compilation), so rotors act by explicit sandwiches.
pub type Even = cl42::Even<(), f64>;
/// A general multivector.
pub type Full = cl42::Multivector<(), f64>;
/// An electromagnetic field: a bivector of spacetime.
pub type Faraday = cl42::Faraday<(), f64>;
pub type Twistor = cl42::Twistor<(), f64>;
/// A linear map on twistors.
pub type TwistorMap = cl42::Twistor<(cl42::Twistor,), f64>;
/// A bilinear form on twistors.
pub type Pairing = cl42::Scalar<(cl42::Twistor, cl42::Twistor), f64>;
/// The null plane of a ray, from twistors.
pub type RayMap = cl42::Bivector<(cl42::Twistor, cl42::Twistor), f64>;
/// A point of space at one instant, in VGA3D (whose `x y z` are the algebra's).
pub type Spatial = gax::vga3d::Vector<(), f64>;

/// The event at `(x, y, z, t)`.
pub fn event(x: f64, y: f64, z: f64, t: f64) -> Event {
    Event::new(x, y, z, t)
}

/// The observer's time axis.
pub fn t() -> Event {
    event(0.0, 0.0, 0.0, 1.0)
}

pub fn x() -> Event {
    event(1.0, 0.0, 0.0, 0.0)
}

pub fn y() -> Event {
    event(0.0, 1.0, 0.0, 0.0)
}

pub fn z() -> Event {
    event(0.0, 0.0, 1.0, 0.0)
}

/// The conformal origin, `(v - u) / 2`.
pub fn origin() -> V {
    V::new(0.0, 0.0, 0.0, 0.0, -0.5, 0.5)
}

/// The point at infinity, `v + u`.
pub fn infinity() -> V {
    V::new(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
}

/// The spacetime volume `x y z t`, which squares to minus one.
pub fn spacetime_volume() -> cl42::Phasor<(), f64> {
    cl42::Phasor::new(0.0, 1.0)
}

/// An event of space at an instant.
pub fn at(position: Spatial, time: f64) -> Event {
    event(position.e1(), position.e2(), position.e3(), time)
}

/// The part of an event across the time axis, as a point of space.
pub fn spatial(e: Event) -> Spatial {
    Spatial::new(e.ex(), e.ey(), e.ez())
}

/// An event as a null conformal vector of weight one.
pub fn point(e: Event) -> V {
    origin() + V::from(e) + infinity() * (e.norm_squared() / 2.0)
}

/// The event a conformal vector stands for: its spacetime part over its weight, minus its
/// pairing with infinity.
pub fn event_of(p: V) -> Event {
    p.cast::<cl42::Event>() * (-1.0 / (infinity() | p).s())
}

/// The `k`th of `N` unit coefficients: a basis element's coefficients.
fn unit_coefficients<const N: usize>(k: usize) -> [f64; N] {
    core::array::from_fn(|i| if i == k { 1.0 } else { 0.0 })
}

/// The basis twistors, in coefficient order.
pub fn twistor_basis() -> [Twistor; 8] {
    core::array::from_fn(|k| Twistor::from_coeffs(unit_coefficients(k)))
}

/// The idempotent `(1 + x)(1 + y t)(1 + z v) / 8` whose left ideal is the twistors.
pub fn projector() -> Full {
    static PROJECTOR: OnceLock<Full> = OnceLock::new();
    *PROJECTOR.get_or_init(|| {
        let v = V::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let factor = |b: Full| Full::from(cl42::Scalar::new(1.0)) + b;
        factor(Full::from(x()))
            * factor(Full::from(y() * t()))
            * factor(Full::from(z()) * Full::from(v))
            * 0.125
    })
}

/// A twistor as the multivector of the left ideal it stands for, `ψ P`.
pub fn embed(psi: Twistor) -> Full {
    psi * projector()
}

/// The twistor a multivector of the left ideal stands for.
pub fn readout(m: Full) -> Twistor {
    m.cast::<cl42::Twistor>() * 8.0
}

/// A multivector's action on a twistor, multiplication from the left.
pub fn act(a: impl Into<Full>, psi: Twistor) -> Twistor {
    readout(a.into() * embed(psi))
}

/// A multivector's action on twistors, as a map.
pub fn representation(a: impl Into<Full>) -> TwistorMap {
    let a = a.into();
    TwistorMap::from_images(twistor_basis().map(|b| act(a, b)))
}

/// The conformal volume's action: a complex structure on twistors.
pub fn volume() -> TwistorMap {
    representation(cl42::Pseudoscalar::new(1.0))
}

/// The pairing `8 ⟨y z (ψ P)~ I (φ P)⟩` that conformal rotors keep.
pub fn pairing() -> Pairing {
    static PAIRING: OnceLock<Pairing> = OnceLock::new();
    *PAIRING.get_or_init(|| {
        let yz = Full::from(y() ^ z());
        let volume = Full::from(cl42::Pseudoscalar::new(1.0));
        let pair = |a: Twistor, b: Twistor| (yz * embed(a).reverse() * volume * embed(b)).s() * 8.0;
        let basis = twistor_basis();
        Pairing::from_coeffs([core::array::from_fn(|i| {
            core::array::from_fn(|j| pair(basis[i], basis[j]))
        })])
    })
}

/// The pairing of two twistors.
pub fn pair(psi: Twistor, phi: Twistor) -> f64 {
    pairing().of(psi).of(phi).s()
}

/// The basis bivectors, in coefficient order.
fn plane_basis() -> [B; 15] {
    core::array::from_fn(|k| B::from_coeffs(unit_coefficients(k)))
}

/// The ray map: the bivector whose scalar product with every basis plane `b` is the pairing
/// of the first twistor with the volume times `b`'s action on the second. The basis planes
/// are orthogonal, so the bivector is the sum of their inverses weighted by those pairings.
pub fn ray_map() -> RayMap {
    static RAY: OnceLock<RayMap> = OnceLock::new();
    *RAY.get_or_init(|| {
        plane_basis()
            .into_iter()
            .fold(RayMap::zero(), |sum, plane| {
                let turned = volume().of(representation(plane));
                // The pairing with its second twistor through the plane's turned action.
                let weight = pairing().at::<1>().of(turned).swap();
                // A basis plane's inverse is the plane itself, up to sign.
                sum + plane.inverse().cast::<cl42::Bivector>() * weight
            })
    })
}

/// The null plane of the ray of a twistor, or the bilinear plane of two.
pub fn ray(psi: Twistor, phi: Twistor) -> B {
    ray_map().of(psi).of(phi)
}

/// A twistor of the light ray through two distinct null-separated conformal points: the
/// leading image of their product's action. Each point acts with its square, zero, so its
/// image lies in its kernel; null-separated points anticommute, so the image of their
/// product lies in both kernels.
pub fn through(first: V, second: V) -> Twistor {
    let (_, _, images) = representation(first * second).svd_thin();
    images[0]
}

/// The event where a conformal light ray meets the slice of constant time `time`.
pub fn at_time(plane: B, time: f64) -> Event {
    event_of((V::from(t()) - infinity() * time) | plane)
}

/// The future null direction of a twistor's ray, its time component one.
pub fn direction(psi: Twistor) -> Event {
    let tangent = (infinity() | ray(psi, psi)).cast::<cl42::Event>();
    tangent * (-1.0 / (t() | tangent).s())
}

/// The fixed twistor `1 + y - z u + y z u` that is not null.
pub fn robinson_twistor() -> Twistor {
    // Blades [1, y, z, u, yz, yu, zu, yzu].
    Twistor::from_coeffs([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
}

/// The light ray the Robinson twistor selects at an event: the ray of its image under the
/// event's action.
pub fn robinson(e: Event) -> Event {
    direction(act(point(e), robinson_twistor()))
}

/// The sandwich `r x r̃`, cast back to the kind of `x`.
pub fn moved_vector(r: Even, x: V) -> V {
    (r * x * r.reverse()).cast::<cl42::Vector>()
}

/// The sandwich of a bivector.
pub fn moved_plane(r: Even, b: B) -> B {
    (r * b * r.reverse()).cast::<cl42::Bivector>()
}

/// A source-free null electromagnetic field carried by the Robinson congruence, at an event.
pub fn hopfion(e: Event) -> Faraday {
    let position = (-(e ^ t()) * t()).cast::<cl42::Event>();
    let time = -(e | t()).s();
    let spatial_volume = cl42::SpatialVolume::new(1.0);
    // The spinor turns the polarization and its propagation direction together.
    let spinor = Even::from(cl42::Scalar::new(1.0)) + spatial_volume * (position + y() * time);
    let polarization = (t() + y()) ^ x();
    // The spacetime volume rotates electric into magnetic polarization. Its inverse supplies
    // both the decaying amplitude and the local polarization angle.
    let weight = (cl42::Phasor::new(1.0 + e.norm_squared(), 0.0)
        + spacetime_volume() * (2.0 * (e | t()).s()))
    .inverse();
    let turned = (spinor * polarization * spinor.reverse()).cast::<cl42::Faraday>();
    turned * (weight * weight * weight)
}

/// The electric field an observer at rest sees in a field.
pub fn electric(field: Faraday) -> Event {
    (field | t()).cast::<cl42::Event>()
}

/// The magnetic field an observer at rest sees in a field.
pub fn magnetic(field: Faraday) -> Event {
    ((spacetime_volume() * field) | t()).cast::<cl42::Event>()
}

/// The energy flow of a field an observer at rest sees, as its direction in spacetime: the
/// time axis plus the Poynting vector over the energy density. For a null field it is null.
pub fn energy_flow(field: Faraday) -> Event {
    let (electric, magnetic) = (spatial(electric(field)), spatial(magnetic(field)));
    let energy = (electric.norm_squared() + magnetic.norm_squared()) / 2.0;
    let poynting = -(gax::vga3d::Pseudoscalar::new(1.0) * (electric ^ magnetic));
    at(poynting * (1.0 / energy), 1.0)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use gax::ApproxEq;
    use gax::vga3d;
    use gax_numga_examples::rng::{Draw, Rng, rng};
    use gax_numga_examples::signal::phasor;

    pub const TOLERANCE: f64 = 1e-11;
    const SAMPLES: usize = 24;

    /// An event with normal coordinates of the given spread.
    fn random_event(rng: &mut Rng, spread: f64) -> Event {
        event(rng.normal(), rng.normal(), rng.normal(), rng.normal()) * spread
    }

    /// A twistor with standard normal coefficients.
    fn random_twistor(rng: &mut Rng) -> Twistor {
        Twistor::from_coeffs(core::array::from_fn(|_| rng.normal()))
    }

    /// The rotor `exp(b angle)`.
    fn turn(b: B, angle: f64) -> Even {
        (b * angle).exp().into_inner()
    }

    /// A random rotation, boost, translation and special conformal transformation, composed.
    fn conformal_rotor(rng: &mut Rng) -> Even {
        let (translation, special) = (random_event(rng, 0.2), random_event(rng, 0.2));
        turn(B::from(x() ^ y()), rng.normal() * 0.2)
            * turn(B::from(x() ^ t()), rng.normal() * 0.2)
            * turn(V::from(translation) ^ infinity(), -0.5)
            * turn(V::from(special) ^ origin(), -0.5)
    }

    /// Whether a value is zero to the tolerance.
    pub fn vanishes<M: ApproxEq>(m: M, zero: M, tolerance: f64) -> bool {
        m.max_abs_diff(&zero) < tolerance
    }

    #[test]
    fn clifford_action_and_hermitian_pairing_follow_conformal_rotors() {
        let mut rng = rng(821);
        let full = |rng: &mut Rng| Full::from_coeffs(core::array::from_fn(|_| rng.normal() * 0.2));
        for _ in 0..SAMPLES {
            let (first, second) = (full(&mut rng), full(&mut rng));
            let (state, partner) = (random_twistor(&mut rng), random_twistor(&mut rng));
            let rotor = conformal_rotor(&mut rng);
            let (moved, moved_partner) = (act(rotor, state), act(rotor, partner));
            // Clifford multiplication acts by composition; even actions commute with the
            // volume and preserve both pairings.
            let composition = representation(first).of(representation(second));
            let product = representation(first * second);
            assert!(composition.max_abs_diff(&product) < TOLERANCE);
            let volume = volume();
            assert!(vanishes(
                volume.of(volume.of(state)) + state,
                Twistor::zero(),
                TOLERANCE
            ));
            assert!(volume.of(moved).max_abs_diff(&act(rotor, volume.of(state))) < TOLERANCE);
            assert!((pair(moved, moved_partner) - pair(state, partner)).abs() < TOLERANCE);
            let turned = pair(volume.of(moved), moved_partner) - pair(volume.of(state), partner);
            assert!(turned.abs() < TOLERANCE);
        }
    }

    #[test]
    fn shared_twistor_recovers_a_null_ray_and_transforms_with_its_events() {
        let mut rng = rng(822);
        for _ in 0..SAMPLES {
            let e = random_event(&mut rng, 1.0);
            let heading = t() + at(rng.direction::<vga3d::Vector<(), f64>>(), 0.0);
            let separation = rng.range(0.2, 1.5);
            let (first, second) = (point(e), point(e + heading * separation));
            let twistor = through(first, second);
            let plane = ray(twistor, twistor);
            let angle = phasor(rng.normal());
            let phase = Full::from(cl42::Scalar::new(angle.e20()))
                + Full::from(cl42::Pseudoscalar::new(angle.e01()));
            let rotor = conformal_rotor(&mut rng);
            let moved = act(rotor, twistor);
            // The two events have a shared null twistor. Its bilinear ray is the null plane
            // through them, and turning it by the volume leaves its direction alone.
            assert!(first.norm_squared().abs() < TOLERANCE);
            assert!(second.norm_squared().abs() < TOLERANCE);
            assert!(vanishes(act(first, twistor), Twistor::zero(), TOLERANCE));
            assert!(vanishes(act(second, twistor), Twistor::zero(), TOLERANCE));
            assert!(pair(twistor, twistor).abs() < TOLERANCE);
            assert!(vanishes(Full::from(first ^ plane), Full::zero(), TOLERANCE));
            assert!(vanishes(
                Full::from(second ^ plane),
                Full::zero(),
                TOLERANCE
            ));
            assert!(vanishes(plane * plane, Even::zero(), TOLERANCE));
            assert!(direction(twistor).max_abs_diff(&heading) < TOLERANCE);
            assert!(direction(act(phase, twistor)).max_abs_diff(&heading) < TOLERANCE);
            assert!(vanishes(
                act(moved_vector(rotor, first), moved),
                Twistor::zero(),
                TOLERANCE
            ));
            assert!(vanishes(
                act(moved_vector(rotor, second), moved),
                Twistor::zero(),
                TOLERANCE
            ));
            assert!(ray(moved, moved).max_abs_diff(&moved_plane(rotor, plane)) < TOLERANCE);
        }
    }

    #[test]
    fn robinson_twistors_stay_orthogonal_along_straight_future_null_rays() {
        let mut rng = rng(823);
        let fixed = robinson_twistor();
        for _ in 0..SAMPLES {
            let e = random_event(&mut rng, 1.0);
            let twistor = act(point(e), fixed);
            let heading = robinson(e);
            // The event's selected twistor pairs to zero with the fixed twistor and with its
            // turn by the volume. Every event along its straight light ray selects the same
            // propagation direction and remains incident with the initial twistor.
            assert!(pair(fixed, twistor).abs() < TOLERANCE);
            assert!(pair(volume().of(fixed), twistor).abs() < TOLERANCE);
            assert!(pair(twistor, twistor).abs() < TOLERANCE);
            assert!(heading.norm_squared().abs() < TOLERANCE);
            assert!(((t() | heading).s() + 1.0).abs() < TOLERANCE);
            for k in 0..9 {
                let along = e + heading * (-1.0 + 0.25 * k as f64);
                assert!(robinson(along).max_abs_diff(&heading) < TOLERANCE);
                assert!(vanishes(
                    act(point(along), twistor),
                    Twistor::zero(),
                    TOLERANCE
                ));
            }
        }
    }

    /// Events with coordinates uniform in `[-2, 2]`, numga's samples of the Hopfion.
    fn hopfion_events(rng: &mut Rng) -> Vec<Event> {
        (0..128)
            .map(|_| {
                let mut coordinate = || rng.range(-2.0, 2.0);
                event(coordinate(), coordinate(), coordinate(), coordinate())
            })
            .collect()
    }

    #[test]
    fn hopfion_is_null_and_its_energy_follows_the_robinson_congruence() {
        let mut rng = rng(41);
        for e in hopfion_events(&mut rng) {
            let field = hopfion(e);
            let heading = robinson(e);
            assert!(vanishes(field * field, Even::zero(), 1e-12));
            assert!(energy_flow(field).max_abs_diff(&heading) < 1e-12);
            // The field's planes all contain the ray.
            assert!(vanishes(Full::from(field | heading), Full::zero(), 1e-12));
        }
    }

    /// The field's spacetime derivative `∇F = Σ eⁱ ∂ᵢF` vanishes: both of Maxwell's equations
    /// without sources. The partial derivatives are fourth-order centred differences.
    #[test]
    fn hopfion_satisfies_vacuum_maxwell_equations() {
        let mut rng = rng(73);
        let step = 1e-4;
        for e in hopfion_events(&mut rng) {
            let maxwell = [x(), y(), z(), t()]
                .into_iter()
                .fold(Full::zero(), |sum, axis| {
                    let at = |k: f64| hopfion(e + axis * (k * step));
                    let partial =
                        (at(-2.0) - at(2.0) + (at(1.0) - at(-1.0)) * 8.0) * (1.0 / (12.0 * step));
                    let reciprocal = axis * (1.0 / axis.norm_squared());
                    sum + Full::from(reciprocal) * Full::from(partial)
                });
            // numga's tolerance: the differences' round-off, about 1e-12, with room.
            assert!(vanishes(maxwell, Full::zero(), 1e-9));
        }
    }

    /// The Robinson direction and the Hopfion at one event, against numga's numbers.
    #[test]
    fn robinson_and_hopfion_match_numga() {
        let e = event(0.3, -0.7, 1.1, 0.4);
        let heading = event(0.84518828, -0.08786611, -0.52719665, 1.0);
        assert!(robinson(e).max_abs_diff(&heading) < 1e-8);
        let field = Faraday::new(
            -0.06405577,
            -0.09055565,
            -0.03054138,
            0.05336897,
            0.0702405,
            0.07385303,
        );
        assert!(hopfion(e).max_abs_diff(&field) < 1e-8);
    }
}
