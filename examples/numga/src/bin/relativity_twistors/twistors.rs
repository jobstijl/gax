//! numga's `twistors/core.py`: the conformal algebra of spacetime, twistors as its spinors, the
//! light rays they carry, the Robinson congruence and the Hopfion it carries.
//!
//! Points, events and light rays are intrinsic, numga's antivectors and antibivectors (ADR-051):
//! an event is the Hodge dual of a conformal vector, and a ray the join of its events. An event
//! acts on twistors by multiplication from the left like every multivector, and being intrinsic
//! it turns a twistor's orientation: a ray's twistor is intrinsic, the twistor an event selects
//! it from extrinsic, and the pairing runs between the two.

// numga's API in full: the tests use parts the animation does not.
#![cfg_attr(not(test), allow(dead_code))]

use std::sync::OnceLock;

gax::algebra! {
    algebra cl42 "Cl(4,2): spacetime's x y z t (t squaring to minus one) and the conformal pair u (plus one) and v (minus one).";
    basis ex = 1, ey = 1, ez = 1, et = -1, eu = 1, ev = -1;
    kind Scalar = [1];
    kind Vector = [ex, ey, ez, et, eu, ev];
    kind Bivector = [exy, exz, eyz, ext, eyt, ezt, exu, eyu, ezu, etu, exv, eyv, ezv, etv, euv];
    kind Across = [etuv] "Time and the conformal pair: a patch of space joined across them is a point of space.";
    kind Even = [1, exy, exz, eyz, ext, eyt, ezt, exu, eyu, ezu, etu, exv, eyv, ezv, etv, euv, exyzt, exyzu, exyzv, exytu, exytv, exyuv, exztu, exztv, exzuv, extuv, eyztu, eyztv, eyzuv, eytuv, eztuv, exyztuv];
    kind Pseudoscalar = [exyztuv];
    kind Point = [eyztuv, ezxtuv, exytuv, eyxzuv, exyztv, exyztu] intrinsic "A conformal point, numga's antivector: the duals of x y z, minus the dual of t (one unit along time), and the duals of u and v.";
    kind Event = [eyztuv, ezxtuv, exytuv, eyxzuv] intrinsic "An event of spacetime, `x X + y Y + z Z + t T` with the duals `X Y Z` of x y z and `T` one unit along time.";
    kind Spatial = [eyztuv, ezxtuv, exytuv] intrinsic "A point of space at one instant.";
    kind Ray = [exyzt, exyzu, exytu, exztu, eyztu, exyzv, exytv, exztv, eyztv, exyuv, exzuv, eyzuv, extuv, eytuv, eztuv] intrinsic "A light ray, numga's antibivector: the join of its events.";
    kind Faraday = [exy, exz, eyz, ext, eyt, ezt];
    kind Phasor = [1, exyzt];
    kind Twistor = [1, ey, ez, eu, eyz, eyu, ezu, eyzu] "A twistor: extrinsic, a twistor an event selects a ray from; intrinsic, a ray's twistor.";
}

/// A conformal vector.
pub type V = cl42::Vector;
/// A conformal point, numga's antivector.
pub type Point = cl42::Point;
/// An event of spacetime.
pub type Event = cl42::Event;
/// A point of space at an instant.
pub type Spatial = cl42::Spatial;
/// A light ray, the join of its events.
pub type Ray = cl42::Ray;
pub type B = cl42::Bivector;
/// An even multivector: the conformal rotors. The algebra declares no versors (their sandwich
/// kernels would slow its compilation), so rotors act by explicit sandwiches.
pub type Even = cl42::Even;
/// A general multivector, and one of intrinsic orientation (events and their products).
pub type Full = cl42::Multivector;
pub type IntrinsicFull = cl42::intrinsic::Multivector;
/// An electromagnetic field: a bivector of spacetime.
pub type Faraday = cl42::Faraday;
/// A ray's twistor, intrinsic, and the twistors events select rays from, extrinsic. (The kind
/// is declared extrinsic: declared intrinsic, gax's generated `inverse` of a kind closed under
/// the product mixes the orientations and does not compile.)
pub type Twistor = cl42::intrinsic::Twistor;
pub type ExtrinsicTwistor = cl42::Twistor;
/// A linear map on twistors.
pub type TwistorMap = cl42::intrinsic::Twistor<(cl42::intrinsic::Twistor,)>;
/// The action of an intrinsic multivector: from ray twistors to extrinsic ones.
pub type IncidenceMap = cl42::Twistor<(cl42::intrinsic::Twistor,)>;
/// A bilinear form between twistors of the two orientations.
pub type Pairing = cl42::Scalar<(cl42::intrinsic::Twistor, cl42::Twistor)>;
/// A ray from pairs of twistors.
pub type RayMap = cl42::Ray<(cl42::intrinsic::Twistor, cl42::intrinsic::Twistor)>;

/// The conformal vectors `x y z t u v`.
pub fn x() -> V {
    V::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

pub fn y() -> V {
    V::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
}

pub fn z() -> V {
    V::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
}

pub fn t() -> V {
    V::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
}

pub fn u() -> V {
    V::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
}

pub fn v() -> V {
    V::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
}

/// The origin, `(v* - u*) / 2`, with `*` the Hodge dual (numga's `dual`).
pub fn origin() -> Point {
    (v().hodge() - u().hodge()) * 0.5
}

/// The point at infinity, `v* + u*`.
pub fn infinity() -> Point {
    v().hodge() + u().hodge()
}

/// The plane at infinity, `-(v + u)`, which reads a point's weight.
pub fn horizon() -> V {
    -(v() + u())
}

/// One unit along time, `-t*`.
pub fn time() -> Event {
    (-t().hodge()).cast::<cl42::Event>()
}

/// Time and the conformal pair, `t u v`: a patch of space joined across them is a point of space.
pub fn across() -> cl42::Across {
    cl42::Across::new(1.0)
}

/// The handedness of the pulse and of the fibres it is carried along: the intrinsic scalar one.
pub fn helicity() -> cl42::intrinsic::Scalar {
    cl42::intrinsic::Scalar::new(1.0)
}

/// `y` as a patch of space, `across⁻¹ ⌋ y*`: the plane the pulse and its fibres turn in.
pub fn symmetry() -> cl42::intrinsic::Bivector {
    (across().inverse() | y().hodge()).cast::<cl42::Bivector>()
}

/// The spacetime volume `x y z t`, which squares to minus one.
pub fn spacetime_volume() -> cl42::Phasor {
    cl42::Phasor::new(0.0, 1.0)
}

/// The event at `(x, y, z, t)`: `x X + y Y + z Z + t T`.
pub fn event(x: f64, y: f64, z: f64, t: f64) -> Event {
    Event::new(x, y, z, t)
}

/// The point of space at `(x, y, z)`.
pub fn spatial_point(x: f64, y: f64, z: f64) -> Spatial {
    Spatial::new(x, y, z)
}

/// A point of space at an instant.
pub fn at(p: Spatial, instant: f64) -> Event {
    p.cast::<cl42::Event>() + time() * instant
}

/// An event's place in space, its part across time.
pub fn spatial(e: Event) -> Spatial {
    e.cast::<cl42::Spatial>()
}

/// An event's time, read by `t`: `t & e`.
pub fn time_of(e: Event) -> f64 {
    (t() & e).s()
}

/// The coordinates of a point of space, read by the planes `x`, `y` and `z` (numga's
/// `coordinates`): a vector of VGA3D, for drawing and for Euclidean lengths.
pub fn coordinates(p: Spatial) -> gax::vga3d::Vector {
    gax::vga3d::Vector::new((x() & p).s(), (y() & p).s(), (z() & p).s())
}

/// An event as a null conformal point of weight one.
pub fn point(e: Event) -> Point {
    origin() + e.cast::<cl42::Point>() + infinity() * (e.scalar_product(e).s() / 2.0)
}

/// The event a conformal point stands for: its spacetime part over its weight.
pub fn event_of(p: Point) -> Event {
    p.cast::<cl42::Event>() * (1.0 / (horizon() & p).s())
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
        let factor = |b: Full| Full::from(cl42::Scalar::new(1.0)) + b;
        factor(Full::from(x()))
            * factor(Full::from(y() * t()))
            * factor(Full::from(z() * v()))
            * 0.125
    })
}

/// A multivector's action on a ray's twistor: multiplication from the left of the left ideal
/// `ψ P` the twistor stands for, read back as a twistor.
pub fn act(a: impl Into<Full>, psi: Twistor) -> Twistor {
    (a.into() * (psi * projector())).cast::<cl42::Twistor>() * 8.0
}

/// An intrinsic multivector's action on a ray's twistor, an extrinsic twistor: for an event,
/// zero on the twistors of the rays through it.
pub fn incidence(a: IntrinsicFull, psi: Twistor) -> ExtrinsicTwistor {
    (a * (psi * projector())).cast::<cl42::Twistor>() * 8.0
}

/// An intrinsic multivector's action on an extrinsic twistor, a ray's twistor: an event picks
/// a ray through it from a fixed twistor.
pub fn selection(a: IntrinsicFull, psi: ExtrinsicTwistor) -> Twistor {
    (a * (psi * projector())).cast::<cl42::Twistor>() * 8.0
}

/// A multivector's action on twistors, as a map.
pub fn representation(a: impl Into<Full>) -> TwistorMap {
    let a = a.into();
    TwistorMap::from_images(twistor_basis().map(|b| act(a, b)))
}

/// An intrinsic multivector's action on ray twistors, as a map.
pub fn incidence_map(a: IntrinsicFull) -> IncidenceMap {
    IncidenceMap::from_images(twistor_basis().map(|b| incidence(a, b)))
}

/// The conformal volume's action: a complex structure on twistors.
pub fn volume() -> TwistorMap {
    representation(cl42::Pseudoscalar::new(1.0))
}

/// The pairing `8 ⟨y z (ψ P)~ I (φ P)⟩` of a ray's twistor with an extrinsic one, across the unit
/// intrinsic volume `I`, that conformal rotors keep.
pub fn pairing() -> Pairing {
    static PAIRING: OnceLock<Pairing> = OnceLock::new();
    *PAIRING.get_or_init(|| {
        let yz = Full::from(y() ^ z());
        let volume = cl42::intrinsic::Pseudoscalar::new(1.0).cast::<cl42::Multivector>();
        let pair = |a: Twistor, b: ExtrinsicTwistor| {
            (yz * (a * projector()).reverse() * volume * (b * projector())).s() * 8.0
        };
        let basis = twistor_basis();
        Pairing::from_coeffs([core::array::from_fn(|i| {
            core::array::from_fn(|j| pair(basis[i], basis[j].with_orientation()))
        })])
    })
}

/// The pairing of a ray's twistor with an extrinsic one.
pub fn pair(psi: Twistor, phi: ExtrinsicTwistor) -> f64 {
    pairing().of(psi).of(phi).s()
}

/// The basis rays, in coefficient order.
fn ray_basis() -> [Ray; 15] {
    core::array::from_fn(|k| Ray::from_coeffs(unit_coefficients(k)))
}

/// The ray readout: a twistor paired with each basis ray's incidence on a second one is a form
/// on rays, and contracting it with the reciprocal basis turns it into the ray itself. The basis
/// rays are orthogonal blades, so each reciprocal is the blade's inverse. On a null twistor
/// twice, the result is its light ray, the join of its events.
pub fn ray_map() -> RayMap {
    static RAY: OnceLock<RayMap> = OnceLock::new();
    *RAY.get_or_init(|| {
        ray_basis().into_iter().fold(RayMap::zero(), |sum, r| {
            // The pairing with its second twistor through the ray's incidence.
            let weight = pairing()
                .at::<1>()
                .of(incidence_map(r.reverse().cast::<cl42::Multivector>()))
                .swap();
            sum + (r.inverse() * weight).cast::<cl42::Ray>()
        })
    })
}

/// The ray of a twistor, or the bilinear ray of two.
pub fn ray(psi: Twistor, phi: Twistor) -> Ray {
    ray_map().of(psi).of(phi)
}

/// A twistor of the light ray through two distinct null-separated conformal points: the
/// leading image of their product's action. Each point acts with its square, zero, so its
/// image lies in its kernel; null-separated points anticommute, so the image of their
/// product lies in both kernels.
pub fn through(first: Point, second: Point) -> Twistor {
    representation(first * second).svd().outputs[0]
}

/// The event where a conformal light ray meets the slice of constant time `instant`: the ray
/// joined with the hyperplane `t - horizon instant`, over its weight.
pub fn at_time(ray: Ray, instant: f64) -> Event {
    let crossing = (t() - horizon() * instant) ^ ray;
    event_of(crossing.cast::<cl42::Point>())
}

/// The future null direction of a ray's twistor, its time one: where the ray meets the plane at
/// infinity.
pub fn direction(psi: Twistor) -> Event {
    let tangent = (horizon() ^ ray(psi, psi)).cast::<cl42::Event>();
    tangent * (1.0 / time_of(tangent))
}

/// The fixed twistor `1 + y - z u + y z u` that is not null.
pub fn robinson_twistor() -> ExtrinsicTwistor {
    // Blades [1, y, z, u, yz, yu, zu, yzu].
    ExtrinsicTwistor::from_coeffs([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
}

/// The ray's twistor the event selects from the Robinson twistor.
pub fn robinson_ray_twistor(e: Event) -> Twistor {
    selection(point(e).cast::<cl42::Multivector>(), robinson_twistor())
}

/// The light ray the Robinson twistor selects at an event: the ray of its image under the
/// event's action.
pub fn robinson(e: Event) -> Event {
    direction(robinson_ray_twistor(e))
}

/// The sandwich `r x r̃`, cast back to the kind of `x` and keeping its orientation.
pub fn moved_point(r: Even, p: Point) -> Point {
    (r * p * r.reverse()).cast::<cl42::Point>()
}

/// The sandwich of a ray.
pub fn moved_ray(r: Even, l: Ray) -> Ray {
    (r * l * r.reverse()).cast::<cl42::Ray>()
}

/// The sandwich of a point of space.
pub fn moved_spatial(r: Even, p: Spatial) -> Spatial {
    (r * p * r.reverse()).cast::<cl42::Spatial>()
}

/// A source-free null electromagnetic field carried by the Robinson congruence, at an event.
pub fn hopfion(e: Event) -> Faraday {
    let elapsed = time_of(e);
    // The event's place in space relative to a flash of light running along y, as a patch of
    // space: its extent across time and the conformal pair undone.
    let flash = time() + y().hodge().cast::<cl42::Event>();
    let position = (across().inverse() | (e - flash * elapsed)).cast::<cl42::Bivector>();
    // The spinor turns the polarization and its propagation direction together; the helicity
    // turns the patch into the plane it turns in.
    let spinor = Even::from(cl42::Scalar::new(1.0)) + (position * helicity()).cast::<cl42::Even>();
    let polarization = (t() + y()) ^ x();
    // The spacetime volume rotates electric into magnetic polarization. Its inverse supplies
    // both the decaying amplitude and the local polarization angle.
    let weight = (cl42::Phasor::new(1.0 + e.scalar_product(e).s(), 0.0)
        + spacetime_volume() * (2.0 * elapsed))
        .inverse();
    let turned = (spinor * polarization * spinor.reverse()).cast::<cl42::Faraday>();
    turned * (weight * weight * weight)
}

/// The field's flux along the observer's time, `F T F̃`: its energy along time, its flow across
/// space.
pub fn flux(field: Faraday) -> Event {
    (field * time() * field.reverse()).cast::<cl42::Event>()
}

/// The field's part through time: its electric part, as planes `xt yt zt`.
pub fn temporal(field: Faraday) -> Faraday {
    Faraday::new(0.0, 0.0, 0.0, field.ext(), field.eyt(), field.ezt())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use gax::ApproxEq;
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

    /// A random rotation, boost, translation and special conformal transformation, composed:
    /// the translation and the special one generated by the duals of the joins `∞ ∨ a` and
    /// `o ∨ b`.
    fn conformal_rotor(rng: &mut Rng) -> Even {
        let (translation, special) = (random_event(rng, 0.2), random_event(rng, 0.2));
        let generator = |p: Point, e: Event| {
            (p & e.cast::<cl42::Point>())
                .hodge()
                .cast::<cl42::Bivector>()
        };
        turn(B::from(x() ^ y()), rng.normal() * 0.2)
            * turn(B::from(x() ^ t()), rng.normal() * 0.2)
            * turn(generator(infinity(), translation), 0.5)
            * turn(generator(origin(), special), 0.5)
    }

    /// The action on extrinsic twistors, numga's test-local `extrinsic`.
    fn act_extrinsic(a: Full, psi: ExtrinsicTwistor) -> ExtrinsicTwistor {
        (a * (psi * projector())).cast::<cl42::Twistor>() * 8.0
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
            let state = random_twistor(&mut rng);
            let partner: ExtrinsicTwistor = random_twistor(&mut rng).with_orientation();
            let rotor = conformal_rotor(&mut rng);
            let (moved, moved_partner) = (act(rotor, state), act_extrinsic(rotor.into(), partner));
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
            let across_space = spatial_point(rng.normal(), rng.normal(), rng.normal());
            let heading = time() + across_space.normalized().cast::<cl42::Event>();
            let separation = rng.range(0.2, 1.5);
            let (first, second) = (point(e), point(e + heading * separation));
            let twistor = through(first, second);
            let light = ray(twistor, twistor);
            let angle = phasor(rng.normal());
            let phase = Full::from(cl42::Scalar::new(angle.e20()))
                + Full::from(cl42::Pseudoscalar::new(angle.e01()));
            let rotor = conformal_rotor(&mut rng);
            let moved = act(rotor, twistor);
            let incident = |p: Point, psi: Twistor| incidence(p.cast::<cl42::Multivector>(), psi);
            // The two events have a shared null twistor. Its bilinear ray is the join of the
            // two, and turning it by the volume leaves its direction alone.
            assert!(first.scalar_product(first).s().abs() < TOLERANCE);
            assert!(second.scalar_product(second).s().abs() < TOLERANCE);
            assert!(vanishes(
                incident(first, twistor),
                ExtrinsicTwistor::zero(),
                TOLERANCE
            ));
            assert!(vanishes(
                incident(second, twistor),
                ExtrinsicTwistor::zero(),
                TOLERANCE
            ));
            // The port's: the ray's twistor is null. numga dropped this check when its pairing
            // came to run between the two orientations; relabelled, the number is the same.
            assert!(pair(twistor, twistor.with_orientation()).abs() < TOLERANCE);
            for p in [first, second] {
                let joined = p & light;
                assert!(vanishes(joined, joined * 0.0, TOLERANCE));
            }
            assert!(vanishes(light * light, Even::zero(), TOLERANCE));
            assert!(direction(twistor).max_abs_diff(&heading) < TOLERANCE);
            assert!(direction(act(phase, twistor)).max_abs_diff(&heading) < TOLERANCE);
            for p in [first, second] {
                let incident_moved = incident(moved_point(rotor, p), moved);
                assert!(vanishes(
                    incident_moved,
                    ExtrinsicTwistor::zero(),
                    TOLERANCE
                ));
            }
            assert!(ray(moved, moved).max_abs_diff(&moved_ray(rotor, light)) < TOLERANCE);
        }
    }

    #[test]
    fn robinson_twistors_stay_orthogonal_along_straight_future_null_rays() {
        let mut rng = rng(823);
        let fixed = robinson_twistor();
        let extrinsic_volume = act_extrinsic(cl42::Pseudoscalar::new(1.0).into(), fixed);
        for _ in 0..SAMPLES {
            let e = random_event(&mut rng, 1.0);
            let twistor = robinson_ray_twistor(e);
            let heading = robinson(e);
            // The event's selected twistor pairs to zero with the fixed twistor and with its
            // turn by the volume. Every event along its straight light ray selects the same
            // propagation direction and remains incident with the initial twistor.
            assert!(pair(twistor, fixed).abs() < TOLERANCE);
            assert!(pair(twistor, extrinsic_volume).abs() < TOLERANCE);
            // The port's: the selected twistor is null (numga dropped it, as above).
            assert!(pair(twistor, twistor.with_orientation()).abs() < TOLERANCE);
            assert!(heading.scalar_product(heading).s().abs() < TOLERANCE);
            assert!((time_of(heading) - 1.0).abs() < TOLERANCE);
            for k in 0..9 {
                let along = e + heading * (-1.0 + 0.25 * k as f64);
                assert!(robinson(along).max_abs_diff(&heading) < TOLERANCE);
                let incident = incidence(point(along).cast::<cl42::Multivector>(), twistor);
                assert!(vanishes(incident, ExtrinsicTwistor::zero(), TOLERANCE));
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

    /// numga's test: the field is null, its energy flows along the Robinson ray at unit time,
    /// and the ray's direction is left alone by the field's action (numga's 1e-14 and 1e-13).
    #[test]
    fn hopfion_is_null_and_its_energy_follows_the_robinson_congruence() {
        let mut rng = rng(41);
        for e in hopfion_events(&mut rng) {
            let field = hopfion(e);
            let flux = flux(field);
            let energy = time_of(flux);
            let heading = robinson(e);
            assert!(vanishes(field * field, Even::zero(), 1e-14));
            assert!((flux * (1.0 / energy)).max_abs_diff(&heading) < 1e-13);
            let commutator = field.commutator(heading);
            assert!(vanishes(commutator, commutator * 0.0, 1e-14));
            assert!(energy > 0.0);
        }
    }

    /// The field's spacetime derivative `∇F = Σ eⁱ ∂ᵢF` vanishes: both of Maxwell's equations
    /// without sources. The partial derivatives are fourth-order centred differences along the
    /// unit steps `X Y Z T` of the events (the duals of `x y z` and one unit along time, `-t*`),
    /// and `eⁱ` the reciprocal of each step, `s / s²`, which is the dual of the reciprocal of
    /// the vector the step is the dual of: so the sum is `(∇F)*`, as in numga. (The step along
    /// time is the dual of `-t`, whose reciprocal is `+t`.)
    #[test]
    fn hopfion_satisfies_vacuum_maxwell_equations() {
        let mut rng = rng(73);
        let step = 1e-4;
        let steps = [
            event(1.0, 0.0, 0.0, 0.0),
            event(0.0, 1.0, 0.0, 0.0),
            event(0.0, 0.0, 1.0, 0.0),
            time(),
        ];
        for e in hopfion_events(&mut rng) {
            let maxwell = steps.into_iter().fold(IntrinsicFull::zero(), |sum, along| {
                let at = |k: f64| hopfion(e + along * (k * step));
                let partial =
                    (at(-2.0) - at(2.0) + (at(1.0) - at(-1.0)) * 8.0) * (1.0 / (12.0 * step));
                let reciprocal = along * (1.0 / along.scalar_product(along).s());
                sum + reciprocal.cast::<cl42::Multivector>() * Full::from(partial)
            });
            // numga's tolerance: the differences' round-off, about 1e-12, with room.
            assert!(vanishes(maxwell, IntrinsicFull::zero(), 1e-9));
        }
    }

    /// The pairing, a point, a selected twistor, its ray and direction, a slice of the ray, the
    /// Robinson direction, the Hopfion and its flux at one event, against numga 35797de's numbers.
    #[test]
    fn the_numbers_match_numga() {
        let mut want_pairing = [[0.0; 8]; 8];
        for (i, sign) in [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
            .into_iter()
            .enumerate()
        {
            want_pairing[i][7 - i] = sign;
        }
        assert!(pairing().max_abs_diff(&Pairing::from_coeffs([want_pairing])) < 1e-15);
        assert!(origin().approx_eq(&Point::new(0.0, 0.0, 0.0, 0.0, -0.5, 0.5), 0.0));
        let minus_xz: cl42::intrinsic::Bivector = B::from(-(x() ^ z())).with_orientation();
        assert!(symmetry().approx_eq(&minus_xz, 0.0));

        let e = event(0.3, -0.7, 1.1, 0.4);
        let p = point(e);
        assert!(p.approx_eq(&Point::new(0.3, -0.7, 1.1, 0.4, 0.315, 1.315), 1e-15));
        let selected = robinson_ray_twistor(e);
        let want = Twistor::from_coeffs([-0.53, -0.1, -0.6, 0.0, -0.8, 1.4, -2.73, -2.1]);
        assert!(selected.max_abs_diff(&want) < 1e-14, "{selected:?}");
        let light = ray(selected, selected);
        let want = Ray::from_coeffs([
            -8.372, 15.1138, -6.676, 1.0738, 10.612, 1.0738, 5.404, 4.7138, 4.452, -10.612, -8.372,
            -6.676, 4.452, 10.4, 5.404,
        ]);
        assert!(light.max_abs_diff(&want) < 1e-12, "{light:?}");
        let heading = event(
            -0.43874643874643876,
            -0.25925925925925936,
            0.8603988603988606,
            1.0,
        );
        assert!(direction(selected).max_abs_diff(&heading) < 1e-14);
        assert!(robinson(e).max_abs_diff(&heading) < 1e-14);
        let slice = event(
            0.08062678062678076,
            -0.8296296296296298,
            1.5301994301994306,
            0.9,
        );
        assert!(at_time(light, 0.9).max_abs_diff(&slice) < 1e-14);
        let field = Faraday::new(
            -0.08512448434467576,
            0.05119945522225468,
            -0.13667820679783704,
            -0.06612126368229745,
            0.15494583772657225,
            0.01297151200676909,
        );
        assert!(hopfion(e).max_abs_diff(&field) < 1e-15);
        let want = event(
            -0.025051100379345513,
            -0.01480292295143144,
            0.049126183860794446,
            0.05709698852694984,
        );
        assert!(flux(hopfion(e)).max_abs_diff(&want) < 1e-15);
    }

    /// The coordinates the planes `x y z` read from a point of space are its coefficients.
    #[test]
    fn planes_read_a_point_s_coordinates() {
        let p = spatial_point(0.3, -0.7, 1.1);
        let read = coordinates(p);
        assert!(read.approx_eq(&gax::vga3d::Vector::new(0.3, -0.7, 1.1), 0.0));
        assert!(time_of(time()) == 1.0);
    }
}
