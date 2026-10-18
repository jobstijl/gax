//! numga's `coating/core.py`, the frequency domain: a layer's boundary amplitudes and their
//! generator from its constitutive map, their spectral propagation through stacks, and the
//! reflection and transmission of the waves of isotropic exteriors.

// numga's API in full: the tests use parts the animation does not.
#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::{FRAC_1_SQRT_2, TAU};
use gax::Complex;
use gax::sta::{Bivector, Even, Odd, Scalar, Vector, intrinsic};

pub type C = Complex<f64>;
pub type V = Vector<(), f64>;
pub type B = Bivector<(), f64>;
/// An excitation: the dual of a field, a bivector of the other orientation (intrinsic).
pub type Excitation = intrinsic::Bivector<(), f64>;
pub type Rotor = gax::Unit<Even<(), f64>>;
/// numga's `Constitutive`: a field in, its excitation out (`Antibivector <- Bivector`).
pub type Constitutive = intrinsic::Bivector<(Bivector,), f64>;
/// An axion coupling is a pseudoscalar: a scalar with the orientation of the dual.
pub type AxionCoupling = intrinsic::Scalar<(), f64>;

/// numga's `Polarization`, the tangential blades `x y`: a vector of the interface plane,
/// VGA2D's vector, with complex coefficients for its phase. [`tangential`] sends it into
/// spacetime and [`in_plane`] reads it back.
pub use gax::vga2d::Vector as Polarization;
/// numga's `Boundary`, the four amplitudes continuous across an interface: the blades
/// `x y txz tyz` of STA's odd part. gax's kinds are a closed family per algebra, and STA
/// declares none for these four blades (in `Odd` they would sit among four structural
/// zeros, which the 4 x 4 eigenproblem must not see). So the amplitudes are the four
/// coefficients of another four-blade kind, VGA3D's paravector, used as plain coordinates:
/// [`embed`] sends them to their blades in STA and [`amplitudes`] reads them back.
pub use gax::vga3d::Paravector as Boundary;

/// numga's `BoundaryMap`: the boundary amplitudes of a field.
pub type BoundaryMap = Boundary<(Bivector,), f64>;
/// The boundary amplitudes of an excitation.
pub type ExcitationBoundary = Boundary<(intrinsic::Bivector,), f64>;
/// numga's `FieldMap`: the field reconstructed from boundary amplitudes.
pub type FieldMap = Bivector<(Boundary,), f64>;
/// The amplitudes' generator along the layer's normal (numga's `Propagation`, real).
pub type Generator = Boundary<(Boundary,), f64>;
/// numga's `Propagation`: amplitudes carried across layers, with their phases.
pub type Propagation = Boundary<(Boundary,), C>;
/// numga's `Outgoing`: the amplitudes of the waves leaving a face, by their polarization.
pub type Outgoing = Boundary<(Polarization,), C>;
/// numga's `Readout`: a wave's polarization read from the amplitudes.
pub type Readout = Polarization<(Boundary,), C>;
/// numga's `ElectricMap`: a wave's electric vector from its tangential polarization.
pub type ElectricMap = Vector<(Polarization,), C>;
/// numga's `PolarizationMap`.
pub type PolarizationMap = Polarization<(Polarization,), C>;
/// A tangential polarization with its temporal phase.
pub type Wave = Polarization<(), C>;

pub fn t() -> V {
    Vector::new(1.0, 0.0, 0.0, 0.0)
}
pub fn x() -> V {
    Vector::new(0.0, 1.0, 0.0, 0.0)
}
pub fn y() -> V {
    Vector::new(0.0, 0.0, 1.0, 0.0)
}
pub fn z() -> V {
    Vector::new(0.0, 0.0, 0.0, 1.0)
}

/// The plane across the line, `x ∧ y`.
pub fn xy() -> B {
    x() ^ y()
}

/// The rotor turning by `angle` about the line `z`, from `x` towards `y`.
pub fn about_line(angle: f64) -> Rotor {
    (xy() * (angle / 2.0)).exp()
}

/// The boundary amplitudes as STA's odd elements: their blades `x y txz tyz`.
pub fn embed() -> Odd<(Boundary,), f64> {
    let (t, x, y, z) = (t(), x(), y(), z());
    Odd::from_images([
        x.cast::<Odd>(),
        y.cast::<Odd>(),
        (t ^ x ^ z).cast::<Odd>(),
        (t ^ y ^ z).cast::<Odd>(),
    ])
}

/// The boundary amplitudes of an odd element, its other blades dropped: the embedding's
/// pseudo-inverse, which reads each blade's coefficient with its sign.
pub fn amplitudes() -> Boundary<(Odd,), f64> {
    embed().pinv()
}

/// The interface plane's vectors in spacetime: `e1, e2` to `x, y`.
pub fn tangential() -> Vector<(Polarization,), f64> {
    Vector::from_images([x(), y()])
}

/// A vector's tangential part, read in the interface plane.
pub fn in_plane() -> Polarization<(Vector,), f64> {
    tangential().pinv()
}

/// The tangential electric field of a field, `t ∧ (t · (z ∧ F))`.
pub fn field_boundary() -> BoundaryMap {
    let (t, z) = (t(), z());
    amplitudes().of((t ^ (t | (z ^ Bivector::slot()))).cast::<Odd>())
}

/// The tangential magnetic excitation of an excitation, `t · (t ∧ (z ∧ G)*)` (numga's `dual`
/// is gax's `hodge`).
pub fn excitation_boundary() -> ExcitationBoundary {
    let (t, z) = (t(), z());
    amplitudes().of((t | (t ^ (z ^ Excitation::slot()).hodge())).cast::<Odd>())
}

/// The tangential electric field of the amplitudes as a polarization, `-t · (z · B)`:
/// undoing the two wedges of [`field_boundary`].
pub fn electric_trace() -> Polarization<(Boundary,), f64> {
    let (t, z) = (t(), z());
    in_plane().of((-(t | (z | embed()))).cast::<Vector>())
}

/// Principal electric responses along `x y z` and a scalar inverse permeability, in vacuum
/// units: each axis's plane with the observer weighted by its permittivity, the magnetic
/// planes by the reluctivity, then the Hodge dual (numga's `dual`).
pub fn dielectric(permittivity: [f64; 3], reluctivity: f64) -> Constitutive {
    let open = Bivector::slot();
    let electric: Bivector<(Bivector,), f64> = [x(), y(), z()]
        .into_iter()
        .zip(permittivity)
        .map(|(axis, e)| {
            let plane = axis ^ t();
            plane * (plane | open) * e
        })
        .sum();
    let magnetic = (open + (t() >> open)) * 0.5;
    (electric + magnetic * reluctivity).hodge()
}

/// The axion response of strength `alpha`: the coupling, a pseudoscalar, times the identity
/// on field planes. The coupling's number is declared the pseudoscalar's orientation.
pub fn axion_response(alpha: f64) -> Constitutive {
    let coupling: AxionCoupling = Scalar::new(alpha).with_orientation();
    coupling * Bivector::slot()
}

/// A lossless nonmagnetic isotropic medium of refractive index `index`.
pub fn isotropic(index: f64) -> Constitutive {
    dielectric([index * index; 3], 1.0)
}

/// The vacuum wavenumber times a thickness: the phase a unit index gathers across it.
pub fn optical_distance(thickness: f64, wavelength: f64) -> f64 {
    TAU * thickness / wavelength
}

/// A layer normal to `z` at a fixed tangential wave covector.
#[derive(Clone, Copy, Debug)]
pub struct Medium {
    pub boundary: BoundaryMap,
    pub reconstruct: FieldMap,
    pub generator: Generator,
}

impl Medium {
    /// Reduce Maxwell to a layer normal to `z`, at the tangential wave covector `parallel`
    /// (its temporal part the unit `t`). The longitudinal material response must be
    /// invertible; no free surface charge or current is present.
    pub fn from_chi(chi: Constitutive, parallel: V) -> Medium {
        let (z, open) = (z(), Bivector::slot());
        // The four shared boundary amplitudes are continuous between different media.
        let boundary = field_boundary() + excitation_boundary().of(chi);
        // The remaining two equations contain no normal derivative: impose them as zero.
        let constraint = (-(z | (z ^ (parallel ^ open)))).cast::<Odd>()
            - (z * (z | (parallel ^ chi).hodge())).cast::<Odd>();
        // Six equations in STA's eight odd blades (the other two rows are zero): an exact
        // least-squares solve.
        let reconstruct = (embed().of(boundary) + constraint).lstsq(embed());
        // Positive eigenvalues give phase propagation towards +z when temporal phase
        // increases.
        let generator = field_boundary().of(z | (parallel ^ reconstruct))
            + excitation_boundary().of(z | (parallel ^ chi.of(reconstruct)));
        Medium {
            boundary,
            reconstruct,
            generator,
        }
    }
}

/// Eigenvalues closer than this are one, repeated: a double normal index (both
/// polarizations of an isotropic layer) comes out of the eigensolver split by rounding,
/// and Sylvester's formula divides by the differences.
const DEGENERATE: f64 = 1e-7;

/// A homogeneous layer's generator split by its distinct eigenvalues, the normal indices of
/// its waves, into spectral projectors, so that it carries boundary amplitudes across any
/// thickness: `exp(i k₀ d G) = Σ exp(i k₀ d n) Πₙ`, right to left, as numga's `propagate`.
///
/// numga reads each mode's amplitude through the reciprocal frame of the eigenvectors, a
/// complex solve. gax solves for real coefficients only, and its `eig` gives one
/// eigenvector per eigenvalue (a repeated index repeats it). So the projectors come from
/// the eigenvalues alone, by Sylvester's formula `Πₙ = Π_{m ≠ n} (G − m) / (n − m)`. The
/// generator must be diagonalizable, as numga's must.
#[derive(Clone, Debug)]
pub struct Spectrum {
    indices: Vec<C>,
    projectors: Vec<Propagation>,
}

impl Spectrum {
    /// The spectrum of a layer's generator.
    pub fn of(generator: Generator) -> Spectrum {
        let indices = generator
            .eigvals()
            .into_iter()
            .fold(Vec::new(), |mut kept: Vec<C>, n| {
                if kept.iter().all(|m| (*m - n).abs() > DEGENERATE) {
                    kept.push(n);
                }
                kept
            });
        let g = generator.map_coefs(C::real);
        let one: Propagation = Boundary::slot();
        let projectors = indices
            .iter()
            .map(|&n| {
                indices
                    .iter()
                    .filter(|&&m| m != n)
                    .fold(one, |p, &m| p.of((g - one.gp(m)).gp((n - m).recip())))
            })
            .collect();
        Spectrum {
            indices,
            projectors,
        }
    }

    /// Carry boundary amplitudes right to left across the optical distance (the vacuum
    /// wavenumber times the thickness).
    pub fn propagate(&self, optical_distance: f64) -> Propagation {
        self.indices
            .iter()
            .zip(&self.projectors)
            .map(|(n, p)| p.gp((C::i() * n.scale(optical_distance)).exp()))
            .sum()
    }
}

/// Carry boundary amplitudes right to left through a homogeneous layer.
pub fn propagate(generator: Generator, optical_distance: f64) -> Propagation {
    Spectrum::of(generator).propagate(optical_distance)
}

/// Layers ordered from the incident medium to the substrate.
pub fn compose(layers: impl IntoIterator<Item = Propagation>) -> Propagation {
    layers
        .into_iter()
        .reduce(|propagation, layer| propagation.of(layer))
        .expect("at least one layer")
}

/// The physical fields inside a stack, `[layers][samples]`, from the state at its exit
/// face. `fractions` measure distance from each layer's left face to its right face.
pub fn interior(
    media: &[Medium],
    thicknesses: &[f64],
    wavelength: f64,
    exit_state: Boundary<(), C>,
    fractions: &[f64],
) -> Vec<Vec<Bivector<(), C>>> {
    // Each layer's right face holds the exit state carried back through every layer
    // after it; from there back to the samples inside it.
    let mut right_face = exit_state;
    let mut fields: Vec<Vec<Bivector<(), C>>> = media
        .iter()
        .zip(thicknesses)
        .rev()
        .map(|(medium, thickness)| {
            let spectrum = Spectrum::of(medium.generator);
            let distance = optical_distance(*thickness, wavelength);
            let reconstruct = medium.reconstruct.map_coefs(C::real);
            let samples = fractions
                .iter()
                .map(|f| {
                    let inward = spectrum.propagate(distance * (1.0 - f));
                    reconstruct.of(inward.of(right_face))
                })
                .collect();
            right_face = spectrum.propagate(distance).of(right_face);
            samples
        })
        .collect();
    fields.reverse();
    fields
}

/// The waves of an exterior medium at its face.
#[derive(Clone, Copy, Debug)]
pub struct Ports {
    pub outgoing: Outgoing,
    pub incoming: Readout,
    pub returning: Readout,
    pub electric_field: ElectricMap,
    pub normal_index: f64,
}

impl Ports {
    /// Waves in a lossless, nonmagnetic isotropic exterior of refractive index `index`,
    /// including a real constant axion term. The normal index is real and nonzero: the
    /// exterior waves must propagate. A polarization gives the tangential electric field,
    /// including its phase.
    #[allow(clippy::disallowed_methods)] // the normal index √(n² - k²): Snell's law
    pub fn from_medium(medium: &Medium, index: f64, parallel: V) -> Ports {
        let tangential_wave = parallel - t();
        let normal_index =
            (index * index + tangential_wave.scalar_product(tangential_wave).s()).sqrt();
        // Complete the tangential electric field to a field perpendicular to the ray.
        let electric_field = tangential() + z() * (parallel | tangential()) / normal_index;
        let wavevector = parallel + z() * normal_index;
        let outgoing = medium.boundary.of(wavevector ^ electric_field);
        // Exterior evolution has just two eigenvalues, the signed normal indices.
        let one = Boundary::slot();
        let half = |sign: f64| {
            let projector = (one + medium.generator * (sign / normal_index)) * 0.5;
            electric_trace().of(projector).map_coefs(C::real)
        };
        Ports {
            outgoing: outgoing.map_coefs(C::real),
            incoming: half(1.0),
            returning: half(-1.0),
            electric_field: electric_field.map_coefs(C::real),
            normal_index,
        }
    }

    /// The isotropic exterior of refractive index `index`.
    pub fn isotropic(index: f64, parallel: V) -> Ports {
        let medium = Medium::from_chi(self::isotropic(index), parallel);
        Ports::from_medium(&medium, index, parallel)
    }

    /// Normal power flux, omitting the common factor of half the vacuum admittance.
    pub fn power(&self, polarization: Wave) -> f64 {
        let field = self.electric_field.of(polarization);
        // Complex coefficients carry time phase: average the two real quadratures. The
        // spatial metric is negative, hence the minus sign in their squared lengths.
        let (real, imaginary) = (field.map_coefs(|c| c.re), field.map_coefs(|c| c.im));
        let magnitude = -(real.scalar_product(real).s() + imaginary.scalar_product(imaginary).s());
        self.normal_index * magnitude
    }
}

/// The reflection and transmission of an interface, as maps on the incident polarization.
#[derive(Clone, Copy, Debug)]
pub struct Scattering {
    pub reflection: PolarizationMap,
    pub transmission: PolarizationMap,
}

impl Scattering {
    /// The reflected and transmitted powers as fractions of the incident power.
    pub fn powers(&self, polarization: Wave, incident: &Ports, substrate: &Ports) -> (f64, f64) {
        let normalization = incident.power(polarization);
        let reflected = incident.power(self.reflection.of(polarization));
        let transmitted = substrate.power(self.transmission.of(polarization));
        (reflected / normalization, transmitted / normalization)
    }
}

/// The inverse of a map of the interface plane by Cayley-Hamilton, `M⁻¹ = (tr M - M) /
/// det M`, the determinant being the area `M e1 ∧ M e2`: no solve, so complex coefficients
/// do.
fn inverse(m: PolarizationMap) -> PolarizationMap {
    let one: PolarizationMap = Polarization::slot();
    let (zero, unit) = (C::real(0.0), C::real(1.0));
    let (a, b) = (
        m.of(Polarization::new(unit, zero)),
        m.of(Polarization::new(zero, unit)),
    );
    let (trace, det) = (a.e1() + b.e2(), (a ^ b).e12());
    (one.gp(trace) - m).gp(det.recip())
}

/// Solve the interface conditions with the incident polarization left open. The entrance
/// state is the substrate's outgoing wave, carried back through any coating to the entrance
/// face.
pub fn scatter(entrance: Outgoing, incident: &Ports) -> Scattering {
    let transmission = inverse(incident.incoming.of(entrance));
    let reflection = incident.returning.of(entrance).of(transmission);
    Scattering {
        reflection,
        transmission,
    }
}

/// The circular polarization `(x ± i y) / √2` of sense `sign`.
pub fn circular(sign: f64) -> Wave {
    Polarization::new(C::real(FRAC_1_SQRT_2), C::new(0.0, sign * FRAC_1_SQRT_2))
}

/// The plain `x` polarization.
pub fn along_x() -> Wave {
    Polarization::new(C::real(1.0), C::real(0.0))
}

/// The plain `y` polarization.
pub fn along_y() -> Wave {
    Polarization::new(C::real(0.0), C::real(1.0))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use gax::sta::Trivector;
    use gax::{ApproxEq, Extensor, Slots};
    use gax_numga_examples::signal::linspace;

    /// numga's tolerance for the frequency-domain checks.
    pub const ROUND_OFF: f64 = 1e-10;

    /// The largest difference between the coefficients of two complex values or maps.
    fn gap<M: Extensor<Coef = C>>(a: M, b: M) -> f64 {
        let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
        a.iter()
            .zip(b)
            .flat_map(|(p, q)| {
                (0..<M::Slots as Slots>::SIZE)
                    .map(move |k| (M::Slots::get_flat(p, k) - M::Slots::get_flat(q, k)).abs())
            })
            .fold(0.0, |m: f64, d| if d.is_nan() || d > m { d } else { m })
    }

    /// Whether two complex values or maps agree to numga's round-off.
    pub fn close<M: Extensor<Coef = C>>(a: M, b: M) -> bool {
        gap(a, b) <= ROUND_OFF
    }

    /// The same for real values or maps.
    fn near<M: ApproxEq>(a: M, b: M) -> bool {
        a.max_abs_diff(&b) <= ROUND_OFF
    }

    /// The polarization with complex coefficients `x` and `y`.
    fn wave(x: C, y: C) -> Wave {
        Polarization::new(x, y)
    }

    fn real(v: f64) -> C {
        C::real(v)
    }

    /// A real vector as a complex one.
    pub fn complex(v: Vector<(), f64>) -> Vector<(), C> {
        v.map_coefs(C::real)
    }

    /// The transpose of a map of the plane (numga's `adjoint`; the plane's metric is
    /// Euclidean): `Mᵀ e_j = Σ_i (e_j · M e_i) e_i`.
    pub fn adjoint(m: PolarizationMap) -> PolarizationMap {
        let basis = [along_x(), along_y()];
        PolarizationMap::from_images(basis.map(|ej| {
            basis
                .iter()
                .map(|ei| ei.gp((ej | m.of(*ei)).s()))
                .fold(Wave::zero(), |a, b| a + b)
        }))
    }

    /// Fresnel's reflection and transmission amplitudes between two admittances.
    fn fresnel(left: f64, right: f64) -> (f64, f64) {
        ((left - right) / (left + right), 2.0 * left / (left + right))
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // Fresnel's angles and admittances: the references
    fn bare_interface_matches_both_fresnel_polarizations_and_brewster_angle() {
        let (incident_index, substrate_index) = (1.0, 1.5);
        let tangent_at = |angle: f64| incident_index * angle.sin();
        let ports_at =
            |angle: f64, index: f64| Ports::isotropic(index, t() + x() * tangent_at(angle));
        let brewster = (substrate_index / incident_index).atan();
        for angle in [0.0f64, 0.4, 0.9, brewster] {
            let tangent = tangent_at(angle);
            let incident_normal = incident_index * angle.cos();
            let substrate_normal = (substrate_index * substrate_index - tangent * tangent).sqrt();
            // `y` (s) and `x` (p) admittances.
            let admittances = |index: f64, normal: f64| [normal, index * index / normal];
            let incident_admittance = admittances(incident_index, incident_normal);
            let substrate_admittance = admittances(substrate_index, substrate_normal);

            let incident = ports_at(angle, incident_index);
            let substrate = ports_at(angle, substrate_index);
            let result = scatter(substrate.outgoing, &incident);
            for (k, polarization) in [along_y(), along_x()].into_iter().enumerate() {
                let (a, b) = (incident_admittance[k], substrate_admittance[k]);
                let (reflected, transmitted) = fresnel(a, b);
                let (reflectance, transmittance) =
                    result.powers(polarization, &incident, &substrate);
                let scaled = |s: f64| polarization.gp(real(s));
                assert!(close(result.reflection.of(polarization), scaled(reflected)));
                assert!(close(
                    result.transmission.of(polarization),
                    scaled(transmitted)
                ));
                assert!((reflectance - reflected * reflected).abs() < ROUND_OFF);
                let expected = b / a * transmitted * transmitted;
                assert!((transmittance - expected).abs() < ROUND_OFF);
            }
        }
        // At Brewster's angle no `x` light is reflected.
        let result = scatter(
            ports_at(brewster, substrate_index).outgoing,
            &ports_at(brewster, incident_index),
        );
        assert!(close(result.reflection.of(along_x()), Wave::zero()));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the matching index √(n₀ n₁): the reference
    fn quarter_wave_matching_layer_cancels_reflection_with_the_correct_phase() {
        let (incident_index, substrate_index) = (1.0f64, 1.52);
        let coating_index = (incident_index * substrate_index).sqrt();
        let optical_distance = core::f64::consts::PI / (2.0 * coating_index);
        let polarization = circular(1.0);
        let medium = Medium::from_chi(isotropic(coating_index), t());
        let incident = Ports::isotropic(incident_index, t());
        let substrate = Ports::isotropic(substrate_index, t());
        let coating = propagate(medium.generator, optical_distance);
        let result = scatter(coating.of(substrate.outgoing), &incident);
        let transmitted = polarization.gp(C::new(0.0, -(incident_index / substrate_index).sqrt()));
        let (reflectance, transmittance) = result.powers(polarization, &incident, &substrate);

        assert!(close(result.reflection, PolarizationMap::zero()));
        assert!(close(result.transmission.of(polarization), transmitted));
        assert!(reflectance.abs() < ROUND_OFF);
        assert!((transmittance - 1.0).abs() < ROUND_OFF);
    }

    /// The quarter-wave mirror of `pairs` high- and low-index pairs, both ways round.
    fn quarter_wave_mirror(pairs: i32) {
        let (incident_index, substrate_index) = (1.0, 1.52);
        let (high_index, low_index) = (2.3, 1.46);
        let incident = Ports::isotropic(incident_index, t());
        let substrate = Ports::isotropic(substrate_index, t());
        // Two stacks: high-index first and low-index first.
        for (first, second) in [(high_index, low_index), (low_index, high_index)] {
            let ratio: f64 = first / second;
            let layers = (0..pairs).flat_map(|_| [first, second]).map(|index| {
                let medium = Medium::from_chi(isotropic(index), t());
                propagate(medium.generator, core::f64::consts::PI / (2.0 * index))
            });
            let result = scatter(compose(layers).of(substrate.outgoing), &incident);
            // Each pair of quarter waves scales the substrate's effective index by `ratio²`.
            let effective = substrate_index * ratio.powi(2 * pairs);
            let reflected = (incident_index - effective) / (incident_index + effective);
            let transmitted = (-1.0f64).powi(pairs) * 2.0 * incident_index
                / (incident_index / ratio.powi(pairs) + substrate_index * ratio.powi(pairs));
            let (reflectance, transmittance) = result.powers(along_x(), &incident, &substrate);

            assert!(close(
                result.reflection.of(along_x()),
                along_x().gp(real(reflected))
            ));
            assert!(close(
                result.transmission.of(along_x()),
                along_x().gp(real(transmitted))
            ));
            assert!((reflectance + transmittance - 1.0).abs() < ROUND_OFF);
        }
    }

    #[test]
    fn quarter_wave_mirror_respects_layer_order_and_impedance_inversion_1() {
        quarter_wave_mirror(1);
    }

    #[test]
    fn quarter_wave_mirror_respects_layer_order_and_impedance_inversion_3() {
        quarter_wave_mirror(3);
    }

    #[test]
    fn quarter_wave_mirror_respects_layer_order_and_impedance_inversion_6() {
        quarter_wave_mirror(6);
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the films' angles, admittances and phases: the references
    fn oblique_multilayer_matches_recursive_multiple_reflections() {
        let (incident_index, substrate_index) = (1.0, 1.52);
        let indices = [1.38, 2.2, 1.65];
        let thicknesses = [137.0, 82.0, 110.0];
        let normal_in = |index: f64, tangent: f64| (index * index - tangent * tangent).sqrt();
        // `y` (s) and `x` (p) admittances.
        let admittances = |index: f64, normal: f64| [normal, index * index / normal];
        for angle in [0.0f64, 0.4, 0.85] {
            let tangent = incident_index * angle.sin();
            let parallel = t() + x() * tangent;
            let incident = Ports::isotropic(incident_index, parallel);
            let substrate = Ports::isotropic(substrate_index, parallel);
            let media = indices.map(|index| Medium::from_chi(isotropic(index), parallel));
            for wavelength in [450.0, 550.0, 710.0] {
                let layers = media
                    .iter()
                    .zip(thicknesses)
                    .map(|(m, d)| propagate(m.generator, optical_distance(d, wavelength)));
                let result = scatter(compose(layers).of(substrate.outgoing), &incident);
                for (k, polarization) in [along_y(), along_x()].into_iter().enumerate() {
                    // Independently sum repeated reflections within each film, starting at
                    // the substrate.
                    let film =
                        |i: usize| admittances(indices[i], normal_in(indices[i], tangent))[k];
                    let substrate_admittance =
                        admittances(substrate_index, normal_in(substrate_index, tangent))[k];
                    let (r, tr) = fresnel(film(2), substrate_admittance);
                    let (mut reflected, mut transmitted) = (real(r), real(tr));
                    for i in (0..indices.len()).rev() {
                        let left = match i {
                            0 => admittances(incident_index, incident_index * angle.cos())[k],
                            _ => film(i - 1),
                        };
                        let (interface_reflection, interface_transmission) = fresnel(left, film(i));
                        let phase = normal_in(indices[i], tangent)
                            * optical_distance(thicknesses[i], wavelength);
                        let travel = C::new(0.0, -phase).exp();
                        let denominator =
                            real(1.0) + reflected * travel * travel * real(interface_reflection);
                        transmitted =
                            transmitted * travel * real(interface_transmission) / denominator;
                        reflected = (real(interface_reflection) + reflected * travel * travel)
                            / denominator;
                    }
                    let (reflectance, transmittance) =
                        result.powers(polarization, &incident, &substrate);
                    assert!(close(
                        result.reflection.of(polarization),
                        polarization.gp(reflected)
                    ));
                    assert!(close(
                        result.transmission.of(polarization),
                        polarization.gp(transmitted)
                    ));
                    assert!((reflectance - reflected.norm_sqr()).abs() < ROUND_OFF);
                    assert!((reflectance + transmittance - 1.0).abs() < ROUND_OFF);
                }
            }
        }
    }

    #[test]
    fn anisotropic_chi_reconstruction_satisfies_full_maxwell_and_rotates_covariantly() {
        let permittivity = [2.2, 3.1, 4.3];
        let (reluctivity, axion) = (0.8, 0.37);
        let yz = y() ^ z();
        let material_turn = (xy() * 0.23 + yz * 0.17).exp();
        let frame_turn = (xy() * 0.31).exp();
        let parallel = t() + x() * 0.4 + y() * 0.1;
        let base = dielectric(permittivity, reluctivity);
        let open = Bivector::slot();
        let chi = (material_turn >> base.of(material_turn << open)) + axion_response(axion);

        let medium = Medium::from_chi(chi, parallel);
        let (indices, modes) = medium.generator.eig();
        let reconstruct = medium.reconstruct.map_coefs(C::real);
        let moved_chi = frame_turn >> chi.of(frame_turn << open);
        let moved = Medium::from_chi(moved_chi, frame_turn >> parallel);
        // The turn's action on boundary amplitudes, and its inverse: a turn about the normal
        // keeps the boundary blades among themselves.
        let turned = amplitudes().of(frame_turn >> embed());
        let unturned = amplitudes().of(frame_turn << embed());

        assert!(near(
            medium.boundary.of(medium.reconstruct),
            Boundary::slot()
        ));
        for (index, mode) in indices.into_iter().zip(modes) {
            let field = reconstruct.of(mode);
            let wavevector = complex(parallel) + complex(z()).gp(index);
            assert!(close(wavevector ^ field, gax::sta::Trivector::zero()));
            let excitation = chi.map_coefs(C::real).of(field);
            assert!(close(
                wavevector ^ excitation,
                gax::sta::intrinsic::Trivector::zero()
            ));
        }
        assert!(near(
            moved.generator,
            turned.of(medium.generator).of(unturned)
        ));
        assert!(near(
            moved.reconstruct.of(turned),
            frame_turn >> medium.reconstruct
        ));
    }

    #[test]
    fn axion_jump_changes_interfaces_without_changing_bulk_wave_indices() {
        let index = 1.5;
        let common_axion = 0.43;
        let base = isotropic(index);
        let scatter_between = |left: Constitutive, right: Constitutive| {
            let incident = Ports::from_medium(&Medium::from_chi(left, t()), index, t());
            let substrate_medium = Medium::from_chi(right, t());
            let substrate = Ports::from_medium(&substrate_medium, index, t());
            (
                scatter(substrate.outgoing, &incident),
                incident,
                substrate,
                substrate_medium,
            )
        };
        for jump in [0.0, 0.2, 0.7, 2.0] {
            let (result, incident, substrate, substrate_medium) =
                scatter_between(base, base + axion_response(jump));
            let denominator = 4.0 * index * index + jump * jump;
            let reflected = wave(
                real(-jump * jump / denominator),
                real(2.0 * index * jump / denominator),
            );
            let (reflectance, transmittance) = result.powers(along_x(), &incident, &substrate);
            let shifted_base = base + axion_response(common_axion);
            let (shifted, ..) = scatter_between(shifted_base, shifted_base + axion_response(jump));

            assert!(close(result.reflection.of(along_x()), reflected));
            assert!(close(
                result.transmission.of(along_x()),
                along_x() + reflected
            ));
            assert!((reflectance - jump * jump / denominator).abs() < ROUND_OFF);
            assert!((reflectance + transmittance - 1.0).abs() < ROUND_OFF);
            let generator = substrate_medium.generator;
            assert!(near(
                generator.of(generator),
                Boundary::slot() * (index * index)
            ));
            assert!(close(shifted.reflection, result.reflection));
            assert!(close(shifted.transmission, result.transmission));
        }
    }

    #[test]
    fn interior_fields_match_every_interface_and_conserve_normal_power() {
        let (incident_index, substrate_index) = (1.2, 1.6);
        let permittivities = [[2.2, 3.1, 4.3], [3.3, 2.5, 3.8], [2.8, 3.7, 2.4]];
        let reluctivities = [0.8, 1.1, 0.9];
        let axions = [-0.2, 0.3, 0.1];
        let turns = [(0.2, 0.15), (-0.3, 0.1), (0.1, -0.2)];
        let thicknesses = [80.0, 140.0, 65.0];
        let wavelength = 630.0;
        let fractions = linspace(0.0, 1.0, 9);
        let parallel = t() + x() * 0.3;
        let yz = y() ^ z();
        let open = Bivector::slot();
        let chis: Vec<Constitutive> = (0..3)
            .map(|k| {
                let rotation = (xy() * turns[k].0 + yz * turns[k].1).exp();
                let base = dielectric(permittivities[k], reluctivities[k]);
                (rotation >> base.of(rotation << open)) + axion_response(axions[k])
            })
            .collect();
        let (incident_chi, substrate_chi) = (isotropic(incident_index), isotropic(substrate_index));
        let media: Vec<Medium> = chis
            .iter()
            .map(|chi| Medium::from_chi(*chi, parallel))
            .collect();
        let incident = Ports::from_medium(
            &Medium::from_chi(incident_chi, parallel),
            incident_index,
            parallel,
        );
        let substrate_medium = Medium::from_chi(substrate_chi, parallel);
        let substrate = Ports::from_medium(&substrate_medium, substrate_index, parallel);
        let layers = media
            .iter()
            .zip(thicknesses)
            .map(|(m, d)| propagate(m.generator, optical_distance(d, wavelength)));
        let result = scatter(compose(layers).of(substrate.outgoing), &incident);
        let (normal, spatial_volume) = (z(), x() ^ y() ^ z());

        for polarization in [circular(1.0), circular(-1.0)] {
            let transmitted = result.transmission.of(polarization);
            let exit_state = substrate.outgoing.of(transmitted);
            let mut right_field = substrate_medium
                .reconstruct
                .map_coefs(C::real)
                .of(exit_state);
            let mut right_excitation = substrate_chi.map_coefs(C::real).of(right_field);
            let transmitted_power = substrate.power(transmitted);
            let fields = interior(&media, &thicknesses, wavelength, exit_state, &fractions);
            for k in (0..thicknesses.len()).rev() {
                let chi = chis[k].map_coefs(C::real);
                let (first, last) = (fields[k][0], fields[k][fractions.len() - 1]);
                // Maxwell's jump conditions compare physical fields across different materials.
                assert!(close(
                    complex(normal) ^ (last - right_field),
                    Trivector::zero()
                ));
                assert!(close(
                    complex(normal) ^ (chi.of(last) - right_excitation),
                    intrinsic::Trivector::zero()
                ));
                for field in &fields[k] {
                    let electric = complex(t()) | *field;
                    let magnetic = complex(t()) | chi.of(*field);
                    let flux = |f: fn(C) -> f64| {
                        (electric.map_coefs(f) ^ magnetic.map_coefs(f) ^ normal)
                            .scalar_product(spatial_volume)
                            .s()
                    };
                    let total = flux(|c| c.re) + flux(|c| c.im);
                    assert!((total - transmitted_power).abs() < ROUND_OFF);
                }
                (right_field, right_excitation) = (first, chi.of(first));
            }

            // At the entrance the sampled field meets the incident and reflected exterior waves.
            let n = incident.normal_index;
            let forward_wavevector = complex(parallel + z() * n);
            let backward_wavevector = complex(parallel - z() * n);
            let backward_electric =
                (tangential() - z() * (parallel | tangential()) / n).map_coefs(C::real);
            let exterior_field = (forward_wavevector ^ incident.electric_field.of(polarization))
                + (backward_wavevector ^ backward_electric.of(result.reflection.of(polarization)));
            assert!(close(
                complex(normal) ^ (right_field - exterior_field),
                Trivector::zero()
            ));
            let exterior_excitation = incident_chi.map_coefs(C::real).of(exterior_field);
            assert!(close(
                complex(normal) ^ (right_excitation - exterior_excitation),
                intrinsic::Trivector::zero()
            ));
        }
    }

    #[test]
    fn the_readouts_invert_the_embeddings() {
        assert!(near(amplitudes().of(embed()), Boundary::slot()));
        assert!(near(in_plane().of(tangential()), Polarization::slot()));
    }
}
