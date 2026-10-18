//! numga's `coating/scenarios.py`: coatings on glass, a twisted birefringent stack (a helix),
//! an axion interface, and a pulse stepped through the helix in time.

use core::f64::consts::TAU;

use crate::coating::{
    B, C, Constitutive, Medium, Polarization, Ports, Scattering, Spectrum, V, about_line, along_x,
    axion_response, circular, compose, dielectric, isotropic, optical_distance, scatter, t, x,
};
use crate::line::{Line, swept};
use gax::field::Field;
use gax::sta::Bivector;
use gax_numga_examples::signal::linspace;

/// The coatings' design wavelength, nm.
const DESIGN_WAVELENGTH: f64 = 550.0;
const INCIDENT_INDEX: f64 = 1.0;
const SUBSTRATE_INDEX: f64 = 1.5;
/// Bare glass, the ideal matching layer `√(n₀ n₁)` (1.2247…), and magnesium fluoride.
const COATING_INDICES: [f64; 3] = [SUBSTRATE_INDEX, 1.224_744_871_391_589, 1.38];
/// The coatings' optical thicknesses: none, and a quarter of the design wavelength.
const OPTICAL_THICKNESSES: [f64; 3] = [0.0, DESIGN_WAVELENGTH / 4.0, DESIGN_WAVELENGTH / 4.0];
const ORDINARY_INDEX: f64 = 1.5;
const EXTRAORDINARY_INDEX: f64 = 1.8;
pub const SURROUNDING_INDEX: f64 = (ORDINARY_INDEX + EXTRAORDINARY_INDEX) / 2.0;
/// The helix's pitch, nm per full turn of the optic axis.
const PITCH: f64 = 400.0;
const TURNS: usize = 6;
pub const SLICES_PER_TURN: usize = 16;
pub const LAYERS: usize = TURNS * SLICES_PER_TURN;
/// Each slice's thickness, nm.
pub const THICKNESS: f64 = PITCH / SLICES_PER_TURN as f64;
/// The crystal's permittivities: extraordinary along its optic axis `x`, ordinary across.
const CRYSTAL_PERMITTIVITY: [f64; 3] = [
    EXTRAORDINARY_INDEX * EXTRAORDINARY_INDEX,
    ORDINARY_INDEX * ORDINARY_INDEX,
    ORDINARY_INDEX * ORDINARY_INDEX,
];
/// The two senses of twist, and of the incident circular polarizations.
const TWIST_SIGNS: [f64; 2] = [1.0, -1.0];
/// The centre of the helix's reflection band, nm.
pub const BAND_WAVELENGTH: f64 = PITCH * SURROUNDING_INDEX;

// The time domain: the band's pulse sent along a line of nodes through the same helix, at
// light speed one; lengths and times in nm.
const SPACING: f64 = 40.0;
pub const CELLS: usize = 1000;
/// The time step per node spacing.
const COURANT: f64 = 1.0;
pub const SLAB_START: f64 = 18000.0;
pub const SLAB_END: f64 = SLAB_START + TURNS as f64 * PITCH;
/// The pulse's centre and its envelope's standard deviation.
const PULSE_START: f64 = 10000.0;
const PULSE_WIDTH: f64 = 2000.0;
/// How far light travels in the run.
const DURATION: f64 = 27000.0;
/// The states kept of the run, evenly spaced.
const FRAMES: usize = 120;
/// The pulse's cases: its carrier turning with the helix, against it, and their average,
/// which does not turn. The animation shows the last; the tests read the first two.
#[cfg_attr(not(test), allow(dead_code))]
pub const WITH: usize = 0;
#[cfg_attr(not(test), allow(dead_code))]
pub const AGAINST: usize = 1;
pub const LINEAR: usize = 2;

/// The optic axis and the crystal's map at `depth` into a helix of sense `twist` (±1).
/// Both senses use the same crystal; only the rotor transporting `χ` changes.
pub fn helix(depth: f64, twist: f64) -> (V, Constitutive) {
    let rotor = about_line(TAU * depth / PITCH * twist);
    let crystal = dielectric(CRYSTAL_PERMITTIVITY, 1.0);
    (rotor >> x(), rotor >> crystal.of(rotor << Bivector::slot()))
}

/// The reflected and transmitted powers of `x` light on glass behind each coating, per
/// coating and wavelength, at normal incidence.
pub fn antireflection(wavelengths: &[f64]) -> [Vec<(f64, f64)>; 3] {
    let incident = Ports::isotropic(INCIDENT_INDEX, t());
    let substrate = Ports::isotropic(SUBSTRATE_INDEX, t());
    core::array::from_fn(|k| {
        let index = COATING_INDICES[k];
        let coating = Spectrum::of(Medium::from_chi(isotropic(index), t()).generator);
        let thickness = OPTICAL_THICKNESSES[k] / index;
        wavelengths
            .iter()
            .map(|w| {
                let layer = coating.propagate(optical_distance(thickness, *w));
                let result = scatter(layer.of(substrate.outgoing), &incident);
                result.powers(along_x(), &incident, &substrate)
            })
            .collect()
    })
}

/// The layers of a helix of sense `twist` at normal incidence, from the entrance.
pub fn helix_layers(twist: f64) -> Vec<Spectrum> {
    (0..LAYERS)
        .map(|k| {
            let (_, chi) = helix((k as f64 + 0.5) * THICKNESS, twist);
            Spectrum::of(Medium::from_chi(chi, t()).generator)
        })
        .collect()
}

/// A stack in a surrounding medium at one wavelength.
pub fn stack_scattering(layers: &[Spectrum], wavelength: f64, ports: &Ports) -> Scattering {
    let distance = optical_distance(THICKNESS, wavelength);
    let propagation = compose(layers.iter().map(|layer| layer.propagate(distance)));
    scatter(propagation.of(ports.outgoing), ports)
}

/// The reflected and transmitted powers of each circular polarization on each helix,
/// `[polarization][helix][wavelength]`.
pub fn twisted(wavelengths: &[f64]) -> [[Vec<(f64, f64)>; 2]; 2] {
    let ports = Ports::isotropic(SURROUNDING_INDEX, t());
    let helices = TWIST_SIGNS.map(|twist| {
        let layers = helix_layers(twist);
        wavelengths
            .iter()
            .map(|w| stack_scattering(&layers, *w, &ports))
            .collect::<Vec<_>>()
    });
    TWIST_SIGNS.map(|sign| {
        let light = circular(sign);
        helices.each_ref().map(|results| {
            results
                .iter()
                .map(|r| r.powers(light, &ports, &ports))
                .collect()
        })
    })
}

/// The reflected power of `x` light on an axion interface, per jump: in `x`, in `y`, and
/// in total. Identical ordinary dielectric responses: all reflection comes from the jump.
pub fn axion(jumps: &[f64]) -> Vec<[f64; 3]> {
    let chi = isotropic(SURROUNDING_INDEX);
    let incident = Ports::isotropic(SURROUNDING_INDEX, t());
    let total = incident.power(along_x());
    jumps
        .iter()
        .map(|jump| {
            let medium = Medium::from_chi(chi + axion_response(*jump), t());
            let substrate = Ports::from_medium(&medium, SURROUNDING_INDEX, t());
            let reflected = scatter(substrate.outgoing, &incident)
                .reflection
                .of(along_x());
            let zero = C::real(0.0);
            let parallel = Polarization::new(reflected.e1(), zero);
            let perpendicular = Polarization::new(zero, reflected.e2());
            [parallel, perpendicular, reflected].map(|r| incident.power(r) / total)
        })
        .collect()
}

/// The incoming pulse's electric planes at `at` (nm along the line) for each case: the
/// band's carrier turning with the helix, against it, and their average, which does not
/// turn.
pub fn pulse(at: f64) -> [B; 3] {
    let envelope = (-((at - PULSE_START) / PULSE_WIDTH).powi(2) / 2.0).exp();
    let phase = optical_distance(SURROUNDING_INDEX * at, BAND_WAVELENGTH);
    let [with, against] = TWIST_SIGNS.map(|hand| about_line(hand * phase) >> x());
    [with, against, (with + against) * 0.5].map(|e| (e * envelope) ^ t())
}

/// The nodes' positions along the line, in nm.
pub fn positions() -> Vec<f64> {
    (0..CELLS).map(|i| i as f64 * SPACING).collect()
}

/// Whether a position lies in the slab.
pub fn in_slab(position: f64) -> bool {
    (SLAB_START..SLAB_END).contains(&position)
}

/// One case of the pulse run through the line.
pub struct PulseRun {
    /// The electric planes along the line at each frame.
    pub frames: Vec<Field<B>>,
    /// The energy at the start.
    pub before: f64,
    /// At the end, the energy in front of the slab and behind it.
    pub reflected: f64,
    pub transmitted: f64,
}

/// The pulse through the positively twisted helix, stepped in time, for each case.
pub fn time_domain() -> [PulseRun; 3] {
    let positions = positions();
    let surrounding = isotropic(SURROUNDING_INDEX);
    let nodes = positions
        .iter()
        .map(|p| {
            if in_slab(*p) {
                helix(p - SLAB_START, 1.0).1
            } else {
                surrounding
            }
        })
        .collect();
    let line = Line::from_chi(nodes, surrounding, SPACING);
    let interval = COURANT * SPACING;
    // A pulse already travelling towards +z: its magnetic planes are its electric planes
    // swept along z, times the index, half a node ahead and half a step earlier.
    let lead = SPACING / 2.0 + interval / (2.0 * SURROUNDING_INDEX);
    let steps = (DURATION / interval).round() as usize;
    let stride = steps / FRAMES;
    core::array::from_fn(|case| {
        let electric: Field<B> = positions.iter().map(|p| pulse(*p)[case]).collect();
        let magnetic = positions
            .iter()
            .map(|p| swept(pulse(p + lead)[case]) * SURROUNDING_INDEX)
            .collect();
        let before = line.energy(&electric, &magnetic).sum().s();
        let states: Vec<(Field<B>, Field<B>)> = line
            .evolve(electric, magnetic, interval)
            .take(steps)
            .skip(stride - 1)
            .step_by(stride)
            .collect();
        let (electric, magnetic) = states.last().expect("a frame");
        let energy = line.energy(electric, magnetic);
        let energy_where = |inside: fn(f64) -> bool| -> f64 {
            positions
                .iter()
                .zip(energy.iter())
                .filter(|(p, _)| inside(**p))
                .map(|(_, e)| e.s())
                .sum()
        };
        PulseRun {
            reflected: energy_where(|p| p < SLAB_START),
            transmitted: energy_where(|p| p >= SLAB_END),
            before,
            frames: states.into_iter().map(|(e, _)| e).collect(),
        }
    })
}

/// The spectra and the run at numga's sizes, computed once: the coatings' and the helix's
/// reflectances over the visible band, the axion interface's over its jumps, and the pulse.
pub struct Results {
    pub wavelengths: Vec<f64>,
    pub coatings: [Vec<(f64, f64)>; 3],
    pub helix: [[Vec<(f64, f64)>; 2]; 2],
    pub jumps: Vec<f64>,
    pub axion: Vec<[f64; 3]>,
    pub pulses: [PulseRun; 3],
}

pub fn results() -> &'static Results {
    static RESULTS: std::sync::OnceLock<Results> = std::sync::OnceLock::new();
    RESULTS.get_or_init(|| {
        let wavelengths = linspace(450.0, 850.0, 401);
        let jumps = linspace(-3.0, 3.0, 241);
        Results {
            coatings: antireflection(&wavelengths),
            helix: twisted(&wavelengths),
            axion: axion(&jumps),
            wavelengths,
            jumps,
            pulses: time_domain(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coating::interior;
    use crate::coating::tests::{ROUND_OFF, adjoint, close, complex};

    #[test]
    fn twisted_chi_selects_circular_polarization_and_remains_reciprocal() {
        // numga's windings: the negative twist first.
        let windings = [-1.0, 1.0];
        let wavelengths = [500.0, 660.0, 850.0];
        let ports = Ports::isotropic(SURROUNDING_INDEX, t());
        // [winding][wavelength] scattering, forwards and with the layers reversed.
        let scattering = |reverse: bool| -> Vec<Vec<Scattering>> {
            windings
                .iter()
                .map(|w| {
                    let mut layers = helix_layers(*w);
                    if reverse {
                        layers.reverse();
                    }
                    wavelengths
                        .iter()
                        .map(|l| stack_scattering(&layers, *l, &ports))
                        .collect()
                })
                .collect()
        };
        let (forward, backward) = (scattering(false), scattering(true));
        // [polarization][winding][wavelength]: x + iy, then x - iy.
        let reflectance = |p: usize, w: usize, l: usize| {
            forward[w][l].powers(circular([1.0, -1.0][p]), &ports, &ports)
        };
        for l in 0..wavelengths.len() {
            // Reversing the twist exchanges the selected circular polarization.
            assert!((reflectance(0, 0, l).0 - reflectance(1, 1, l).0).abs() < ROUND_OFF);
            assert!((reflectance(1, 0, l).0 - reflectance(0, 1, l).0).abs() < ROUND_OFF);
            for p in 0..2 {
                for w in 0..2 {
                    let (reflected, transmitted) = reflectance(p, w, l);
                    assert!((reflected + transmitted - 1.0).abs() < ROUND_OFF);
                }
            }
            for w in 0..2 {
                // Reciprocity transposes the amplitude map; it does not conjugate the phase.
                let reverse = backward[w][l].transmission;
                assert!(close(reverse, adjoint(forward[w][l].transmission)));
            }
        }
        assert!(reflectance(1, 0, 1).0 > 0.98);
        assert!(reflectance(0, 0, 1).0 < 0.01);
        for l in [0, 2] {
            for p in 0..2 {
                for w in 0..2 {
                    assert!(reflectance(p, w, l).0 < 0.2);
                }
            }
        }
    }

    #[test]
    fn selected_circular_wave_decays_through_the_twisted_reflector() {
        let wavelength = 660.0;
        let fractions = linspace(0.0, 1.0, 9);
        let ports = Ports::isotropic(SURROUNDING_INDEX, t());
        let media: Vec<Medium> = (0..LAYERS)
            .map(|k| Medium::from_chi(helix((k as f64 + 0.5) * THICKNESS, 1.0).1, t()))
            .collect();
        let thicknesses = [THICKNESS; LAYERS];
        let result = stack_scattering(&helix_layers(1.0), wavelength, &ports);
        // The mean intensity over the samples of a run of layers.
        let intensity = |layers: &[Vec<Bivector<(), C>>]| -> f64 {
            let samples = layers.iter().flatten();
            let count = layers.len() * fractions.len();
            samples
                .map(|field| {
                    let electric = complex(t()) | *field;
                    let (re, im) = (electric.map_coefs(|c| c.re), electric.map_coefs(|c| c.im));
                    -(re.scalar_product(re).s() + im.scalar_product(im).s())
                })
                .sum::<f64>()
                / count as f64
        };
        let penetration = [1.0, -1.0].map(|sign| {
            let exit_state = ports.outgoing.of(result.transmission.of(circular(sign)));
            let fields = interior(&media, &thicknesses, wavelength, exit_state, &fractions);
            let first_pitch = intensity(&fields[..SLICES_PER_TURN]);
            let last_pitch = intensity(&fields[LAYERS - SLICES_PER_TURN..]);
            last_pitch / first_pitch
        });
        assert!(penetration[0] < 0.05, "{penetration:?}");
        assert!(
            0.9 < penetration[1] && penetration[1] < 1.1,
            "{penetration:?}"
        );
    }

    /// numga's test and scenario check of the run: the leapfrog keeps the pulse's energy in
    /// every case, measured once the pulses have left the helix (a method-accuracy check), and
    /// the hand turning with the helix comes back while the other passes, as the reflectance
    /// band says.
    #[test]
    fn stepped_pulse_keeps_its_energy_and_returns_the_hand_turning_with_the_helix() {
        let runs = &results().pulses;
        for run in runs {
            assert!(((run.reflected + run.transmitted) / run.before - 1.0).abs() < 1e-3);
        }
        assert!(runs[WITH].reflected / runs[WITH].before > 0.98);
        assert!(runs[AGAINST].reflected / runs[AGAINST].before < 0.01);
    }

    /// numga's scenario check of the spectra: every one conserves energy.
    #[test]
    fn spectra_are_lossless() {
        let s = results();
        let lossless =
            |powers: &[(f64, f64)]| powers.iter().all(|(r, t)| (r + t - 1.0).abs() < ROUND_OFF);
        assert!(s.coatings.iter().all(|c| lossless(c)));
        assert!(s.helix.iter().flatten().all(|h| lossless(h)));
    }
}
