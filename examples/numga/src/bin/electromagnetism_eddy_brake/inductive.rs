//! numga's `eddy_brake/core.py`, the inductive part: closed current patterns with a memory, the
//! magnetic energy of their own field, and a disc spun up and braked with them.
//!
//! A current that stays in a disc with an insulating rim is closed: the turned gradient of a
//! streamfunction that is zero on the rim, `J = n · ∇ψ` per face (`n` the face's plane), so the
//! normal flux across every interior edge is continuous. The resistance is the Joule form of
//! such currents, `Cᵀ A ρ C` with `ρ` each face's conductance inverted by least squares (no
//! conduction across the sheet), and the inductance the magnetic energy of their field,
//! `Cᵀ M C` with `M` the face-to-face Coulomb integral of a planar sheet. Their generalized
//! eigenvectors are the current patterns orthonormal in magnetic energy, and the eigenvalues
//! their resistive decay rates.
//!
//! The magnet drives each pattern by the work rate of its motional field on the pattern's
//! current. Over one revolution that drive is periodic in the disc's angle, a short Fourier
//! series, and the coupled midpoint steps of the pattern amplitudes and the rotation keep the
//! rotational, magnetic and thermal energies equal to the motor's work.

use super::brake::{Conductivity, Magnet, Turn, turned};
use core::ops::{Add, Mul};
use gax::field::{Field, SparseMap};
use gax::vga3d::{Scalar, Vector};
use gax_numga_examples::mesh::{Bivector3, Mesh, Scalar3, Vector3};
use gax_numga_examples::signal::phasor;

/// A field map with scalar cells, multiplying the values.
pub type Scalars = SparseMap<Scalar3>;

/// The interior vertices, off the rim.
pub fn interior(mesh: &Mesh) -> Vec<usize> {
    let rim = mesh.boundary_vertices();
    (0..rim.len()).filter(|&v| !rim[v]).collect()
}

/// `[V, interior]`: each interior vertex's value placed on the mesh, zero on the rim.
pub fn embedder(mesh: &Mesh, interior: &[usize]) -> Scalars {
    let n = interior.len();
    let ones = vec![Scalar3::new(1.0); n];
    SparseMap::new(
        ones,
        interior.to_vec(),
        (0..n).collect(),
        (mesh.vertices.len(), n),
    )
}

/// `[F, interior]`, cells `Vector <- Scalar`: the closed current of a streamfunction zero on the
/// rim, its face gradient turned in the face's plane.
pub fn closed_currents(mesh: &Mesh, embedder: &Scalars) -> SparseMap<Vector<(Scalar,), f64>> {
    let differences = &(&mesh.d0() * embedder) * Scalar::slot();
    let gradient = mesh.reconstruction().compose(&differences);
    let turn = mesh.face_planes().map(|plane| plane | Vector::slot());
    SparseMap::from_diagonal(&turn).compose(&gradient)
}

/// Each face's conductance inverted by least squares: the resistance on tangential currents,
/// without inventing a normal component.
pub fn resistivity(conductivity: &Field<Conductivity>) -> Field<Conductivity> {
    conductivity.map(|c| c.pinv())
}

/// The Coulomb integrals between face currents on a planar sheet, `[F, F]`, before reciprocal
/// symmetrization: each source triangle's potential integrated in closed form, the target
/// triangle by three quadrature points, and each face's finite self interaction exactly.
/// Thickness enters the sheet conductance, not a cutoff in the kernel.
pub fn coulomb_integrals(mesh: &Mesh, permeability: f64) -> Scalars {
    let corners = mesh.corners();
    let edges = [0, 1, 2].map(|k| &corners[(k + 1) % 3] - &corners[k]);
    let lengths = edges.each_ref().map(|e| e.map(|e| Scalar3::new(e.norm())));
    let planes = mesh.face_planes();
    // Each edge's outward normal in its face's plane.
    let outward = [0, 1, 2].map(|k| {
        let unit = edges[k].zip_with(&lengths[k], |e, l| e * (1.0 / l.s()));
        planes.zip_with(&unit, |p, e| p | e)
    });
    let (areas, centres) = (mesh.triangle_areas(), mesh.face_centers());
    let faces = mesh.faces.len();
    let scale = permeability / (4.0 * core::f64::consts::PI);
    // A source face's potential at a point: the sum over its edges of the height above the edge
    // times the logarithm of its endpoint distances.
    let potential = |f: usize, point: Vector3| -> f64 {
        let distance = |k: usize| (corners[k].site(f) - point).norm();
        (0..3)
            .map(|k| {
                let (length, ends) = (lengths[k].site(f).s(), distance(k) + distance((k + 1) % 3));
                let height = ((corners[k].site(f) - point) | outward[k].site(f)).s();
                height * ((ends + length) / (ends - length)).ln()
            })
            .sum()
    };
    let mut cells = Vec::with_capacity(faces * faces);
    for t in 0..faces {
        // Three points at barycentric weights two thirds, a sixth, a sixth: their equal weights
        // integrate a quadratic exactly.
        let samples = [0, 1, 2].map(|k| (corners[k].site(t) + centres.site(t)) * 0.5);
        for f in 0..faces {
            let integral = if f == t {
                self_integral(&lengths, &areas, t)
            } else {
                let mean = samples.iter().map(|s| potential(f, *s)).sum::<f64>() / 3.0;
                areas.site(t).s() * mean
            };
            cells.push(Scalar3::new(integral * scale));
        }
    }
    dense((faces, faces), cells)
}

/// The field map of `shape` with a cell for every pair of sites, `cells` row by row.
fn dense<C: Copy>(shape: (usize, usize), cells: Vec<C>) -> SparseMap<C> {
    let (outputs, inputs) = shape;
    let rows = (0..outputs)
        .flat_map(|r| core::iter::repeat_n(r, inputs))
        .collect();
    let cols = (0..outputs).flat_map(|_| 0..inputs).collect();
    SparseMap::new(cells, rows, cols, shape)
}

/// A triangle's Coulomb integral with itself, finite despite the kernel's singularity:
/// `4/3 A² Σ ln(s / (s - l)) / l` over its edges, `s` the semiperimeter.
fn self_integral(lengths: &[Field<Scalar3>; 3], areas: &Field<Scalar3>, f: usize) -> f64 {
    let sides = lengths.each_ref().map(|l| l.site(f).s());
    let half = sides.iter().sum::<f64>() / 2.0;
    let logs: f64 = sides.iter().map(|l| (half / (half - l)).ln() / l).sum();
    4.0 / 3.0 * areas.site(f).s() * areas.site(f).s() * logs
}

/// The magnetic coupling between face currents on a planar sheet, `[F, F]`: the Coulomb
/// integrals, made reciprocal by their symmetric part.
#[cfg_attr(not(test), allow(dead_code))] // the tests check the magnetic energy with it
pub fn magnetic_coupling(mesh: &Mesh, permeability: f64) -> Scalars {
    symmetric(&coulomb_integrals(mesh, permeability))
}

/// The symmetric part `(m + mᵀ) / 2` of a field map with scalar cells (its cells their own
/// transposes).
fn symmetric<C>(map: &SparseMap<C>) -> SparseMap<C>
where
    C: Copy + Add<Output = C> + Mul<f64, Output = C> + gax::Reverse<Output = C>,
{
    (map + &map.reverse()).map_cells(|c| c * 0.5)
}

/// Closed current patterns, orthonormal in magnetic energy, and their resistive decay rates.
pub struct CurrentModes {
    /// `[modes]`: each pattern's decay rate, inverse seconds.
    pub decay: Field<Scalar3>,
    /// `[modes, interior]`: each pattern's streamfunction on the interior, read as the map onto
    /// its amplitude.
    pub projection: Scalars,
    /// `[F, interior]`: the closed current of an interior streamfunction.
    pub current: SparseMap<Vector<(Scalar,), f64>>,
    /// `[V, interior]`: an interior streamfunction on the mesh.
    pub embedder: Scalars,
    /// `[F]`: each face's sheet resistance.
    pub resistivity: Field<Conductivity>,
}

/// The `count` slowest inductive current patterns of a conducting disc with an insulating rim.
pub fn current_modes(
    mesh: &Mesh,
    conductivity: &Field<Conductivity>,
    permeability: f64,
    count: usize,
) -> CurrentModes {
    let embedder = embedder(mesh, &interior(mesh));
    let current = closed_currents(mesh, &embedder);
    let resistivity = resistivity(conductivity);
    let areas = mesh.triangle_areas();
    let weighted = SparseMap::from_diagonal(&resistivity.zip_with(&areas, |r, a| r * a.s()));
    let adjoint = current.transpose();
    let resistance = adjoint.compose(&weighted.compose(&current));
    let pulled = &adjoint * &coulomb_integrals(mesh, permeability);
    let inductance = symmetric(&pulled.compose(&current));
    // The eigenfields are orthonormal in magnetic energy; their eigenvalues are decay rates.
    let (decay, modes) = resistance.eigh(&inductance, count);
    let cells = modes.iter().flat_map(|m| m.iter()).collect();
    CurrentModes {
        decay: decay.into_iter().map(Scalar3::new).collect(),
        projection: dense((count, embedder.shape().1), cells),
        current,
        embedder,
        resistivity,
    }
}

/// Two fields over the patterns combined pattern by pattern.
fn per_mode(a: &Field<Scalar3>, b: &Field<Scalar3>, f: impl Fn(f64, f64) -> f64) -> Field<Scalar3> {
    a.zip_with(b, |x, y| Scalar3::new(f(x.s(), y.s())))
}

impl CurrentModes {
    /// The work rate of a face electric field `[F]` on each pattern's current, `[modes]`.
    pub fn drive(&self, mesh: &Mesh, electric: &Field<Vector3>) -> Field<Scalar3> {
        let sheet = electric.zip_with(&mesh.triangle_areas(), |e, a| e * a.s());
        &self.projection * &self.current.transpose().of(&sheet)
    }

    /// The streamfunction on the interior of mode amplitudes `[modes]`.
    fn interior_stream(&self, amplitudes: &Field<Scalar3>) -> Field<Scalar3> {
        &self.projection.reverse() * amplitudes
    }

    /// The sheet current `[F]` of mode amplitudes `[modes]`.
    pub fn currents(&self, amplitudes: &Field<Scalar3>) -> Field<Vector3> {
        self.current.of(&self.interior_stream(amplitudes))
    }

    /// The streamfunction `[V]` of mode amplitudes `[modes]`.
    pub fn streamfunctions(&self, amplitudes: &Field<Scalar3>) -> Field<Scalar3> {
        &self.embedder * &self.interior_stream(amplitudes)
    }

    /// The heat `[F]` deposited over substeps of length `interval` at their midpoint amplitudes,
    /// each face's Joule heating of the currents summed: the cross terms of the patterns
    /// included, without a dense covariance.
    pub fn deposited_heat(
        &self,
        mesh: &Mesh,
        interval: f64,
        midpoints: &[Field<Scalar3>],
    ) -> Field<Scalar3> {
        let heating = |amplitudes: &Field<Scalar3>| {
            let current = self.currents(amplitudes);
            let resisted = self.resistivity.zip_with(&current, |r, j| r.of(j));
            current.zip_with(&resisted, |j, e| j | e)
        };
        let total = midpoints
            .iter()
            .map(heating)
            .reduce(|a, b| &a + &b)
            .expect("a substep");
        mesh.triangle_areas()
            .zip_with(&total, |a, h| a * h * interval)
    }

    /// The moving current and heat of the disc from its material amplitudes.
    pub fn state(
        &self,
        motion: Rotation,
        heat: Field<Scalar3>,
        amplitudes: Field<Scalar3>,
        ledger: Ledger,
    ) -> InductiveMotion {
        InductiveMotion {
            current: self.currents(&amplitudes),
            streamfunction: self.streamfunctions(&amplitudes),
            magnetic_energy: amplitudes.dot(&amplitudes) / 2.0,
            motion,
            heat,
            ledger,
        }
    }
}

/// The magnetic drive projected onto the patterns over one revolution: its mean and its
/// harmonics' cosine and sine parts, each `[modes]`.
pub struct PeriodicDrive {
    pub mean: Field<Scalar3>,
    pub cosine: Vec<Field<Scalar3>>,
    pub sine: Vec<Field<Scalar3>>,
}

impl PeriodicDrive {
    /// The drive with the disc turned by `angle` from where it was sampled.
    pub fn at(&self, angle: f64) -> Field<Scalar3> {
        let harmonics = self.cosine.iter().zip(&self.sine).enumerate();
        harmonics.fold(self.mean.clone(), |sum, (h, (c, s))| {
            let turn = phasor(angle * (h + 1) as f64);
            &(&sum + &c.scaled(turn.e20())) + &s.scaled(turn.e01())
        })
    }
}

/// The electric field the magnet makes in the material of a disc turned by `turn` and turning
/// at unit speed in the magnet's plane, at the face centres, in the material frame.
pub fn motional(mesh: &Mesh, magnet: &Magnet, turn: Turn) -> Field<Vector3> {
    let centres = mesh.face_centers();
    let body_plane = turn << magnet.plane;
    let velocity = centres.map(|p| -(body_plane | p));
    let magnetic = magnet.in_body(&centres, turn);
    magnetic.zip_with(&velocity, |b, v| b | v)
}

/// One revolution of the magnetic forcing on a disc turned by `orientation`, projected onto the
/// patterns as `count` real harmonics.
pub fn periodic_drive(
    mesh: &Mesh,
    modes: &CurrentModes,
    magnet: &Magnet,
    orientation: Turn,
    count: usize,
) -> PeriodicDrive {
    let samples = 2 * count + 1;
    let angle = |k: usize| core::f64::consts::TAU * k as f64 / samples as f64;
    let values: Vec<Field<Scalar3>> = (0..samples)
        .map(|k| {
            let turn = (magnet.plane * (-angle(k) / 2.0)).exp() * orientation;
            modes.drive(mesh, &motional(mesh, magnet, turn))
        })
        .collect();
    let average = |weight: &dyn Fn(usize) -> f64| {
        let weighted = (0..samples).map(|k| values[k].scaled(weight(k) / samples as f64));
        weighted.reduce(|a, b| &a + &b).expect("samples")
    };
    PeriodicDrive {
        mean: average(&|_| 1.0),
        cosine: (1..=count)
            .map(|h| average(&|k| 2.0 * phasor(angle(k) * h as f64).e20()))
            .collect(),
        sine: (1..=count)
            .map(|h| average(&|k| 2.0 * phasor(angle(k) * h as f64).e01()))
            .collect(),
    }
}

/// The disc's turn and spin.
#[derive(Clone, Copy, Debug)]
pub struct Rotation {
    pub orientation: Turn,
    pub spin: Bivector3,
}

/// The energy accounts: the modal dissipation accumulated, and the motor's work.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ledger {
    pub thermal_energy: f64,
    pub motor_work: f64,
}

/// The disc with inductive currents: its rotation, the heat on each material
/// face, its body-frame current and streamfunction, the energy in the current's magnetic field,
/// and the ledger.
#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))] // the tests check the closure and the energy accounts
pub struct InductiveMotion {
    pub motion: Rotation,
    pub heat: Field<Scalar3>,
    pub current: Field<Vector3>,
    pub streamfunction: Field<Scalar3>,
    pub magnetic_energy: f64,
    pub ledger: Ledger,
}

/// A disc whose currents keep a memory: its mesh, its current patterns and their periodic drive
/// by the magnet, the plane it turns in, and its moment of inertia.
pub struct InductiveDisc<'a> {
    pub mesh: &'a Mesh,
    pub modes: &'a CurrentModes,
    pub forcing: &'a PeriodicDrive,
    pub plane: Bivector3,
    pub inertia: f64,
}

impl InductiveDisc<'_> {
    /// The current memory, rotation and material heat advanced by coupled midpoint steps, from
    /// material amplitudes `[modes]`, under the motor's torque `[steps][substeps]`: every state
    /// from the first. The substeps resolve the electrical relaxation between states. Thermal
    /// diffusion, displacement current and variation through the sheet are left out.
    pub fn braking(
        &self,
        start: Rotation,
        heat: Field<Scalar3>,
        amplitudes: Field<Scalar3>,
        motor_torque: &[Vec<Bivector3>],
        step: f64,
    ) -> Vec<InductiveMotion> {
        let substeps = motor_torque.first().map_or(1, Vec::len);
        let interval = step / substeps as f64;
        let decay = &self.modes.decay;
        let damping = decay.map(|d| Scalar3::new(1.0 + interval * d.s() / 2.0));
        let lever = interval / (2.0 * self.inertia);
        let mut ledger = Ledger {
            thermal_energy: heat.sum().s(),
            motor_work: 0.0,
        };
        let (mut motion, mut heat, mut amplitudes, mut angle) = (start, heat, amplitudes, 0.0);
        let mut states = vec![
            self.modes
                .state(motion, heat.clone(), amplitudes.clone(), ledger),
        ];
        for torques in motor_torque {
            let mut midpoints = Vec::with_capacity(torques.len());
            for torque in torques {
                let speed = -(self.plane | motion.spin).s();
                let motor = -(self.plane | *torque).s();
                let drive = self.forcing.at(angle + speed * interval / 2.0);
                let free = per_mode(&amplitudes, &damping, |a, d| a / d);
                let driven = per_mode(&drive, &damping, |f, d| interval / 2.0 * f / d);
                // The midpoint currents eliminated from the angular momentum, then both states
                // recovered together: the energy returning from the magnetic field included.
                let pushed = speed + lever * (motor - drive.dot(&free));
                let midpoint_speed = pushed / (1.0 + lever * drive.dot(&driven));
                let midpoint = &free + &driven.scaled(midpoint_speed);
                // The modal damping gives the total heat exactly.
                let dissipation = per_mode(decay, &midpoint, |d, m| d * m * m).sum().s();
                ledger.thermal_energy += interval * dissipation;
                ledger.motor_work += interval * motor * midpoint_speed;
                amplitudes = &midpoint.scaled(2.0) - &amplitudes;
                angle += interval * midpoint_speed;
                let midpoint_spin = self.plane * midpoint_speed;
                let (orientation, spin) =
                    turned(motion.orientation, motion.spin, midpoint_spin, interval);
                motion = Rotation { orientation, spin };
                midpoints.push(midpoint);
            }
            heat = &heat + &self.modes.deposited_heat(self.mesh, interval, &midpoints);
            states.push(
                self.modes
                    .state(motion, heat.clone(), amplitudes.clone(), ledger),
            );
        }
        states
    }
}
