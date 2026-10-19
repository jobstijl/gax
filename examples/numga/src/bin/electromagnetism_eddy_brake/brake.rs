//! numga's `eddy_brake/core.py`, the resistive part: the currents a magnetic field induces in a
//! turning conducting disc, their braking torque and heat, and the disc braking under them.
//!
//! A disc turning with spin `ω` (a bivector) moves its material at `v = -ω · x`, and a magnetic
//! field `B` (a bivector) drives charges along `B · v`. Integrated along each mesh edge that is
//! the edge's electromotive force. Charge piles up until a potential on the vertices balances
//! the drive: the face fields, reconstructed from edge circulations by Whitney's forms, are
//! conducted by each face's material map, and the charge balance is the adjoint of the face
//! gradient applied to the face currents, `Gᵀ A σ (G φ - drive) = 0`, `A` the face areas (the
//! metric is Euclidean, so the adjoint is the coefficients' transpose). The same skew map
//! `x ↦ B · x` gives the force on the dual-edge fluxes, and the adjoint pairing makes the
//! mechanical power lost equal the Joule heat.

use gax::Unit;
use gax::field::{Field, SparseLu, SparseMap};
use gax::vga3d::{Rotor, Scalar, Vector};
use gax_numga_examples::mesh::{Bivector3, Mesh, Scalar3, Vector3};

/// A face's sheet conductance, `Vector <- Vector`.
pub type Conductivity = Vector<(Vector,)>;
/// A turn of the disc.
pub type Turn = Unit<Rotor>;

/// The plane the discs turn in, numga's `xy`.
pub const XY: Bivector3 = Bivector3::new(0.0, 0.0, 1.0);

/// The material law of conductances `principal` along `directions` and none across them: a sum
/// of dyads, each conducting along one direction.
pub fn dyads(directions: &[Vector3], principal: &[f64]) -> Conductivity {
    directions
        .iter()
        .zip(principal)
        .map(|(d, g)| (*d * (*d | Vector::slot())) * *g)
        .sum()
}

/// Sheet conductance of radial and circumferential fibres in the annulus between two radii,
/// `principal` the conductance along the radius and around it; outside the annulus the disc
/// conducts `mean` in every direction.
pub fn fibre_conductivity(
    mesh: &Mesh,
    principal: [f64; 2],
    inner: f64,
    outer: f64,
    mean: f64,
) -> Field<Conductivity> {
    mesh.face_centers().map(|centre| {
        let radial = centre.normalized();
        let distance = centre.norm();
        let in_band = distance >= inner && distance <= outer;
        let local = principal.map(|p| if in_band { p } else { mean });
        dyads(&[radial, XY | radial], &local)
    })
}

/// A magnet fixed in the world: its centre, its plane, and its footprint's strength and width.
#[derive(Clone, Copy, Debug)]
pub struct Magnet {
    pub centre: Vector3,
    pub plane: Bivector3,
    pub strength: f64,
    pub width: f64,
}

impl Magnet {
    /// The magnetic footprint at the positions: its strength falls as a Gaussian in the distance
    /// from the centre.
    pub fn field(&self, positions: &Field<Vector3>) -> Field<Bivector3> {
        let spread = 2.0 * self.width * self.width;
        positions.map(|p| {
            let squared = (p - self.centre).norm_squared();
            self.plane * (self.strength * (-squared / spread).exp())
        })
    }

    /// The footprint at the material points `positions` of a body turned by `turn`, pulled back
    /// into the body's frame.
    pub fn in_body(&self, positions: &Field<Vector3>, turn: Turn) -> Field<Bivector3> {
        self.field(&positions.map(|p| turn >> p)).map(|b| turn << b)
    }
}

/// The potential, the edge voltages and dual-edge fluxes, and their mechanical and thermal
/// response.
#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))] // the tests check what the animation does not draw
pub struct Response {
    /// `[V]`: the electric potential, zero at the first vertex.
    pub potential: Field<Scalar3>,
    /// `[E]`: the voltage along each edge.
    pub voltage: Field<Scalar3>,
    /// `[E]`: the current across each dual edge.
    pub flux: Field<Scalar3>,
    /// `[F]`: the sheet current on each face.
    pub current: Field<Vector3>,
    /// `[F]`: the Joule heating of each face.
    pub heating: Field<Scalar3>,
    /// The magnetic torque on the disc.
    pub torque: Bivector3,
}

/// A conducting disc in its material frame: its mesh, its faces' conductance, and its charge
/// balance, factored once (the disc conducts alike however it is turned).
pub struct Conductor<'a> {
    pub mesh: &'a Mesh,
    pub conductivity: &'a Field<Conductivity>,
    areas: Field<Scalar3>,
    /// `[F, E]`: Whitney's reconstruction of a face vector from edge circulations.
    reconstruction: SparseMap<Vector<(Scalar,)>>,
    /// `[E, V]`: the difference along each edge.
    derivative: SparseMap<Scalar<(Scalar,)>>,
    /// `[F, V]`: the face gradient of a potential.
    gradient: SparseMap<Vector<(Scalar,)>>,
    /// `[F, F]`: each face's conductance times its area.
    material: SparseMap<Conductivity>,
    /// The charge balance `Gᵀ A σ G`, its free constant pinned.
    balance: SparseLu<Scalar, f64>,
}

impl<'a> Conductor<'a> {
    pub fn new(mesh: &'a Mesh, conductivity: &'a Field<Conductivity>) -> Self {
        let areas = mesh.triangle_areas();
        let reconstruction = mesh.reconstruction();
        let derivative = &mesh.d0() * Scalar::slot();
        let gradient = reconstruction.compose(&derivative);
        let weighted = conductivity.zip_with(&areas, |c, a| c * a.s());
        let material = SparseMap::from_diagonal(&weighted);
        let balance = gradient.transpose().compose(&material.compose(&gradient));
        // The adjoint sums the outgoing currents; no exterior flux is an insulating rim. The
        // first vertex's potential is pinned to zero, fixing the free constant: no edge voltage
        // or current changes, and it is no contact to the disc.
        let first = balance.couplings().filter(|&(r, c, _)| r == 0 && c == 0);
        let pin: Scalar<(Scalar,)> = first.map(|(_, _, cell)| cell).sum();
        let n = mesh.vertices.len();
        let gauge = SparseMap::new(vec![pin], vec![0], vec![0], (n, n));
        let balance = (&balance + &gauge).factor().expect("a pinned balance");
        Conductor {
            mesh,
            conductivity,
            areas,
            reconstruction,
            derivative,
            gradient,
            material,
            balance,
        }
    }

    /// The electric potential that makes the induced currents conserve charge, under the
    /// magnetic field `[E]` on the edges, turning with `spin`, and what follows from it. The
    /// field the currents make themselves is neglected.
    pub fn respond(&self, magnetic: &Field<Bivector3>, spin: Bivector3) -> Response {
        let mesh = self.mesh;
        let velocity = mesh.edge_midpoints().map(|p| -(spin | p));
        let lorentz = magnetic.map(|b| b | Vector::slot());
        // The motion's electrical drive, integrated along each oriented edge.
        let pushed = lorentz.zip_with(&velocity, |l, v| l.of(v));
        let emf = pushed.zip_with(&mesh.edge_vectors(), |f, e| f | e);
        // Whitney vectors reconstruct edge circulations at each face's centroid, where the
        // conductivity acts on the drive.
        let drive = self.reconstruction.of(&emf);
        let source = self.gradient.transpose().of(&self.material.of(&drive));
        let potential = self.balance.solve(&source);
        let voltage = &emf - &self.derivative.of(&potential);
        let electric = self.reconstruction.of(&voltage);
        let current = self.conductivity.zip_with(&electric, |c, e| c.of(e));
        // The reconstruction's adjoint returns the conserved dual-edge fluxes, and the same skew
        // map gives the magnetic force on them.
        let sheet = current.zip_with(&self.areas, |j, a| j * a.s());
        let flux = self.reconstruction.transpose().of(&sheet);
        let skew = mesh.edge_vectors().zip_with(&lorentz, |e, l| l.of(e));
        let force = flux.zip_with(&skew, |f, l| l * f.s());
        let torque = mesh.edge_midpoints().zip_with(&force, |p, f| p ^ f).sum();
        let power = electric.zip_with(&current, |e, j| e | j);
        let heating = self.areas.zip_with(&power, |a, p| a * p);
        Response {
            potential,
            voltage,
            flux,
            current,
            heating,
            torque,
        }
    }
}

/// numga's `solve`: the response of a disc conducting by `conductivity`, under the magnetic
/// field `[E]` on its edges, turning with `spin`.
#[cfg_attr(not(test), allow(dead_code))] // the tests solve single discs
pub fn solve(
    mesh: &Mesh,
    magnetic: &Field<Bivector3>,
    spin: Bivector3,
    conductivity: &Field<Conductivity>,
) -> Response {
    Conductor::new(mesh, conductivity).respond(magnetic, spin)
}

/// The orientation and spin after a step that turns the body at its midpoint spin: the mean
/// spin of the step sets the turn, and the spin changes by twice its departure from it.
pub fn turned(
    orientation: Turn,
    spin: Bivector3,
    midpoint_spin: Bivector3,
    step: f64,
) -> (Turn, Bivector3) {
    let turn = (midpoint_spin * (-step / 2.0)).exp();
    (turn * orientation, midpoint_spin * 2.0 - spin)
}

/// A rotating material mesh: its turn and spin, its vertices in the world, the Joule energy
/// gathered on each material face, and its sheet current in the body's frame.
#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))] // the tests check the turned vertices
pub struct Motion {
    pub orientation: Turn,
    pub spin: Bivector3,
    pub vertices: Field<Vector3>,
    pub heat: Field<Scalar3>,
    pub current: Field<Vector3>,
}

/// A disc braking under a magnet fixed in the world: the disc's conduction, and the moment of
/// inertia of it and its rotor about the magnet's plane.
pub struct Brake<'a> {
    pub conductor: Conductor<'a>,
    pub magnet: Magnet,
    pub inertia: f64,
}

impl Brake<'_> {
    /// The response of the disc turned by `orientation` and spinning at `spin`: the fixed magnet
    /// sampled on the moving edges and pulled back into the material frame, where incidence,
    /// reconstruction and conductivity stay.
    pub fn response(&self, orientation: Turn, spin: Bivector3) -> Response {
        let edges = self.conductor.mesh.edge_midpoints();
        let magnetic = self.magnet.in_body(&edges, orientation);
        self.conductor.respond(&magnetic, orientation << spin)
    }

    /// The state at a turn and spin, with its current.
    fn state(&self, orientation: Turn, spin: Bivector3, heat: Field<Scalar3>) -> Motion {
        Motion {
            orientation,
            spin,
            vertices: self.conductor.mesh.vertices.map(|v| orientation >> v),
            heat,
            current: self.response(orientation, spin).current,
        }
    }

    /// The disc turned under the magnet for `steps` steps, the generated heat gathered on its
    /// faces: every state from the first. Heat stays with the material, without diffusion or
    /// cooling; the currents equilibrate instantly. A predicted midpoint gives second-order
    /// drag, and the midpoint's angular momentum makes the heat gained equal the kinetic energy
    /// lost.
    pub fn braking(
        &self,
        orientation: Turn,
        spin: Bivector3,
        heat: Field<Scalar3>,
        step: f64,
        steps: usize,
    ) -> Vec<Motion> {
        let plane = self.magnet.plane;
        let mut states = vec![self.state(orientation, spin, heat)];
        for _ in 0..steps {
            let last = states.last().expect("a state");
            let (orientation, spin) = (last.orientation, last.spin);
            // Predict where the material is halfway through the step and solve its current at
            // unit spin: the response is linear in the spin, and gives it at any spin.
            let midpoint = (spin * (-step / 4.0)).exp() * orientation;
            let unit = self.response(midpoint, plane);
            let drag = ((midpoint << plane) | unit.torque).s();
            // The same midpoint spin sets the mechanical loss and every face's heat gain; the
            // faces keep their identities as the mesh turns, so the heat moves with them.
            let midpoint_spin = spin * (1.0 / (1.0 + drag * step / (2.0 * self.inertia)));
            let gained = unit.heating.scaled(step * midpoint_spin.norm_squared());
            let heat = &last.heat + &gained;
            let (orientation, spin) = turned(orientation, spin, midpoint_spin, step);
            states.push(self.state(orientation, spin, heat));
        }
        states
    }
}
