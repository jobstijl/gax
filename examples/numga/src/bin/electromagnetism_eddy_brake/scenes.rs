//! numga's `eddy_brake/scenarios.py`: rotating conducting discs, isotropic and with radial and
//! circumferential fibres of the same mean conductivity, braking resistively, and spun up by a
//! motor pulse with inductive current memory.

use super::brake::{Brake, Conductivity, Conductor, Magnet, Motion, XY, fibre_conductivity};
use super::inductive::{InductiveDisc, InductiveMotion, Rotation, current_modes, periodic_drive};
use gax::field::Field;
use gax_numga_examples::mesh::{Bivector3, Mesh, Scalar3, Vector3};
use gax_numga_examples::signal::wave;

/// m.
pub const RADIUS: f64 = 0.1;
pub const THICKNESS: f64 = 0.001;
/// S/m.
pub const CONDUCTIVITY: f64 = 5.8e7;
/// The isotropic, radial-fibre and circumferential-fibre discs: the same mean conductivity in
/// every case, redistributed between the two directions (along the radius, around it).
pub const MATERIALS: [(&str, [f64; 2]); 3] = [
    ("ISOTROPIC", [1.0, 1.0]),
    ("RADIAL FIBRES", [1.9, 0.1]),
    ("CIRCUMFERENTIAL FIBRES", [0.1, 1.9]),
];
/// rad/s.
#[cfg_attr(not(test), allow(dead_code))] // the steady scene, which only the tests solve
pub const ANGULAR_SPEED: f64 = 1.0;
/// T.
pub const FIELD_STRENGTH: f64 = 0.2;
/// m, the Gaussian's standard deviation.
pub const FIELD_WIDTH: f64 = 0.02;
/// m, along x.
pub const MAGNET_OFFSET: f64 = 0.055;
pub const DIVISIONS: usize = 16;
/// kg m², the disc and its rotor.
pub const INERTIA: f64 = 0.005;
/// rad/s.
pub const INITIAL_SPEED: f64 = 2.0;
/// s.
pub const ELAPSED: f64 = 12.0;
pub const STEPS: usize = 120;
pub const STEP: f64 = ELAPSED / STEPS as f64;
pub const FIBRE_INNER_RADIUS: f64 = 0.2 * RADIUS;
pub const FIBRE_OUTER_RADIUS: f64 = 0.85 * RADIUS;
pub const MATERIAL_RADII: [f64; 3] = [FIBRE_INNER_RADIUS, FIBRE_OUTER_RADIUS, RADIUS];
/// H/m, a nonmagnetic conductor in free space.
pub const PERMEABILITY: f64 = 4.0 * core::f64::consts::PI * 1e-7;
pub const CURRENT_MODES: usize = 63;
pub const DRIVE_HARMONICS: usize = 16;
/// T.
pub const INDUCTIVE_FIELD_STRENGTH: f64 = 1.0;
/// rad/s, the unloaded motor pulse's top speed.
pub const INDUCTIVE_TARGET_SPEED: f64 = 600.0;
/// s.
pub const SPINUP_TIME: f64 = 0.012;
pub const INDUCTIVE_ELAPSED: f64 = 0.1;
pub const INDUCTIVE_STEPS: usize = 120;
pub const INDUCTIVE_STEP: f64 = INDUCTIVE_ELAPSED / INDUCTIVE_STEPS as f64;
pub const ELECTRICAL_SUBSTEPS: usize = 8;

/// The magnet of the scenes, at `strength`.
pub fn magnet(strength: f64) -> Magnet {
    Magnet {
        centre: Vector3::new(MAGNET_OFFSET, 0.0, 0.0),
        plane: XY,
        strength,
        width: FIELD_WIDTH,
    }
}

/// The disc mesh, its rings on the fibres' edges, and the three materials' conductances, each
/// with the same mean: fibres in the annulus, the centre and rim conducting alike both ways.
pub fn materials() -> (Mesh, [Field<Conductivity>; 3]) {
    let mesh = Mesh::concentric_disk(&MATERIAL_RADII, DIVISIONS);
    let sheet = THICKNESS * CONDUCTIVITY;
    let conductivity = MATERIALS.map(|(_, weights)| {
        let principal = weights.map(|w| w * sheet);
        fibre_conductivity(
            &mesh,
            principal,
            FIBRE_INNER_RADIUS,
            FIBRE_OUTER_RADIUS,
            sheet,
        )
    });
    (mesh, conductivity)
}

/// No heat on any face.
pub fn cold(mesh: &Mesh) -> Field<Scalar3> {
    Field::from_fn(mesh.faces.len(), |_| Scalar3::new(0.0))
}

/// The disc unturned, turning at `speed` in the magnet's plane.
pub fn released(speed: f64) -> Rotation {
    let orientation = (XY * 0.0).exp();
    Rotation {
        orientation,
        spin: XY * speed,
    }
}

/// A material disc released at the initial speed, braking resistively and keeping its Joule
/// heat on its turning faces.
pub fn materials_braking(mesh: &Mesh, conductivity: &Field<Conductivity>) -> Vec<Motion> {
    let brake = Brake {
        conductor: Conductor::new(mesh, conductivity),
        magnet: magnet(FIELD_STRENGTH),
        inertia: INERTIA,
    };
    let start = released(INITIAL_SPEED);
    brake.braking(start.orientation, start.spin, cold(mesh), STEP, STEPS)
}

/// The motor's torque `[steps][substeps]`: a half-sine pulse that accelerates the unloaded
/// rotor to the target speed, then releases it to magnetic braking.
pub fn motor_pulse() -> Vec<Vec<Bivector3>> {
    let interval = INDUCTIVE_STEP / ELECTRICAL_SUBSTEPS as f64;
    let peak = INDUCTIVE_TARGET_SPEED * core::f64::consts::PI / (2.0 * SPINUP_TIME);
    let torque = |k: usize| {
        let time = (k as f64 + 0.5) * interval;
        let accelerating = time < SPINUP_TIME;
        let phase = core::f64::consts::PI * (time / SPINUP_TIME).min(1.0);
        let acceleration = if accelerating {
            peak * wave(phase)
        } else {
            0.0
        };
        XY * (INERTIA * acceleration)
    };
    (0..INDUCTIVE_STEPS)
        .map(|s| {
            (0..ELECTRICAL_SUBSTEPS)
                .map(|k| torque(s * ELECTRICAL_SUBSTEPS + k))
                .collect()
        })
        .collect()
}

/// A material disc at rest with no current, spun up by the motor pulse under the strong magnet:
/// current build-up and magnetic braking with inductive memory.
pub fn inductive_material(mesh: &Mesh, conductivity: &Field<Conductivity>) -> Vec<InductiveMotion> {
    let modes = current_modes(mesh, conductivity, PERMEABILITY, CURRENT_MODES);
    let start = released(0.0);
    let strong = magnet(INDUCTIVE_FIELD_STRENGTH);
    let forcing = periodic_drive(mesh, &modes, &strong, start.orientation, DRIVE_HARMONICS);
    let disc = InductiveDisc {
        mesh,
        modes: &modes,
        forcing: &forcing,
        plane: XY,
        inertia: INERTIA,
    };
    let silent = modes.decay.map(|_| Scalar3::new(0.0));
    disc.braking(start, cold(mesh), silent, &motor_pulse(), INDUCTIVE_STEP)
}
