//! numga's scenes: a PZT-5A bimorph, two layers poled against each other, 30 mm long and 1 mm
//! thick, clamped at one end; driven through its electrodes as an actuator, and bent by a tip
//! load as a sensor with its top electrode floating.

use super::piezo::*;
use gax::field::{Field, SparseMap};
use gax::mesh::Topology;

/// The units the material's numbers are scaled by: millimetres, `10¹⁰` Pa, `10⁻⁸` F/m, and
/// the field, voltage, force and charge they imply (the field unit is `√(10¹⁰ / 10⁻⁸)`).
pub const LENGTH_UNIT: f64 = 1e-3;
pub const STRESS_UNIT: f64 = 1e10;
pub const PERMITTIVITY_UNIT: f64 = 1e-8;
pub const FIELD_UNIT: f64 = 1e9;
pub const VOLTAGE_UNIT: f64 = LENGTH_UNIT * FIELD_UNIT;
pub const FORCE_UNIT: f64 = STRESS_UNIT * LENGTH_UNIT * LENGTH_UNIT;

pub const LENGTH: f64 = 30.0;
pub const THICKNESS: f64 = 1.0;
pub const WIDTH: f64 = 5.0;
pub const N_LENGTH: usize = 180;
pub const N_THICKNESS: usize = 12;
pub const APPLIED_VOLTAGE: f64 = 100.0 / VOLTAGE_UNIT;
pub const APPLIED_FORCE: f64 = 0.06 / FORCE_UNIT;
/// The drawing's exaggeration of the displacements, and the strain at full colour.
pub const DISPLACEMENT_SCALE: f64 = 100.0;
pub const STRAIN_SCALE: f64 = 2e-5;
/// The sensor's voltage drawn at full colour.
pub const SENSOR_VOLTAGE: f64 = 10.0 / VOLTAGE_UNIT;

/// PZT-5A under plane stress, the second material plane across its poling direction: the
/// reduced stress-charge constants of the compliance data of Table A3 of
/// <https://dael.euracoustics.org/bin/EAA/aaua_dl?document_id=52105>, with no stress through
/// the width and the permittivity at fixed in-plane strain.
pub fn pzt5a(directions: [Plane; 2]) -> Material {
    constitutive(
        directions,
        &Constants {
            stretch_stiffness: [73.38258e9 / STRESS_UNIT, 64.01459e9 / STRESS_UNIT],
            cross_stiffness: 28.18203e9 / STRESS_UNIT,
            shear_stiffness: 21.05263e9 / STRESS_UNIT,
            stretch_polarization: [
                -2.008341 * FIELD_UNIT / STRESS_UNIT,
                19.12233 * FIELD_UNIT / STRESS_UNIT,
            ],
            shear_polarization: 12.29474 * FIELD_UNIT / STRESS_UNIT,
            permittivities: [
                8.137619e-9 / PERMITTIVITY_UNIT,
                7.556942e-9 / PERMITTIVITY_UNIT,
            ],
        },
    )
}

/// A strip's mesh: its triangles, and the vertices of its clamped end, its tip, its bottom and
/// top electrodes and its bonded interface, each in order along it.
pub struct Strip {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    /// Each triangle's poling sign: down in the lower layer, up in the upper.
    pub poling: Vec<f64>,
    pub clamped: Vec<usize>,
    pub tip: Vec<usize>,
    pub bottom: Vec<usize>,
    pub top: Vec<usize>,
    pub interface: Vec<usize>,
}

/// A rectangular grid split into alternating triangles, the material's interface halfway up.
pub fn strip(length: f64, thickness: f64, n_length: usize, n_thickness: usize) -> Strip {
    let rows = n_thickness + 1;
    let node = |i: usize, j: usize| i * rows + j;
    let vertices = (0..=n_length)
        .flat_map(|i| {
            (0..rows).map(move |j| {
                point(
                    length * i as f64 / n_length as f64,
                    -thickness / 2.0 + thickness * j as f64 / n_thickness as f64,
                )
            })
        })
        .collect();
    let mut faces = Vec::new();
    let mut poling = Vec::new();
    for i in 0..n_length {
        for j in 0..n_thickness {
            let (ll, lr, ul, ur) = (
                node(i, j),
                node(i + 1, j),
                node(i, j + 1),
                node(i + 1, j + 1),
            );
            let odd = (i + j) % 2 == 1;
            faces.push([ll, lr, if odd { ul } else { ur }]);
            faces.push([if odd { lr } else { ll }, ur, ul]);
            let layer = if j < n_thickness / 2 { -1.0 } else { 1.0 };
            poling.extend([layer, layer]);
        }
    }
    let mesh = Topology::new((n_length + 1) * rows, &faces).expect("the strip");
    Strip {
        mesh,
        vertices,
        poling,
        clamped: (0..rows).map(|j| node(0, j)).collect(),
        tip: (0..rows).map(|j| node(n_length, j)).collect(),
        bottom: (0..=n_length).map(|i| node(i, 0)).collect(),
        top: (0..=n_length).map(|i| node(i, n_thickness)).collect(),
        interface: (0..=n_length).map(|i| node(i, n_thickness / 2)).collect(),
    }
}

impl Strip {
    /// The free displacements and the voltages: each interior node reads one of its own, and
    /// the `floating` electrode's nodes all read the one after them.
    pub fn boundary(&self, floating: &[usize]) -> Boundary {
        let n = self.mesh.len(0);
        let free: Vec<usize> = (0..n).filter(|v| !self.clamped.contains(v)).collect();
        let movement = SparseMap::from_triples(
            free.iter()
                .enumerate()
                .map(|(k, &v)| (v, k, Point::slot()))
                .collect::<Vec<_>>(),
            (n, free.len()),
        );
        let interior: Vec<usize> = (0..n)
            .filter(|v| !self.top.contains(v) && !self.bottom.contains(v))
            .collect();
        let voltages = interior.len() + usize::from(!floating.is_empty());
        let triples: Vec<_> = interior
            .iter()
            .enumerate()
            .map(|(k, &v)| (v, k))
            .chain(floating.iter().map(|&v| (v, interior.len())))
            .map(|(v, k)| (v, k, Scalar::slot()))
            .collect();
        Boundary {
            movement,
            voltage: SparseMap::from_triples(triples, (n, voltages)),
        }
    }

    /// The top electrode at `voltage`, every other node at zero.
    pub fn electrode_potential(&self, voltage: f64) -> Field<Scalar> {
        Field::from_fn(self.mesh.len(0), |v| {
            Scalar::new(if self.top.contains(&v) { voltage } else { 0.0 })
        })
    }

    /// A force along `y` at the tip, shared by its nodes, the two at its corners taking half.
    pub fn tip_load(&self, force: f64) -> Field<Plane> {
        let weight = |k: usize| {
            if k == 0 || k + 1 == self.tip.len() {
                0.5
            } else {
                1.0
            }
        };
        let total: f64 = (0..self.tip.len()).map(weight).sum();
        let mut load = Field::<Plane>::zeros(self.mesh.len(0));
        for (k, &v) in self.tip.iter().enumerate() {
            load.sites_mut()[v] = y() * (force * weight(k) / total);
        }
        load
    }

    /// The mean deflection of the tip, its corners weighted by half.
    #[cfg(test)]
    pub fn tip_motion(&self, displacement: &Field<Point>) -> f64 {
        let weight = |k: usize| {
            if k == 0 || k + 1 == self.tip.len() {
                0.5
            } else {
                1.0
            }
        };
        let total: f64 = (0..self.tip.len()).map(weight).sum();
        self.tip
            .iter()
            .enumerate()
            .map(|(k, &v)| (y() & displacement.site(v)).s() * weight(k))
            .sum::<f64>()
            / total
    }
}

/// The device: the strip, its triangles' areas and gradients, the material and its blocks.
pub struct Device {
    pub strip: Strip,
    #[cfg_attr(not(test), allow(dead_code))] // the tests reassemble with them
    pub areas: Field<Scalar>,
    pub gradients: [Field<Plane>; 3],
    #[cfg_attr(not(test), allow(dead_code))]
    pub material: Material,
    pub blocks: Blocks,
}

impl Device {
    pub fn new(
        length: f64,
        thickness: f64,
        n_length: usize,
        n_thickness: usize,
        width: f64,
    ) -> Self {
        let strip = strip(length, thickness, n_length, n_thickness);
        let (areas, gradients) = triangles(&strip.mesh, &strip.vertices);
        let material = pzt5a([x(), y()]);
        let blocks = assemble(
            &strip.mesh,
            &areas,
            &gradients,
            &material,
            &strip.poling,
            width,
        );
        Device {
            strip,
            areas,
            gradients,
            material,
            blocks,
        }
    }

    /// The equilibrium with the top electrode at `voltage` (or floating) and a tip load.
    pub fn solve(
        &self,
        voltage: f64,
        force: f64,
        floating: bool,
    ) -> (Field<Point>, Field<Scalar>, Field<Amount>) {
        let floating = if floating {
            self.strip.top.clone()
        } else {
            Vec::new()
        };
        let boundary = self.strip.boundary(&floating);
        let (displacement, potential) = equilibrium(
            &self.blocks,
            &boundary,
            &self.strip.electrode_potential(voltage),
            &self.strip.tip_load(force),
        );
        let charges = charges(&self.blocks, &displacement, &potential);
        (displacement, potential, charges)
    }
}
