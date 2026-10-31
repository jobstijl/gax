//! numga's scenes: a polynomial multivector field on a disk, a velocity of known parts on the
//! perforated disk, and a compressional pulse there.

use super::dirac::*;
use gax::field::Field;
use gax::mesh::Topology;

pub const DISK_DIVISIONS: usize = 9;
pub const DISK_RADIUS: f64 = 1.0;
pub const INNER_RADIUS: f64 = 0.32;
pub const OUTER_RADIUS: f64 = 1.0;
pub const ANNULUS_RINGS: usize = 15;
/// The harmonic part's strength in the decomposed velocity.
pub const CIRCULATION: f64 = 0.65;
/// The pulse: its centre, width, the time step and the steps, and the steps between frames.
pub const PULSE_CENTRE: [f64; 2] = [-0.58, 0.18];
pub const PULSE_WIDTH: f64 = 0.15;
pub const DT: f64 = 0.00533;
pub const STEPS: usize = 300;
pub const STRIDE: usize = 5;

/// A polynomial multivector field: the scalar `a (x²) + b (y²) + c x + d y + e x y`, the vector
/// `(p x + q y, r x + s y)` and the bivector `(f x y + g x + h y) e12`.
pub struct Fields {
    pub scalar: [f64; 5],
    pub vector: [[f64; 2]; 2],
    pub bivector: [f64; 3],
}

impl Fields {
    pub fn scalar(&self, x: f64, y: f64) -> f64 {
        let [a, b, c, d, e] = self.scalar;
        a * x * x + b * y * y + c * x + d * y + e * x * y
    }

    pub fn vector(&self, x: f64, y: f64) -> Plane {
        let [[p, q], [r, s]] = self.vector;
        Plane::new(p * x + q * y, r * x + s * y)
    }

    pub fn bivector(&self, x: f64, y: f64) -> f64 {
        let [f, g, h] = self.bivector;
        f * x * y + g * x + h * y
    }

    pub fn full(&self, x: f64, y: f64) -> Full {
        let u = self.vector(x, y);
        Full::new(self.scalar(x, y), u.e1(), u.e2(), self.bivector(x, y))
    }
}

/// numga's scene: `φ = (x² − y²) / 2 + 0.35 x`, `u` a turned and scaled position, `ω = x y − 0.35 y`.
pub const GRADES: Fields = Fields {
    scalar: [0.5, -0.5, 0.35, 0.0, 0.0],
    vector: [[0.6, -0.8], [0.8, 0.6]],
    bivector: [1.0, 0.0, -0.35],
};

/// The fields of the first figure: the potential, its gradient, the face product of the whole
/// vertex multivector, the even gradient of `φ + ω`, the edge cochain of `u` (its value at each
/// edge's midpoint on the edge), and the mixed Whitney reconstruction of the three cochains.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct Grades {
    pub potential: Field<Scalar>,
    pub scalar_gradient: Field<Plane>,
    pub face_dirac: Field<Full>,
    pub even_gradient: Field<Plane>,
    pub edge_cochain: Field<Scalar>,
    pub whitney: Field<Full>,
}

pub fn grades(
    ops: &Operators,
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    fields: &Fields,
) -> Grades {
    let at = |f: &dyn Fn(f64, f64) -> Full| {
        vertices.map(|p| {
            let [x, y] = coordinates(p);
            f(x, y)
        })
    };
    let multivector = at(&|x, y| fields.full(x, y));
    let potential = multivector.map(|m| m.grade::<0>());
    let even = multivector.map(|m| Even::new(m.s(), m.e12()));
    let midpoints = &mesh.means::<Scalar>(1) * vertices;
    let steps = &mesh.d::<Scalar>(0) * vertices;
    let edge_cochain = midpoints.zip_with(&steps, |m, e| {
        let [x, y] = coordinates(m);
        fields.vector(x, y) & e
    });
    let scalar_part = ops.whitney0.of(&potential);
    let forms = ops.whitney1.of(&edge_cochain);
    let faces = ops.whitney2.of(&(&ops.d1 * &edge_cochain));
    let whitney = Field::from_fn(mesh.len(2), |f| {
        let (s, v, b) = (scalar_part.site(f), forms.site(f), faces.site(f));
        Full::new(s.s(), v.e1(), v.e2(), b.e12())
    });
    Grades {
        scalar_gradient: ops.gradient.of(&potential),
        face_dirac: ops.dirac.of(&multivector),
        even_gradient: ops.even_gradient.of(&even),
        potential,
        edge_cochain,
        whitney,
    }
}

/// The states of a compressional pulse, a Gaussian of the even field's scalar part about
/// `PULSE_CENTRE` with no face field, stepped `steps` times by `dt`.
pub fn pulse(
    ops: &Operators,
    vertices: &Field<Point>,
    width: f64,
    dt: f64,
    steps: usize,
) -> Vec<(Field<Even>, Field<Plane>)> {
    let centre = point(PULSE_CENTRE[0], PULSE_CENTRE[1]);
    let falloff = -0.5 / (width * width);
    let envelope = vertices.map(|p| {
        let [x, y] = coordinates(p - centre);
        Even::new((falloff * (x * x + y * y)).exp(), 0.0)
    });
    let faces = Field::zeros(ops.mesh.len(2));
    propagate(ops, envelope, faces, dt, steps)
}
