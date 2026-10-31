//! The math: reproducing kernels and scalar skinning modes (numga's `weights.py`), the blended
//! affine motions, the Neo-Hookean material, the control-space mass and the potential with its
//! forces (numga's `core.py`). No drawing.

use gax::field::{Field, SparseMap};
use gax::vga3d;

/// A scalar (numga's `Scalar+`).
pub type Scalar<S = ()> = vga3d::Scalar<S>;
/// A point, a displacement, a control: an antivector, the Hodge dual of its vector (numga's
/// `Antivector-`).
pub type Antivector<S = ()> = vga3d::intrinsic::Bivector<S>;
/// A gradient or a force: a covector, a vector, read by pairing with a direction (numga's
/// `Vector+`).
pub type Vector<S = ()> = vga3d::Vector<S>;
/// A plane as the join of the two directions that span it (numga's `Span = Antibivector-`).
pub type Span<S = ()> = vga3d::intrinsic::Vector<S>;
/// A point with an intrinsic one prepended, on which affine functions are linear: grades 0 and
/// 2, intrinsic (numga's `AffinePoint`).
pub type AffinePoint<S = ()> = vga3d::intrinsic::Rotor<S>;
/// The local deformation, a map on directions.
pub type Deformation = Antivector<(Antivector,)>;
/// The stress, from gradients to forces.
pub type Stress = Vector<(Vector,)>;
/// The mass, from accelerations to forces.
pub type Mass = Vector<(Antivector,)>;

/// The point at `(x, y, z)`: the Hodge dual of its vector.
pub fn point(x: f64, y: f64, z: f64) -> Antivector {
    Vector::new(x, y, z).hodge()
}

/// The planes across the axes, `x`, `y` and `z` (numga's `planes`).
pub fn planes() -> [Vector; 3] {
    [
        Vector::new(1.0, 0.0, 0.0),
        Vector::new(0.0, 1.0, 0.0),
        Vector::new(0.0, 0.0, 1.0),
    ]
}

/// The unit directions along the axes, their duals (numga's `directions`).
pub fn directions() -> [Antivector; 3] {
    planes().map(|p| p.hodge())
}

/// A point's coordinates, read by pairing with the planes across the axes.
pub fn coordinates(p: Antivector) -> [f64; 3] {
    planes().map(|plane| (plane & p).s())
}

/// The intrinsic one.
pub fn one() -> vga3d::intrinsic::Scalar {
    vga3d::intrinsic::Scalar::new(1.0)
}

/// The reference volume, the join of the three axis directions: an intrinsic scalar of value
/// one.
pub fn unit_volume() -> vga3d::intrinsic::Scalar {
    let [x, y, z] = directions();
    x & y & z
}

/// The Euclidean pairing of two directions, positive (numga's `dot`, `⟨a ~b⟩₀`; gax's `|` of
/// bivectors is negative).
pub fn dot(a: Antivector, b: Antivector) -> f64 {
    a.scalar_product(b.reverse()).s()
}

/// The point embedded among the affine points: the intrinsic one plus it.
fn affine(p: Antivector) -> AffinePoint {
    one().cast::<AffinePoint>() + p.cast::<AffinePoint>()
}

/// Maps from scalar nodal values to values and spatial gradients at query points: each node's
/// weight at each query, and its gradient, dense `[queries <- nodes]`.
///
/// The Gaussians about the nodes are corrected so that their weighted sum reproduces every
/// affine function: the moment of the affine points solves for the correction, and
/// differentiating the moment equation accounts for the correction changing with position.
pub fn kernels(
    queries: &[Antivector],
    centers: &[Antivector],
    radius: f64,
) -> (SparseMap<Scalar>, SparseMap<Vector>) {
    let points: Vec<AffinePoint> = centers.iter().map(|c| affine(*c)).collect();
    let n = centers.len();
    let mut weights = Vec::with_capacity(queries.len() * n);
    let mut gradients = Vec::with_capacity(queries.len() * n);
    for (q, &query) in queries.iter().enumerate() {
        let difference: Vec<Antivector> = centers.iter().map(|c| query - *c).collect();
        let gaussian: Vec<f64> = difference
            .iter()
            .map(|d| (-dot(*d, *d) / (radius * radius)).exp())
            .collect();
        let moment: AffinePoint<(AffinePoint,)> = points
            .iter()
            .zip(&gaussian)
            .map(|(p, g)| *p * p.scalar_product(AffinePoint::slot()) * *g)
            .sum();
        let correction = moment.solve(affine(query));
        let reproduction: Vec<f64> = points
            .iter()
            .map(|p| p.scalar_product(correction).s())
            .collect();
        // The change of each Gaussian along a direction.
        let changes: Vec<Scalar<(Antivector,)>> = difference
            .iter()
            .zip(&gaussian)
            .map(|(d, g)| {
                d.scalar_product(Antivector::slot().reverse()) * (g * -2.0 / (radius * radius))
            })
            .collect();
        let moment_change: AffinePoint<(Antivector,)> = (0..n)
            .map(|k| points[k] * changes[k] * reproduction[k])
            .sum();
        let identity = Antivector::slot().cast::<AffinePoint>();
        let correction_change = moment.solve(identity - moment_change);
        for k in 0..n {
            weights.push((q, k, Scalar::new(gaussian[k] * reproduction[k])));
            let derivative = changes[k] * reproduction[k]
                + points[k].scalar_product(correction_change) * gaussian[k];
            // The weight's gradient: the covector that reads its change along every direction.
            let gradient: Vector = planes()
                .iter()
                .zip(directions())
                .map(|(plane, d)| *plane * derivative.of(d).s())
                .sum();
            gradients.push((q, k, gradient));
        }
    }
    let shape = (queries.len(), n);
    (
        SparseMap::from_triples(weights, shape),
        SparseMap::from_triples(gradients, shape),
    )
}

/// The lowest scalar skinning modes, orthonormal under volume integration: the nodal fields
/// whose gradients, weighted by the material's stiffness, are smallest against their values.
pub fn modes(
    evaluation: &SparseMap<Scalar>,
    gradient: &SparseMap<Vector>,
    volumes: &[f64],
    lame_lambda: &[f64],
    lame_mu: &[f64],
    count: usize,
) -> (Vec<f64>, Vec<Field<Scalar>>) {
    // Assembling over the quadrature sites integrates all pairs of nodal responses together.
    let values = evaluation * Scalar::slot();
    let changes = gradient * Scalar::slot();
    let volume = SparseMap::from_diagonal(&volumes.iter().map(|v| Scalar::slot() * *v).collect());
    let mass = values.adjoint().compose(&volume.compose(&values));
    let weighted = SparseMap::from_diagonal(
        &(0..volumes.len())
            .map(|s| Vector::slot() * (volumes[s] * (lame_lambda[s] + 4.0 * lame_mu[s])))
            .collect(),
    );
    let stiffness = changes.adjoint().compose(&weighted.compose(&changes));
    stiffness.eigh(&mass, count)
}

/// The control corners' barycentric gradients: each the patch of the face opposite its corner
/// over the cone that face spans to the corner, so that it reads one from any other corner to
/// its own and nothing along the face.
pub fn affine_gradients(corners: &[Antivector; 4]) -> [Vector; 4] {
    core::array::from_fn(|k| {
        let [a, b, c] = [1, 2, 3].map(|i| corners[(k + i) % 4]);
        let patch = (b - a) & (c - a);
        let cone = (corners[k] - a) & patch;
        patch * cone.inverse()
    })
}

/// The displacement map and the feature gradients of four affine controls per weight mode,
/// `[samples <- controls]`, the controls ordered mode by mode, corner by corner. Each feature is
/// a mode's weight times a corner's barycentric weight.
pub fn deformation_basis(
    queries: &[Antivector],
    centers: &[Antivector],
    radius: f64,
    modes: &[Field<Scalar>],
    corners: &[Antivector; 4],
) -> (SparseMap<Scalar>, SparseMap<Vector>) {
    let (evaluation, gradient) = kernels(queries, centers, radius);
    let weights: Vec<Field<Scalar>> = modes.iter().map(|m| &evaluation * m).collect();
    let weight_gradients: Vec<Field<Vector>> = modes.iter().map(|m| &gradient * m).collect();
    let affine = affine_gradients(corners);
    let mut features = Vec::new();
    let mut derivatives = Vec::new();
    for (s, &q) in queries.iter().enumerate() {
        let barycentric =
            core::array::from_fn::<f64, 4, _>(|k| 1.0 + (affine[k] & (q - corners[k])).s());
        for (m, (w, g)) in weights.iter().zip(&weight_gradients).enumerate() {
            let (w, g) = (w.site(s).s(), g.site(s));
            for k in 0..4 {
                let control = 4 * m + k;
                features.push((s, control, Scalar::new(w * barycentric[k])));
                derivatives.push((s, control, g * barycentric[k] + affine[k] * w));
            }
        }
    }
    let shape = (queries.len(), 4 * modes.len());
    (
        SparseMap::from_triples(features, shape),
        SparseMap::from_triples(derivatives, shape),
    )
}

/// The deformation of the blended affine motions at a sample: the identity plus each control
/// displacement paired with its feature's gradient there.
pub fn deformation(state: &Field<Antivector>, gradients: &[(usize, Vector)]) -> Deformation {
    gradients.iter().fold(Antivector::slot(), |local, &(c, g)| {
        local + state.site(c) * (g & Antivector::slot())
    })
}

/// The map on covectors that carries each plane, spanned by two directions, to the plane
/// spanned by their images, over the reference volume: the deformation's outermorphism onto
/// those joins, the determinant times the inverse transpose, singular or not.
pub fn cofactor(local: Deformation) -> Stress {
    let unit = unit_volume();
    let on_spans: Span<(Span,)> = local.outermorphism::<Span>();
    on_spans.of(Vector::slot() * unit) * unit.inverse()
}

/// Stable Neo-Hookean energy density and its stress, from gradients to forces.
pub fn constitutive(local: Deformation, lame_lambda: f64, lame_mu: f64) -> (f64, Stress) {
    let determinant = local.det();
    let bulk = lame_lambda + lame_mu;
    let target = 1.0 + lame_mu / bulk;
    let rest = bulk * (1.0 - target) * (1.0 - target) + 3.0 * lame_mu;
    // The stretch: the deformed metric read on reference directions, the Cauchy-Green form,
    // traced against the reference metric.
    let stretch: f64 = directions()
        .iter()
        .map(|d| {
            let image = local.of(*d);
            dot(image, image)
        })
        .sum();
    let swelling = determinant - target;
    let energy = (bulk * swelling * swelling + lame_mu * stretch - rest) / 2.0;
    // The stretch is metric, so its stress carries the deformation over to covectors through
    // the metric.
    let stress = local.adjoint().adjugate() * lame_mu + cofactor(local) * (bulk * swelling);
    (energy, stress)
}

/// The metric covector of a displacement, which pairs with it as the Euclidean dot (numga's
/// `dual`).
pub fn covector(d: Antivector) -> Vector {
    d.hodge()
}

/// The control-space mass, from control accelerations to forces on the controls: each sample's
/// displacement, turned into the covector the metric pairs with it, collected back onto the
/// controls by the sampling's adjugate.
pub fn mass(sampling: &SparseMap<Scalar>, sample_masses: &[f64]) -> SparseMap<Mass> {
    let displacing = sampling * Antivector::slot();
    let weighted = SparseMap::from_diagonal(
        &sample_masses
            .iter()
            .map(|m| Antivector::slot().hodge() * *m)
            .collect(),
    );
    displacing
        .adjugate()
        .compose(&weighted.compose(&displacing))
}

/// The samples a core takes at a time in the potential.
const BAND: usize = 64;

/// What the potential needs besides the state: the sampling, each sample's feature gradients
/// (the controls it depends on), the samples' volumes, the material, the clamp's stiffness at
/// each sample and the applied force density.
pub struct Body {
    pub sampling: SparseMap<Scalar>,
    /// The sampling's adjugate, from forces at the samples to forces on the controls.
    pub collecting: SparseMap<Vector<(Vector,)>>,
    pub gradients: Vec<Vec<(usize, Vector)>>,
    pub volumes: Vec<f64>,
    pub lame_lambda: f64,
    pub lame_mu: f64,
}

impl Body {
    pub fn new(
        sampling: SparseMap<Scalar>,
        gradients: &SparseMap<Vector>,
        volumes: Vec<f64>,
        lame_lambda: f64,
        lame_mu: f64,
    ) -> Body {
        let mut rows = vec![Vec::new(); gradients.shape().0];
        for (s, c, g) in gradients.couplings() {
            rows[s].push((c, g));
        }
        Body {
            collecting: (&sampling * Antivector::slot()).adjugate(),
            sampling,
            gradients: rows,
            volumes,
            lame_lambda,
            lame_mu,
        }
    }

    /// Elastic energy, a compliant base clamp and an applied force density, with their
    /// gradient, the forces on the controls.
    pub fn potential(
        &self,
        state: &Field<Antivector>,
        clamp: &[f64],
        load: &[Vector],
    ) -> (f64, Field<Vector>) {
        let displacement = &self.sampling * state;
        let controls = state.len();
        // The samples in bands, spread over the cores: each band's energy and forces on the
        // controls, added up in order.
        let bands: Vec<core::ops::Range<usize>> = (0..self.volumes.len())
            .step_by(BAND)
            .map(|start| start..(start + BAND).min(self.volumes.len()))
            .collect();
        let parts = gax_numga_examples::par_map(&bands, |band| {
            let mut energy = 0.0;
            let mut elastic = vec![Vector::zero(); controls];
            for s in band.clone() {
                let gradients = &self.gradients[s];
                let local = deformation(state, gradients);
                let (density, stress) = constitutive(local, self.lame_lambda, self.lame_mu);
                let d = displacement.site(s);
                let volume = self.volumes[s];
                let density = density + clamp[s] * dot(d, d) / 2.0 - (load[s] & d).s();
                energy += density * volume;
                for &(c, g) in gradients {
                    elastic[c] += stress.of(g) * volume;
                }
            }
            (energy, elastic)
        });
        let mut energy = 0.0;
        let mut elastic = vec![Vector::zero(); controls];
        for (e, forces) in parts {
            energy += e;
            for (total, f) in elastic.iter_mut().zip(forces) {
                *total += f;
            }
        }
        // The clamp pulls back along the metric covector of the displacement; the adjugate of
        // the sampling collects every local force onto the affine controls.
        let applied: Field<Vector> = (0..self.volumes.len())
            .map(|s| {
                let d = displacement.site(s);
                (covector(d) * clamp[s] - load[s]) * self.volumes[s]
            })
            .collect();
        let collected = self.collecting.of(&applied);
        (energy, &Field::new(elastic) + &collected)
    }
}
